"""CUDA graph replay and a vision producer isolated from action decoding."""

from concurrent.futures import ThreadPoolExecutor
import time

import torch

from openpi.models_pytorch.reflex_cache import StaticPrefixSnapshot


class DenoisingRunner:
    """One bounded graph per current shape, with all mutable inputs copied in.

    A different shape invalidates the graph. A different observation only updates
    state, prefix, mask, position and noise buffers; it never recaptures a graph.
    """

    def __init__(self, step_fn):
        self.step_fn = step_fn
        self.entry = None
        self.captures = 0
        self.replays = 0

    def prepare(self, state, mask, prefix, noise, offset, num_steps, *, cuda_graph,
                action_context=None, action_context_mask=None):
        context_horizon = getattr(self.step_fn.__self__.config, "action_context_horizon", 0)
        if context_horizon and action_context is None:
            action_context = state.new_zeros((state.shape[0], context_horizon, state.shape[-1]))
            action_context_mask = torch.zeros(action_context.shape[:2], device=state.device, dtype=torch.bool)
        key = (
            tuple(state.shape),
            tuple(mask.shape),
            tuple(noise.shape),
            noise.device,
            noise.dtype,
            state.dtype,
            prefix[0][0].dtype,
            num_steps,
            cuda_graph,
            context_horizon,
        )
        entry = self.entry
        if entry is None or entry["key"] != key:
            suffix_length = noise.shape[1] + context_horizon + (0 if self.step_fn.__self__.pi05 else 1)
            entry = {
                "key": key,
                "state": torch.empty_like(state),
                "mask": torch.empty_like(mask),
                "noise": torch.empty_like(noise),
                "offset": torch.empty_like(offset),
                "prefix": StaticPrefixSnapshot(
                    [pair[0] for pair in prefix],
                    [pair[1] for pair in prefix],
                    suffix_length,
                ),
                "graph": None,
            }
            if context_horizon:
                entry["action_context"] = torch.empty_like(action_context)
                entry["action_context_mask"] = torch.empty_like(action_context_mask)
            self.entry = entry
        entry["state"].copy_(state)
        entry["mask"].copy_(mask)
        entry["noise"].copy_(noise)
        entry["offset"].copy_(offset)
        if context_horizon:
            entry["action_context"].copy_(action_context)
            entry["action_context_mask"].copy_(action_context_mask)
        entry["prefix"].copy_prefix([pair[0] for pair in prefix], [pair[1] for pair in prefix])
        return entry

    def run(self, entry, num_steps, *, cuda_graph):
        def denoise():
            x_t = entry["noise"]
            dt = -1.0 / num_steps
            for step in range(num_steps):
                timestep = torch.full(
                    (x_t.shape[0],),
                    1.0 + step * dt,
                    dtype=torch.float32,
                    device=x_t.device,
                )
                velocity = self.step_fn(
                    entry["state"],
                    entry["mask"],
                    entry["prefix"],
                    x_t,
                    timestep,
                    prefix_position_offset=entry["offset"],
                    **({"action_context": entry["action_context"],
                        "action_context_mask": entry["action_context_mask"]}
                       if "action_context" in entry else {}),
                )
                x_t = x_t + dt * velocity
            return x_t

        device = entry["noise"].device
        if not cuda_graph or device.type != "cuda":
            return denoise()
        if entry["graph"] is None:
            caller = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    denoise()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                entry["output"] = denoise()
            caller.wait_stream(stream)
            entry["graph"] = graph
            self.captures += 1
        entry["graph"].replay()
        self.replays += 1
        # The graph owns its output. Do not expose an alias that next replay changes.
        return entry["output"].clone()


class VisionProducer:
    """Single writer; exceptions propagate through Future.result().

    Producer inputs are owned by the job until it finishes. A fresh CUDA event
    marks each publication; consumers never wait on a reused event. Decode owns
    a separate KV snapshot, so overwriting a full visual ring cannot race decode.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=device) if self.device.type == "cuda" else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflex-vision")
        self.pending = None
        self.last_producer_ms = None

    def submit(self, fn):
        if self.pending is not None:
            raise RuntimeError("Consume the previous visual update before submitting another")
        caller = torch.cuda.current_stream(self.device) if self.stream is not None else None
        ready = torch.cuda.Event() if self.stream is not None else None
        if ready is not None:
            ready.record(caller)

        def produce():
            with torch.no_grad():
                if self.stream is None:
                    start = time.perf_counter()
                    result = fn()
                    self.last_producer_ms = (time.perf_counter() - start) * 1000
                    return result, None
                with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                    self.stream.wait_event(ready)
                    start = torch.cuda.Event(enable_timing=True)
                    done = torch.cuda.Event(enable_timing=True)
                    start.record(self.stream)
                    result = fn()
                    done.record(self.stream)
                    # Keep all job inputs alive until their kernels have consumed them.
                    done.synchronize()
                    self.last_producer_ms = start.elapsed_time(done)
                    return result, done

        self.pending = self.executor.submit(produce)

    def consume(self):
        if self.pending is None:
            return None
        future, self.pending = self.pending, None
        result, done = future.result()
        if done is not None:
            torch.cuda.current_stream(self.device).wait_event(done)
        return result

    def close(self):
        try:
            self.consume()
        finally:
            self.executor.shutdown(wait=True)
