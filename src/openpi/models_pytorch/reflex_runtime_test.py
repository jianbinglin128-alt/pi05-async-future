import dataclasses

import pytest
import torch
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.reflex_cache import StaticPrefixSnapshot
from openpi.models_pytorch.reflex_runtime import VisionProducer


@torch.no_grad()
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gemma_fusion_and_static_suffix_match_original(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(4)
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    model = GemmaModel(config).to(device).eval()
    prefix = torch.randn(1, 4, 16, device=device)
    suffix = torch.randn(1, 3, 16, device=device)
    mask = torch.zeros(1, 1, 3, 7, device=device)
    prefix_kv = model(
        inputs_embeds=prefix, attention_mask=torch.zeros(1, 1, 4, 4, device=device), use_cache=True
    ).past_key_values
    expected = model(
        inputs_embeds=suffix, attention_mask=mask, past_key_values=prefix_kv, use_cache=False
    ).last_hidden_state
    for layer in model.layers:
        layer.self_attn.fuse_qkv()
        layer.mlp.fuse_gate_up()
    static = StaticPrefixSnapshot([pair[0] for pair in prefix_kv], [pair[1] for pair in prefix_kv], 3)
    original_prefix = static[0][0].clone()
    for _ in range(3):
        actual = model(
            inputs_embeds=suffix, attention_mask=mask, past_key_values=static, use_cache=False
        ).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(static[0][0], original_prefix)
        assert static.get_seq_length() == 4


@pytest.fixture(scope="module")
def small_policy_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(8)
    config = Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pytorch_compile_mode=None,
        pi05=True,
        action_dim=7,
        action_horizon=3,
        max_token_len=4,
        dtype="float32",
    )
    model = PI0Pytorch(config).cuda().eval()
    # PI0 enables TF32 globally. Use full FP32 for this numerical equivalence
    # test: cuBLAS can choose different TF32 kernels on the capture stream.
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    model.fuse_projection_layers()
    yield model
    model.close()
    torch.set_float32_matmul_precision(previous_precision)


def observation(seed=1):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return model_lib.Observation(
        images={
            name: torch.rand(1, 3, 224, 224, device="cuda", generator=generator)
            for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        image_masks={
            name: torch.ones(1, device="cuda", dtype=torch.bool)
            for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        state=torch.rand(1, 7, device="cuda", generator=generator),
        tokenized_prompt=torch.tensor([[1, 2, 3, 0]], device="cuda"),
        tokenized_prompt_mask=torch.tensor([[True, True, True, False]], device="cuda"),
    )


@torch.no_grad()
def test_graph_replay_uses_changed_inputs_and_does_not_mutate_returned_actions(small_policy_model):
    model = small_policy_model
    first, second = observation(1), observation(2)
    noise = torch.randn(1, 3, 7, device="cuda")
    expected = model.sample_actions("cuda", first, noise=noise, num_steps=3, cuda_graph=False)
    actual = model.sample_actions("cuda", first, noise=noise, num_steps=3, cuda_graph=True)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    before = actual.clone()
    captures = model._denoising_runner.captures  # noqa: SLF001
    changed = model.sample_actions("cuda", second, noise=noise + 1, num_steps=3, cuda_graph=True)
    assert model._denoising_runner.captures == captures  # noqa: SLF001
    torch.testing.assert_close(actual, before)
    expected_changed = model.sample_actions("cuda", second, noise=noise + 1, num_steps=3, cuda_graph=False)
    torch.testing.assert_close(changed, expected_changed, atol=1e-5, rtol=1e-5)
    assert not torch.equal(actual, changed)


@torch.no_grad()
def test_streaming_concurrency_reset_and_prompt_change(small_policy_model):
    model = small_policy_model
    model.reset_streaming_cache()
    for seed in (3, 4, 5, 6):
        actions = model.sample_actions_streaming(
            "cuda",
            observation(seed),
            num_steps=2,
            streaming_window=2,
            cuda_graph=True,
            concurrent_vision=True,
        )
        assert actions.shape == (1, 3, 7)
        assert torch.isfinite(actions).all()
    changed = dataclasses.replace(observation(7), tokenized_prompt=torch.tensor([[4, 5, 6, 0]], device="cuda"))
    model.sample_actions_streaming(
        "cuda", changed, num_steps=2, streaming_window=2, cuda_graph=True, concurrent_vision=True
    )
    assert model._streaming_prefix.frame_count == 1  # noqa: SLF001
    model.reset_streaming_cache()
    assert model._streaming_prefix is None  # noqa: SLF001


@torch.no_grad()
def test_streaming_uses_current_frame_and_single_frame_positions(small_policy_model):
    model = small_policy_model
    noise = torch.randn(1, 3, 7, device="cuda")
    inputs = [observation(seed) for seed in (10, 11, 12, 12)]
    references = []
    for obs in inputs:
        model.reset_streaming_cache()
        references.append(
            model.sample_actions_streaming(
                "cuda",
                obs,
                noise=noise,
                num_steps=2,
                streaming_window=1,
                concurrent_vision=False,
                cuda_graph=False,
            )
        )
    assert not torch.equal(references[0], references[1])
    model.reset_streaming_cache()
    captures = None
    for obs, expected in zip(inputs, references, strict=True):
        actual = model.sample_actions_streaming(
            "cuda",
            obs,
            noise=noise,
            num_steps=2,
            streaming_window=1,
            concurrent_vision=True,
            cuda_graph=True,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        assert model._vision_producer.pending is None  # noqa: SLF001
        if captures is not None:
            assert model._denoising_runner.captures == captures  # noqa: SLF001
        captures = model._denoising_runner.captures  # noqa: SLF001
    model.reset_streaming_cache()


@torch.no_grad()
@pytest.mark.parametrize("worker", [False, True])
def test_streaming_split_timing_preserves_actions_and_separates_language_bootstrap(small_policy_model, worker):
    model = small_policy_model
    model.reset_streaming_cache()
    obs = observation(15)
    noise = torch.randn(1, 3, 7, device="cuda")
    kwargs = {
        "noise": noise,
        "num_steps": 2,
        "streaming_window": 1,
        "concurrent_vision": worker,
        "cuda_graph": True,
    }
    reference = model.sample_actions_streaming("cuda", obs, **kwargs)
    model.reset_streaming_cache()
    model.profile_inference = True
    try:
        for first in (True, False):
            actual = model.sample_actions_streaming("cuda", obs, **kwargs)
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
            timing = model.last_inference_timing
            assert timing["vision_encoder_ms"] > 0
            assert timing["visual_prefill_ms"] > 0
            assert timing["language_init_ms"] > 0 if first else timing["language_init_ms"] == 0
            assert timing["llm_prefill_ms"] == timing["language_init_ms"] + timing["visual_prefill_ms"]
            assert timing["vlm_total_ms"] == timing["llm_ms"]
            assert timing["vision_encoder_ms"] + timing["llm_prefill_ms"] <= timing["vlm_total_ms"] + 0.1
    finally:
        model.profile_inference = False
        model.reset_streaming_cache()


@torch.no_grad()
def test_state_conditioning_is_identity_initially_and_trainable(small_policy_model):
    model = small_policy_model
    state, noisy, timestep = (
        torch.randn(2, 7, device="cuda"),
        torch.randn(2, 3, 7, device="cuda"),
        torch.ones(2, device="cuda"),
    )
    before = model.embed_suffix(state, noisy, timestep)[-1]
    model.enable_state_conditioning()
    after = model.embed_suffix(state, noisy, timestep)[-1]
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    model.state_cond_mlp_out.weight.fill_(0.05)
    altered = model.embed_suffix(state + 1, noisy, timestep)[-1]
    assert not torch.equal(after, altered)


def test_vision_worker_propagates_failures_and_closes():
    producer = VisionProducer("cpu")
    try:

        def fail():
            raise ValueError("camera failed")

        producer.submit(fail)
        with pytest.raises(ValueError, match="camera failed"):
            producer.consume()
        producer.submit(lambda: 17)
        assert producer.consume() == 17
    finally:
        producer.close()
