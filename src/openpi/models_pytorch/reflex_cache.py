"""Bounded visual ring and immutable, static-address decode snapshots."""

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma import modeling_gemma


def project_qkv(attention, hidden):
    shape = (*hidden.shape[:2], -1, attention.head_dim)
    fused = getattr(attention, "qkv_proj", None)
    if fused is None:
        parts = (attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden))
    else:
        q_size = attention.config.num_attention_heads * attention.head_dim
        kv_size = attention.config.num_key_value_heads * attention.head_dim
        parts = fused(hidden).split((q_size, kv_size, kv_size), dim=-1)
    return tuple(part.view(shape).transpose(1, 2) for part in parts)


class StaticPrefixSnapshot(DynamicCache):
    """Own a prefix plus reusable suffix scratch; never append diffusion steps.

    Decode only writes the scratch region, so another stream can safely update
    the source ring. Graph inputs are copied here once per action chunk.
    """

    def __init__(self, keys, values, suffix_length):
        super().__init__()
        self.prefix_length = keys[0].shape[2]
        self.suffix_length = suffix_length
        self.storage = torch.empty(
            len(keys),
            2,
            *keys[0].shape[:2],
            self.prefix_length + suffix_length,
            keys[0].shape[-1],
            device=keys[0].device,
            dtype=keys[0].dtype,
        )
        self.key_cache = [self.storage[layer, 0, :, :, : self.prefix_length] for layer in range(len(keys))]
        self.value_cache = [self.storage[layer, 1, :, :, : self.prefix_length] for layer in range(len(keys))]
        self.copy_prefix(keys, values)

    def copy_prefix(self, keys, values):
        for layer, (key, value) in enumerate(zip(keys, values, strict=True)):
            self.key_cache[layer].copy_(key)
            self.value_cache[layer].copy_(value)

    def get_seq_length(self, layer_idx=0):
        return self.prefix_length

    def compose_suffix(self, key, value, layer_idx):
        if key.shape[2] != self.suffix_length:
            raise ValueError("Suffix length changed; create a new snapshot")
        self.storage[layer_idx, 0, :, :, self.prefix_length :].copy_(key)
        self.storage[layer_idx, 1, :, :, self.prefix_length :].copy_(value)
        return self.storage[layer_idx, 0], self.storage[layer_idx, 1]


class StreamingPrefixCache:
    """Pinned language plus preallocated visual slots; eviction moves no KV.

    Preserve the existing partitioned mask: frames attend to language and their
    own tokens. Only the action expert attends across the visual window.
    A single slot overwrites at fixed language-relative RoPE coordinates.
    """

    def __init__(self, model, max_frames: int):
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self.model = model
        self.max_frames = max_frames
        self.reset()

    def reset(self):
        self.tokens = None
        self.token_masks = None
        self.language_keys = None
        self.language_values = None
        self.storage = None
        self.pad_mask = None
        self.frame_length = None
        self.frame_count = 0
        self.next_slot = 0
        self.next_position = None
        self._snapshot = None

    def matches(self, tokens, masks):
        return self.tokens is not None and torch.equal(self.tokens, tokens) and torch.equal(self.token_masks, masks)

    def initialize(self, tokens, masks, embeddings):
        self.reset()
        self.tokens = tokens.detach().clone()
        self.token_masks = masks.detach().clone()
        self.next_position = torch.zeros(tokens.shape[0], 1, dtype=torch.long, device=tokens.device)
        self.language_keys, self.language_values = self._prefill(embeddings, masks, language=True)

    def append_frame(self, embeddings, masks):
        if self.tokens is None:
            raise RuntimeError("Initialize the language prefix before appending frames")
        if self.frame_length is None:
            self.frame_length = embeddings.shape[1]
            first = self.language_keys[0]
            capacity = self.token_masks.shape[1] + self.max_frames * self.frame_length
            self.storage = torch.zeros(
                len(self.model.layers),
                2,
                *first.shape[:2],
                capacity,
                first.shape[-1],
                dtype=first.dtype,
                device=first.device,
            )
            self.pad_mask = torch.zeros(masks.shape[0], capacity, dtype=torch.bool, device=masks.device)
            for layer, (key, value) in enumerate(zip(self.language_keys, self.language_values, strict=True)):
                self.storage[layer, 0, :, :, : self.token_masks.shape[1]].copy_(key)
                self.storage[layer, 1, :, :, : self.token_masks.shape[1]].copy_(value)
        elif embeddings.shape[1] != self.frame_length:
            raise ValueError("The number of visual tokens changed within a stream")
        if self.max_frames == 1:
            # No visual KV survives this overwrite. Reuse the original frame's
            # coordinates relative to the pinned language, including padding.
            self.next_position = self.token_masks.long().sum(dim=1, keepdim=True)
        keys, values = self._prefill(embeddings, masks, language=False)
        start = self.token_masks.shape[1] + self.next_slot * self.frame_length
        end = start + self.frame_length
        for layer, (key, value) in enumerate(zip(keys, values, strict=True)):
            self.storage[layer, 0, :, :, start:end].copy_(key)
            self.storage[layer, 1, :, :, start:end].copy_(value)
        self.pad_mask[:, start:end].copy_(masks)
        self.pad_mask[:, : self.token_masks.shape[1]].copy_(self.token_masks)
        self.frame_count = min(self.frame_count + 1, self.max_frames)
        self.next_slot = (self.next_slot + 1) % self.max_frames

    def _prefill(self, embeddings, masks, *, language):
        batch, count = masks.shape
        positions = self.next_position + torch.cumsum(masks.long(), dim=1) - 1
        self.next_position = self.next_position + masks.long().sum(dim=1, keepdim=True)
        all_masks = masks if language else torch.cat((self.token_masks, masks), dim=1)
        attention_mask = torch.where(masks[:, None, :, None] & all_masks[:, None, None, :], 0.0, -2.3819763e38)
        hidden = embeddings.to(self.model.layers[0].self_attn.o_proj.weight.dtype)
        cos, sin = self.model.rotary_emb(hidden, positions)
        keys, values = [], []
        for index, layer in enumerate(self.model.layers):
            residual = hidden
            normalized, gate = layer.input_layernorm(hidden)
            query, key, value = project_qkv(layer.self_attn, normalized)
            query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin)
            keys.append(key)
            values.append(value)
            if not language:
                key = torch.cat((self.language_keys[index], key), dim=2)
                value = torch.cat((self.language_values[index], value), dim=2)
            output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn,
                query,
                key,
                value,
                attention_mask,
                layer.self_attn.scaling,
            )
            output = layer.self_attn.o_proj(output.reshape(batch, count, -1).to(layer.self_attn.o_proj.weight.dtype))
            hidden = modeling_gemma._gated_residual(residual, output, gate)  # noqa: SLF001
            residual = hidden
            hidden, gate = layer.post_attention_layernorm(hidden)
            hidden = modeling_gemma._gated_residual(residual, layer.mlp(hidden), gate)  # noqa: SLF001
        return keys, values

    def snapshot(self):
        """Copy into stable chronological storage once per chunk, including masked empty slots."""
        if self.storage is None:
            raise RuntimeError("Append a frame before taking a snapshot")
        if self._snapshot is None:
            self._snapshot = torch.empty_like(self.storage)
            self._snapshot_masks = torch.empty_like(self.pad_mask)
            self._snapshot_position = torch.empty_like(self.next_position)
            self._snapshot_cache = DynamicCache()
            for index in range(len(self.model.layers)):
                self._snapshot_cache.update(self._snapshot[index, 0], self._snapshot[index, 1], index)
        lang_length = self.token_masks.shape[1]
        self._snapshot[..., :lang_length, :].copy_(self.storage[..., :lang_length, :])
        self._snapshot_masks[:, :lang_length].copy_(self.token_masks)
        slots = list(range(self.max_frames))
        if self.frame_count == self.max_frames:
            slots = slots[self.next_slot :] + slots[: self.next_slot]
        for target, source in enumerate(slots):
            start, end = lang_length + source * self.frame_length, lang_length + (source + 1) * self.frame_length
            dest = lang_length + target * self.frame_length
            self._snapshot[..., dest : dest + self.frame_length, :].copy_(self.storage[..., start:end, :])
            self._snapshot_masks[:, dest : dest + self.frame_length].copy_(self.pad_mask[:, start:end])
        self._snapshot_position.copy_(self.next_position)
        return self._snapshot_cache, self._snapshot_masks, self._snapshot_position
