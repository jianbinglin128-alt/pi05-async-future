import pytest
import torch
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from openpi.models_pytorch.reflex_cache import StreamingPrefixCache


def _full_prefix_cache(model, language, frames, positions):
    blocks = [language, *frames]
    total = sum(block.shape[1] for block in blocks)
    allowed = torch.zeros(total, total, dtype=torch.bool, device=language.device)
    start = 0
    language_length = language.shape[1]
    for block in blocks:
        end = start + block.shape[1]
        allowed[start:end, :language_length] = True
        allowed[start:end, start:end] = True
        start = end
    mask = torch.where(allowed[None, None], 0.0, -2.3819763e38)
    result = model(
        inputs_embeds=torch.cat(blocks, dim=1),
        attention_mask=mask,
        position_ids=positions[None],
        use_cache=True,
    )
    assert total == positions.numel()
    return result.past_key_values


@torch.no_grad()
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_incremental_prefix_matches_full_block_attention_after_eviction(device):
    torch.manual_seed(0)
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
    language, first, second, third = [torch.randn(1, 2, 16, device=device) for _ in range(4)]
    valid = torch.ones(1, 2, dtype=torch.bool, device=device)
    cache = StreamingPrefixCache(model, max_frames=2)
    cache.initialize(torch.tensor([[1, 2]], device=device), valid, language)
    cache.append_frame(first, valid)
    cache.append_frame(second, valid)

    incremental, _, _ = cache.snapshot()
    full = _full_prefix_cache(model, language, [first, second], torch.arange(6, device=device))
    for index in range(config.num_hidden_layers):
        for actual, expected in zip(incremental[index], full[index], strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    cache.append_frame(third, valid)
    incremental, masks, next_position = cache.snapshot()
    full = _full_prefix_cache(model, language, [second, third], torch.tensor([0, 1, 4, 5, 6, 7], device=device))
    assert masks.shape == (1, 6)
    assert next_position.item() == 8
    for index in range(config.num_hidden_layers):
        for actual, expected in zip(incremental[index], full[index], strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_ring_and_snapshot_keep_addresses_and_isolate_decode():
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
    model = GemmaModel(config).eval()
    cache = StreamingPrefixCache(model, 2)
    valid = torch.ones(1, 2, dtype=torch.bool)
    cache.initialize(torch.tensor([[1, 2]]), valid, torch.randn(1, 2, 16))
    cache.append_frame(torch.randn(1, 2, 16), valid)
    snapshot, mask, _ = cache.snapshot()
    ring_address = cache.storage.data_ptr()
    snapshot_address = snapshot[0][0].data_ptr()
    old = snapshot[0][0].clone()
    assert mask.tolist() == [[True, True, True, True, False, False]]
    for _ in range(5):
        cache.append_frame(torch.randn(1, 2, 16), valid)
        assert cache.storage.data_ptr() == ring_address
        torch.testing.assert_close(snapshot[0][0], old)
    new_snapshot, _, _ = cache.snapshot()
    assert new_snapshot[0][0].data_ptr() == snapshot_address
    assert not torch.equal(new_snapshot[0][0], old)


@torch.no_grad()
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_frame_overwrite_matches_fresh_cache_with_fixed_positions(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(12)
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
    tokens = torch.tensor([[1, 2, 0], [3, 0, 0]], device=device)
    language_mask = tokens != 0
    language = torch.randn(2, 3, 16, device=device)
    cache = StreamingPrefixCache(model, 1)
    cache.initialize(tokens, language_mask, language)
    addresses = None
    for index in range(5):
        frame = torch.randn(2, 3, 16, device=device)
        masks = torch.tensor([[True, True, index % 2 == 0], [True, False, False]], device=device)
        cache.append_frame(frame, masks)
        actual, actual_mask, position = cache.snapshot()
        fresh = StreamingPrefixCache(model, 1)
        fresh.initialize(tokens, language_mask, language)
        fresh.append_frame(frame, masks)
        expected, expected_mask, expected_position = fresh.snapshot()
        torch.testing.assert_close(position, expected_position, atol=0, rtol=0)
        torch.testing.assert_close(actual_mask, expected_mask)
        for layer in range(config.num_hidden_layers):
            for value, reference in zip(actual[layer], expected[layer], strict=True):
                torch.testing.assert_close(value, reference, atol=0, rtol=0)
        current_addresses = cache.storage.data_ptr(), actual[0][0].data_ptr()
        if addresses is not None:
            assert current_addresses == addresses
        addresses = current_addresses
