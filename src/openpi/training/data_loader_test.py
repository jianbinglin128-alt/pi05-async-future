import dataclasses

import jax
import jax.numpy as jnp

from openpi import transforms
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_split_action_window_preserves_context_target_alignment():
    window = jnp.arange(10 * 7, dtype=jnp.float32).reshape(10, 7)

    for context_steps in (0, 5, 9):
        result = _data_loader.split_action_window(
            window,
            action_horizon=10,
            context_horizon=9,
            context_steps=context_steps,
        )

        assert result["action_context"].shape == (9, 7)
        assert result["action_context_mask"].sum() == context_steps
        assert jnp.array_equal(result["action_context"][:context_steps], window[:context_steps])
        assert jnp.array_equal(result["action_context"][context_steps:], jnp.zeros((9 - context_steps, 7)))
        assert jnp.array_equal(result["actions"], window)


def test_pack_action_context_preserves_variable_training_mask():
    window = jnp.arange(10 * 7, dtype=jnp.float32).reshape(10, 7)
    packed = _data_loader.split_action_window(window, action_horizon=10, context_horizon=9, context_steps=5)
    packed = transforms.PackActionContext(context_horizon=9, model_action_dim=7)(packed)

    assert packed["action_context_mask"].sum() == 5
    assert jnp.array_equal(packed["action_context"][:5], window[:5])
    assert jnp.array_equal(packed["action_context"][5:], jnp.zeros((4, 7)))
    assert jnp.array_equal(packed["actions"], window)


def test_action_context_attention_block_mask():
    # Prefix (2), context (4 slots, only first 2 valid), target (3).
    input_mask = jnp.array([[True, True, True, True, False, False, True, True, True]])
    ar_mask = jnp.array([False, False, True, False, False, False, True, False, False])
    mask = make_attn_mask(input_mask, ar_mask)[0]

    assert mask[2, 0]
    assert mask[2, 3]
    assert not mask[2, 6]
    assert not mask[4].any()
    assert mask[6, 0]
    assert mask[6, 2]
    assert mask[6, 3]
    assert mask[6, 7]
    assert mask[7, 6]
