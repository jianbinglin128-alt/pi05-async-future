import flax.nnx as nnx
import jax

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_async_context_freeze_filter_keeps_only_action_side_parameters_trainable():
    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        action_context_horizon=5,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    frozen = nnx.state(model, nnx.All(nnx.Param, config.get_async_context_freeze_filter())).flat_state()
    trainable = nnx.state(model, nnx.All(nnx.Param, nnx.Not(config.get_async_context_freeze_filter()))).flat_state()
    frozen_paths = ["/".join(map(str, path)) for path in frozen]
    trainable_paths = ["/".join(map(str, path)) for path in trainable]

    assert any("action_context_in_proj" in path for path in trainable_paths)
    assert any("action_in_proj" in path for path in trainable_paths)
    assert any("action_out_proj" in path for path in trainable_paths)
    assert all("img" in path or ("llm" in path and "_1" not in path) for path in frozen_paths)
    assert all("llm" in path and "_1" in path for path in trainable_paths if "llm" in path)
