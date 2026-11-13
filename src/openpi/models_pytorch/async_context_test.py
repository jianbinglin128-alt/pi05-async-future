import dataclasses

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch.reflex_runtime_test import observation


def test_context_uses_action_normalization_and_masks_unused_slots():
    stats = transforms.NormStats(mean=np.arange(7), std=np.full(7, 2.0),
        q01=-np.ones(7), q99=np.ones(7))
    actions = np.arange(14).reshape(2, 7).astype(np.float32)
    normalize = transforms.Normalize({'actions': stats})
    data = normalize({'actions': actions.copy(), 'action_context': actions.copy()})
    np.testing.assert_array_equal(data['actions'], data['action_context'])
    packed = transforms.PackActionContext(9, 32)(data)
    assert packed['action_context'].shape == (9, 32)
    np.testing.assert_array_equal(packed['action_context_mask'], [True, True] + [False]*7)
    np.testing.assert_array_equal(packed['action_context'][2:], 0)


@torch.no_grad()
def test_context_attention_blocks_and_graph_updates():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(5)
    config = Pi0Config(pi05=True, paligemma_variant='dummy', action_expert_variant='dummy',
        action_dim=7, action_horizon=3, max_token_len=4, dtype='float32',
        pytorch_compile_mode=None, action_context_horizon=9, discrete_state_input=False)
    model = PI0Pytorch(config).cuda().eval()
    torch.set_float32_matmul_precision('highest')
    obs = observation()
    context = torch.randn(1,9,7,device='cuda')
    mask = torch.tensor([[True,True]+[False]*7],device='cuda')
    obs = dataclasses.replace(obs,action_context=context,action_context_mask=mask)
    noise = torch.randn(1,3,7,device='cuda')
    try:
        _, valid, blocks, _ = model.embed_suffix(obs.state,noise,torch.ones(1,device='cuda'),context,mask)
        attention = make_att_2d_masks(valid,blocks)[0]
        assert attention[0,1] and not attention[0,9]
        assert attention[9,0] and attention[9,11] and not attention[9,2]
        eager = model.sample_actions('cuda',obs,noise=noise,num_steps=2)
        graph = model.sample_actions('cuda',obs,noise=noise,num_steps=2,cuda_graph=True)
        torch.testing.assert_close(graph,eager,atol=2e-5,rtol=2e-5)
        changed = dataclasses.replace(obs,action_context=context+1)
        captures = model._denoising_runner.captures
        graph2 = model.sample_actions('cuda',changed,noise=noise,num_steps=2,cuda_graph=True)
        assert model._denoising_runner.captures == captures
        expected = model.sample_actions('cuda',changed,noise=noise,num_steps=2)
        torch.testing.assert_close(graph2,expected,atol=2e-5,rtol=2e-5)
        assert not torch.allclose(graph,graph2)
    finally:
        model.close()
