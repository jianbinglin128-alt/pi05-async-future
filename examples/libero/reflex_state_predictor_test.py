"""Controller-specific regression checks; requires the LIBERO environment."""

from types import SimpleNamespace

import numpy as np

from examples.libero.main import _make_state_predictor


def predictor():
    controller = SimpleNamespace(
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
        output_max=np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
    )
    return _make_state_predictor(SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)])))


def test_predictor_scales_clips_and_uses_last_chunk_gripper_target():
    state = np.array([0.2, 0.3, 0.4, 0, 0, 0, 0.01, -0.01], dtype=np.float64)
    remaining = np.zeros((2, 7))
    remaining[:, 0] = 2  # Clip to 1, then apply 5 cm per command.
    remaining[:, 6] = 1
    last = np.zeros(7)
    last[6] = -1
    result = predictor()(state.copy(), remaining, last)
    np.testing.assert_allclose(result, [0.3, 0.3, 0.4, 0, 0, 0, 0.04, -0.04])
    last[6] = 1
    np.testing.assert_allclose(predictor()(state.copy(), remaining, last)[6:], [0, 0])
    last[6] = 0
    np.testing.assert_allclose(predictor()(state.copy(), remaining, last)[6:], state[6:])


def test_predictor_composes_world_rotations_in_execution_order():
    from scipy.spatial.transform import Rotation

    state = np.array([0, 0, 0, 0.3, 0, 0, 0, 0], dtype=np.float64)
    remaining = np.zeros((2, 7))
    remaining[0, 4] = 0.8  # 0.4 radians about world Y.
    remaining[1, 5] = -0.6  # Then -0.3 radians about world Z.
    result = predictor()(state.copy(), remaining, remaining[-1])
    cy, sy, cz, sz, cx, sx = np.cos(0.4), np.sin(0.4), np.cos(-0.3), np.sin(-0.3), np.cos(0.3), np.sin(0.3)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    np.testing.assert_allclose(Rotation.from_rotvec(result[3:6]).as_matrix(), rz @ ry @ rx, atol=1e-12)
