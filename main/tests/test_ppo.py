import numpy as np
import pytest

from earshot import load_config
from earshot.control.lagrangian import Lagrange
from earshot.control.ppo import PolicyController, RunningNorm, gae
from earshot.control.train import train


def test_gae_matches_discounted_return_when_values_are_zero():
    r = np.array([[1.0], [1.0], [1.0]])
    zeros = np.zeros_like(r)
    adv, ret = gae(r, zeros, zeros, np.array([[0.0], [0.0], [1.0]]), gamma=0.5, lam=1.0)
    assert np.allclose(ret[:, 0], [1.75, 1.5, 1.0])


def test_running_norm_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, (1000, 4))
    norm = RunningNorm(4)
    for chunk in np.split(x, 10):
        norm.update(chunk)
    assert np.allclose(norm.mean, x.mean(0), atol=1e-3)
    assert np.allclose(norm.var, x.var(0), rtol=1e-2)


def test_lagrange_rises_with_cost_and_stays_non_negative():
    lag = Lagrange(cost_limit=0.1, lr=1.0)
    assert lag.update(0.6) == 0.5
    assert lag.update(0.0) == 0.4
    for _ in range(10):
        lag.update(0.0)
    assert lag.value == 0.0


@pytest.mark.parametrize("action, n_act", [("setpoint", 4), ("speeds", 6)])
def test_training_smoke(tmp_path, action, n_act):
    cfg = load_config("configs/ppo_lag.yaml")
    cfg.update(total_steps=2 * 64 * 2, n_envs=2, rollout_steps=64, epochs=2, minibatch=64, action=action)
    rack = load_config("configs/rack.yaml")
    rack["episode_steps"] = 100
    train(cfg, rack, tmp_path)
    ctrl = PolicyController(str(tmp_path / "policy.pt"))
    assert ctrl.action == action
    assert ctrl.act(np.zeros(29, np.float32)).shape == (n_act,)
