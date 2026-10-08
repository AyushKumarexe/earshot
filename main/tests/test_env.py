import numpy as np
import pytest

from earshot import load_config
from earshot.control.baselines import PID
from earshot.control.evaluate import make_scenarios, run_episode
from earshot.control.supervisor import SetpointEnv
from earshot.sim.env import RackEnv


@pytest.fixture
def env():
    return RackEnv(load_config("configs/rack.yaml"))


def test_spaces_and_step(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (29,)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.dtype == np.float32
    assert reward < 0
    assert info["cost"] >= 0
    assert not terminated


def test_episode_truncates(env):
    env.reset(seed=0)
    for _ in range(env.episode_steps - 1):
        assert not env.step(np.zeros(6))[3]
    assert env.step(np.zeros(6))[3]


def test_seeded_reset_is_deterministic(env):
    a = [env.reset(seed=3)[0]] + [env.step(np.zeros(6))[0] for _ in range(50)]
    b = [env.reset(seed=3)[0]] + [env.step(np.zeros(6))[0] for _ in range(50)]
    assert np.allclose(a, b)


def test_speed_action_roundtrip(env):
    s = np.linspace(env.min_speed, 1.0, 6)
    assert np.allclose(env.action_to_speed(env.speed_to_action(s)), s)


def test_ramp_limit(env):
    env.reset(seed=0)
    before = env.speeds.copy()
    env.step(np.ones(6))
    assert np.all(env.speeds - before <= env.max_ramp + 1e-9)


def test_health_hidden_when_blind():
    cfg = load_config("configs/rack.yaml")
    blind = RackEnv(cfg, use_health=False)
    blind.reset(seed=0)
    blind.health = np.zeros(6)
    assert np.allclose(blind.decode(blind._obs())["health"], 1.0)


def test_pid_keeps_rack_safe(env):
    scenarios = make_scenarios(env, "faults", days=2, seed=11, steps=2880)
    for i, s in enumerate(scenarios):
        metrics, _ = run_episode(env, PID(env), s, seed=i)
        assert metrics["violation_min"] == 0


def test_setpoint_env_runs_pid_underneath():
    env = SetpointEnv(load_config("configs/rack.yaml"))
    env.reset(seed=0)
    assert env.action_space.shape == (4,)
    _, _, _, _, info = env.step(np.full(4, -1.0))
    assert np.allclose(info["setpoint"], 70.0)
    _, _, _, _, info = env.step(np.full(4, 1.0))
    assert np.allclose(info["setpoint"], 82.0)
