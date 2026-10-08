import numpy as np
import pytest

from earshot import load_config
from earshot.sim.faults import Fault, fan_state
from earshot.sim.rack import Rack


@pytest.fixture
def cfg():
    return load_config("configs/rack.yaml")


def test_mixing_columns_sum_to_one(cfg):
    rack = Rack(cfg)
    assert np.allclose(rack.A.sum(axis=0), 1.0)


def test_full_fans_hold_full_load(cfg):
    rack = Rack(cfg)
    ones = np.ones(cfg["fans"])
    temps = rack.steady_state(np.ones(cfg["zones"]), ones, ones, inlet=30.0)
    assert temps.max() < cfg["t_max"]


def test_min_fans_overheat_under_load(cfg):
    rack = Rack(cfg)
    ones = np.ones(cfg["fans"])
    temps = rack.steady_state(np.ones(cfg["zones"]), ones * cfg["min_speed"], ones, inlet=30.0)
    assert temps.max() > cfg["t_max"]


@pytest.mark.parametrize("fan", range(6))
def test_any_single_stall_is_survivable(cfg, fan):
    rack = Rack(cfg)
    eff, _, _ = fan_state([Fault(fan, "stall", 0)], cfg["fans"], step=10)
    temps = rack.steady_state(np.ones(cfg["zones"]), np.ones(cfg["fans"]), eff, inlet=30.0)
    assert temps.max() < cfg["t_max"]


def test_step_converges_to_steady_state(cfg):
    rack = Rack(cfg)
    load, speeds, eff = np.full(4, 0.6), np.full(6, 0.5), np.ones(6)
    rack.reset(np.full(4, 30.0))
    for _ in range(5000):
        rack.step(load, speeds, eff, 25.0)
    assert np.allclose(rack.temps, rack.steady_state(load, speeds, eff, 25.0), atol=0.01)


def test_wear_is_heard_before_it_bites():
    f = Fault(0, "wear", onset=100, ramp=1000)
    eff, power, sev = fan_state([f], 6, step=350)
    assert sev[0] == pytest.approx(0.5)
    assert eff[0] == pytest.approx(1 - 0.6 * 0.25)
    assert power[0] > 1.0
    assert fan_state([f], 6, step=50)[2][0] == 0.0
