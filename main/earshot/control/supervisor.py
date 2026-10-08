import numpy as np
from gymnasium import spaces

from earshot.control.baselines import PID
from earshot.sim.env import RackEnv


class SetpointEnv(RackEnv):
    """The agent picks each zone's temperature setpoint and a PID loop drives the fans to it.

    Adding speed offsets on top of PID doesn't work: the integral term slowly cancels any steady offset.
    """

    def __init__(self, cfg, setpoint_range=(70.0, 82.0), **kw):
        super().__init__(cfg, **kw)
        self.low, self.high = setpoint_range
        self.pid = PID(self)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_zones,), np.float32)
        self.last_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.pid.reset()
        self.last_obs = obs
        return obs, info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.pid.setpoint = self.low + (a + 1.0) / 2.0 * (self.high - self.low)
        obs, reward, terminated, truncated, info = super().step(self.pid.act(self.last_obs))
        self.last_obs = obs
        info["setpoint"] = self.pid.setpoint.copy()
        return obs, reward, terminated, truncated, info


def make_env(rack_cfg, action="speeds", setpoint_range=(70.0, 82.0), **kw):
    if action == "setpoint":
        return SetpointEnv(rack_cfg, setpoint_range=tuple(setpoint_range), **kw)
    return RackEnv(rack_cfg, **kw)
