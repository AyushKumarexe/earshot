import gymnasium as gym
import numpy as np
from gymnasium import spaces

from earshot.sim.bridge import AudioBridge
from earshot.sim.faults import KINDS, fan_state, sample_faults
from earshot.sim.rack import Rack
from earshot.sim.workload import inlet_profile, make_load_source


class RackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, use_health=True, bridge=None, load_source=None, shield=None):
        self.cfg = cfg
        self.rack = Rack(cfg)
        self.n_zones = self.rack.n_zones
        self.n_fans = self.rack.n_fans
        self.t_max = cfg["t_max"]
        self.min_speed = cfg["min_speed"]
        self.max_ramp = cfg["max_ramp"]
        self.smooth_penalty = cfg["smooth_penalty"]
        self.episode_steps = cfg["episode_steps"]
        self.fault_cfg = cfg["faults"]
        self.use_health = use_health
        self.shield = shield

        if bridge is None:
            if cfg.get("bridge"):
                bridge = AudioBridge.load(cfg["bridge"], smoothing=cfg["bridge_smoothing"], noise=cfg["bridge_noise"])
            else:
                bridge = AudioBridge.synthetic(smoothing=cfg["bridge_smoothing"], noise=cfg["bridge_noise"])
        self.bridge = bridge
        self.load_source = load_source or make_load_source(cfg.get("workload"))

        n_obs = 4 * self.n_zones + 2 * self.n_fans + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_fans,), np.float32)

    def action_to_speed(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.min_speed + (a + 1.0) / 2.0 * (1.0 - self.min_speed)

    def speed_to_action(self, speed):
        s = np.clip(np.asarray(speed, dtype=float), self.min_speed, 1.0)
        return 2.0 * (s - self.min_speed) / (1.0 - self.min_speed) - 1.0

    def make_scenario(self, rng, n_steps, kinds=None, p_clean=None):
        fc = self.fault_cfg
        return {
            "steps": n_steps,
            "load": self.load_source(rng, n_steps, self.n_zones, self.rack.dt),
            "inlet": inlet_profile(rng, n_steps, self.rack.dt, self.cfg["inlet_mean"],
                                   self.cfg["inlet_amp"], self.cfg["inlet_noise"]),
            "faults": sample_faults(
                rng, self.n_fans, n_steps,
                p_clean=fc["p_clean"] if p_clean is None else p_clean,
                kinds=tuple(kinds or fc.get("kinds", KINDS)),
                max_faulty=fc["max_faulty"],
                wear_ramp=tuple(fc["wear_ramp"]),
            ),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = options or self.make_scenario(self.np_random, self.episode_steps)
        self.n_steps = scenario["steps"]
        self.load = scenario["load"]
        self.inlet = scenario["inlet"]
        self.faults = scenario["faults"]
        self.audio_lead = self.fault_cfg["audio_lead"]

        self.t = 0
        self.speeds = np.full(self.n_fans, 0.5)
        self.current_load = self.load[0]
        self.inlet_now = self.inlet[0]
        settled = self.rack.steady_state(self.current_load, self.speeds, np.ones(self.n_fans), self.inlet_now)
        self.rack.reset(np.minimum(settled, self.t_max - 10.0))
        self.prev_temps = self.rack.temps.copy()
        self.health = self.bridge.reset(self.n_fans, self.np_random)
        return self._obs(), {}

    def step(self, action):
        target = self.action_to_speed(action)
        if self.shield is not None and (self.rack.temps > self.t_max - self.shield).any():
            target[:] = 1.0
        new_speeds = self.speeds + np.clip(target - self.speeds, -self.max_ramp, self.max_ramp)
        change = np.abs(new_speeds - self.speeds).sum()
        self.speeds = new_speeds

        efficiency, power_scale, severity = fan_state(self.faults, self.n_fans, self.t, self.audio_lead)
        self.current_load = self.load[self.t]
        self.inlet_now = self.inlet[self.t]
        self.prev_temps = self.rack.temps.copy()
        temps = self.rack.step(self.current_load, self.speeds, efficiency, self.inlet_now)
        fan_w = self.rack.fan_power(self.speeds, power_scale).sum()
        self.health = self.bridge.step(severity)

        over = np.maximum(0.0, temps - self.t_max)
        reward = -fan_w / self.rack.max_fan_power - self.smooth_penalty * change
        self.t += 1
        info = {
            "cost": float(over.sum()),
            "fan_w": float(fan_w),
            "max_temp": float(temps.max()),
            "violation": bool(over.any()),
        }
        return self._obs(), float(reward), False, self.t >= self.n_steps, info

    def _obs(self):
        temps = self.rack.temps
        health = self.health if self.use_health else np.ones(self.n_fans)
        return np.concatenate([
            temps / 100.0,
            (temps - self.prev_temps) * 10.0,
            self.current_load,
            (self.t_max - temps) / 50.0,
            self.speeds,
            health,
            [self.inlet_now / 50.0],
        ]).astype(np.float32)

    def decode(self, obs):
        nz, nf = self.n_zones, self.n_fans
        o = np.asarray(obs, dtype=float)
        return {
            "temps": o[:nz] * 100.0,
            "loads": o[2 * nz:3 * nz],
            "speeds": o[4 * nz:4 * nz + nf],
            "health": o[4 * nz + nf:4 * nz + 2 * nf],
        }
