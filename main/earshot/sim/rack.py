import numpy as np


def mixing_matrix(n_zones, n_fans, spread=2.0):
    zone_pos = np.linspace(0, 1, n_zones)
    fan_pos = np.linspace(0, 1, n_fans)
    d = (zone_pos[:, None] - fan_pos[None, :]) / (spread / n_zones)
    w = np.exp(-0.5 * d**2)
    return w / w.sum(axis=0, keepdims=True)


class Rack:
    def __init__(self, cfg):
        self.n_zones = cfg["zones"]
        self.n_fans = cfg["fans"]
        self.dt = cfg["dt"]
        self.capacity = cfg["heat_capacity"]
        self.g_ref = cfg["conductance"]
        self.g_leak = cfg["leak_conductance"]
        self.p_idle = cfg["server_idle_w"]
        self.p_max = cfg["server_max_w"]
        self.fan_rated = cfg["fan_rated_w"]
        self.A = mixing_matrix(self.n_zones, self.n_fans, cfg["airflow_spread"])
        self.max_fan_power = self.fan_rated * self.n_fans
        self.temps = np.zeros(self.n_zones)

    def server_power(self, load):
        return self.p_idle + (self.p_max - self.p_idle) * np.asarray(load)

    def conductance(self, speeds, efficiency):
        q = self.A @ (speeds * efficiency)
        # forced convection scales roughly with Re^0.8 (Dittus-Boelter)
        return self.g_ref * q**0.8 + self.g_leak

    def steady_state(self, load, speeds, efficiency, inlet):
        return inlet + self.server_power(load) / self.conductance(speeds, efficiency)

    def fan_power(self, speeds, power_scale):
        return self.fan_rated * speeds**3 * power_scale

    def reset(self, temps):
        self.temps = np.array(temps, dtype=float)

    def step(self, load, speeds, efficiency, inlet):
        g = self.conductance(speeds, efficiency)
        target = inlet + self.server_power(load) / g
        decay = np.exp(-g * self.dt / self.capacity)
        self.temps = target + (self.temps - target) * decay
        return self.temps
