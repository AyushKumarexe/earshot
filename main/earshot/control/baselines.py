import numpy as np


def zones_to_fans(env, demand, min_share=0.15):
    # a fan runs as fast as the hottest zone it meaningfully cools, like BMC fan zones
    feeds = env.rack.A >= min_share * env.rack.A.max(axis=0, keepdims=True)
    return np.where(feeds, demand[:, None], 0.0).max(axis=0)


class FanCurve:
    name = "curve"

    def __init__(self, env, low=(45.0, 0.3), high=(80.0, 1.0)):
        self.env = env
        self.low = low
        self.high = high

    def reset(self):
        pass

    def act(self, obs):
        temps = self.env.decode(obs)["temps"]
        demand = np.interp(temps, [self.low[0], self.high[0]], [self.low[1], self.high[1]])
        return self.env.speed_to_action(zones_to_fans(self.env, demand))


class PID:
    name = "pid"

    def __init__(self, env, setpoint=75.0, kp=0.05, ki=0.0004, kd=0.0, base=0.35):
        self.env = env
        self.setpoint = setpoint
        self.kp, self.ki, self.kd = kp, ki, kd
        self.base = base
        self.dt = env.rack.dt
        self.reset()

    def reset(self):
        self.integral = np.zeros(self.env.n_zones)
        self.prev_error = None

    def act(self, obs):
        error = self.env.decode(obs)["temps"] - self.setpoint
        deriv = np.zeros_like(error) if self.prev_error is None else (error - self.prev_error) / self.dt
        self.prev_error = error

        unclamped = self.base + self.kp * error + self.ki * self.integral + self.kd * deriv
        saturated = ((unclamped >= 1.0) & (error > 0)) | ((unclamped <= self.env.min_speed) & (error < 0))
        self.integral += np.where(saturated, 0.0, error * self.dt)

        demand = np.clip(self.base + self.kp * error + self.ki * self.integral + self.kd * deriv,
                         self.env.min_speed, 1.0)
        return self.env.speed_to_action(zones_to_fans(self.env, demand))
