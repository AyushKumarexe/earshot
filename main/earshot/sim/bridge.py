import numpy as np


class AudioBridge:
    """Turns each fan's true fault severity into the health score the detector would report."""

    def __init__(self, normal, abnormal, smoothing=0.3, noise=0.0):
        self.normal = np.asarray(normal, dtype=float)
        self.abnormal = np.asarray(abnormal, dtype=float)
        self.smoothing = smoothing
        self.noise = noise
        self.rng = np.random.default_rng()
        self.health = None

    @classmethod
    def load(cls, path, **kw):
        d = np.load(path)
        return cls(d["normal"], d["abnormal"], **kw)

    @classmethod
    def synthetic(cls, seed=0, n=4000, **kw):
        rng = np.random.default_rng(seed)
        return cls(rng.beta(12, 1.5, n), rng.beta(2, 4, n), **kw)

    def reset(self, n_fans, rng):
        self.rng = rng
        self.health = np.ones(n_fans)
        return self.health

    def step(self, severity):
        n = len(severity)
        from_abnormal = self.rng.random(n) < severity
        raw = np.where(from_abnormal, self.rng.choice(self.abnormal, n), self.rng.choice(self.normal, n))
        if self.noise:
            raw = np.clip(raw + self.rng.normal(0, self.noise, n), 0, 1)
        self.health = (1 - self.smoothing) * self.health + self.smoothing * raw
        return self.health
