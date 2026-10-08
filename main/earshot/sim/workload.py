from pathlib import Path

import numpy as np
from scipy.signal import lfilter

DAY = 86400.0


def ar1(rng, n_steps, n_cols, phi, sigma):
    white = rng.normal(0.0, sigma, (n_steps, n_cols))
    return lfilter([1.0], [1.0, -phi], white, axis=0)


def synthetic_load(rng, n_steps, n_zones, dt, bursts_per_hour=1.5):
    t = np.arange(n_steps) * dt + rng.uniform(0, DAY)
    day = np.sin(2 * np.pi * (t / DAY - 0.25))
    load = 0.4 + 0.2 * day[:, None] + rng.uniform(-0.1, 0.1, n_zones)
    load += ar1(rng, n_steps, n_zones, phi=0.995, sigma=0.01)

    hours = n_steps * dt / 3600
    for z in range(n_zones):
        for _ in range(rng.poisson(bursts_per_hour * hours)):
            start = int(rng.integers(n_steps))
            length = int(rng.uniform(120, 1200) / dt)
            load[start:start + length, z] += rng.uniform(0.2, 0.6)
    return np.clip(load, 0.02, 1.0)


class TraceLoad:
    def __init__(self, path):
        self.cpu = np.load(path).astype(np.float32)

    def __call__(self, rng, n_steps, n_zones, dt):
        machines, length = self.cpu.shape
        if n_steps > length:
            raise ValueError(f"trace has {length} steps, asked for {n_steps}")
        rows = rng.choice(machines, size=n_zones, replace=False)
        start = int(rng.integers(0, length - n_steps + 1))
        return self.cpu[rows, start:start + n_steps].T.astype(float)


def make_load_source(spec):
    if spec in (None, "synthetic"):
        return synthetic_load
    if not Path(spec).exists():
        raise FileNotFoundError(f"{spec} not found, run scripts/prep_alibaba.py first or use 'synthetic'")
    return TraceLoad(spec)


def inlet_profile(rng, n_steps, dt, mean=25.0, amp=5.0, noise=0.3):
    t = np.arange(n_steps) * dt + rng.uniform(0, DAY)
    temp = mean + amp * np.sin(2 * np.pi * (t / DAY - 0.25))
    phi = 0.99
    return temp + ar1(rng, n_steps, 1, phi=phi, sigma=noise * np.sqrt(1 - phi**2))[:, 0]
