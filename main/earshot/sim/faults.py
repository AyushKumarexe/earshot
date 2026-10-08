from dataclasses import dataclass

import numpy as np

KINDS = ("wear", "obstruction", "stall")


@dataclass
class Fault:
    fan: int
    kind: str
    onset: int
    ramp: int = 1

    def progress(self, step):
        if step < self.onset:
            return 0.0
        if self.kind != "wear":
            return 1.0
        return min(1.0, (step - self.onset) / self.ramp)


def fan_state(faults, n_fans, step, audio_lead=2.0):
    efficiency = np.ones(n_fans)
    power_scale = np.ones(n_fans)
    severity = np.zeros(n_fans)
    for f in faults:
        p = f.progress(step)
        if p == 0.0:
            continue
        if f.kind == "wear":
            efficiency[f.fan] = 1.0 - 0.6 * p
            power_scale[f.fan] = 1.0 + 0.4 * p
            severity[f.fan] = min(1.0, audio_lead * p)
        elif f.kind == "obstruction":
            efficiency[f.fan] = 0.5
            severity[f.fan] = 0.7
        elif f.kind == "stall":
            efficiency[f.fan] = 0.0
            power_scale[f.fan] = 0.0
            severity[f.fan] = 1.0
        else:
            raise ValueError(f"unknown fault kind {f.kind!r}")
    return efficiency, power_scale, severity


def sample_faults(rng, n_fans, n_steps, p_clean=0.3, kinds=KINDS, max_faulty=2, wear_ramp=(240, 2160)):
    if rng.random() < p_clean:
        return []
    n = int(rng.integers(1, max_faulty + 1))
    faults = []
    for fan in rng.choice(n_fans, size=n, replace=False):
        kind = kinds[int(rng.integers(len(kinds)))]
        onset = int(rng.integers(0, max(1, int(n_steps * 0.6))))
        ramp = int(rng.integers(*wear_ramp)) if kind == "wear" else 1
        faults.append(Fault(int(fan), kind, onset, ramp))
    return faults
