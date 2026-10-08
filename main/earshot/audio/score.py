import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from earshot.audio.data import load_splits
from earshot.audio.features import Featurizer


@dataclass
class Calibration:
    threshold: float
    scale: float

    def health(self, score):
        z = (np.asarray(score, dtype=float) - self.threshold) / self.scale
        return 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))


def fit_calibration(scores, percentile=99.0):
    s = np.asarray(scores, dtype=float)
    return Calibration(float(np.percentile(s, percentile)), float(s.std() + 1e-9))


def load_bundle(path):
    return joblib.load(path)


def calibration_for(bundle, machine_id):
    return bundle["calibration"].get(machine_id, bundle["calibration"]["all"])


def bridge_cache(bundle, max_windows=10, log=print):
    cfg = bundle["config"]
    _, _, test = load_splits(cfg["data"], cfg["seed"])
    det = bundle["detector"]
    feat = Featurizer(cfg["features"], det.input, cfg["cache_dir"])
    normal, abnormal = [], []
    for i, clip in enumerate(test):
        cal = calibration_for(bundle, clip.machine_id)
        health = cal.health([det.score(w) for w in feat.windows(clip, max_windows)])
        (abnormal if clip.label else normal).extend(health)
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(test)} clips")
    return np.array(normal), np.array(abnormal)


def main():
    parser = argparse.ArgumentParser(description="Score the test set in 1 s windows and save health values "
                                                 "for the simulator's audio bridge.")
    parser.add_argument("bundle")
    parser.add_argument("--out", help="defaults to runs/bridge/<bundle name>.npz")
    parser.add_argument("--max-windows", type=int, default=10)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    out = Path(args.out or f"runs/bridge/{Path(args.bundle).stem}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    normal, abnormal = bridge_cache(bundle, args.max_windows)
    np.savez(out, normal=normal, abnormal=abnormal)
    print(f"normal windows   {len(normal):6d}  mean health {normal.mean():.3f}  below 0.5: {(normal < 0.5).mean():.1%}")
    print(f"abnormal windows {len(abnormal):6d}  mean health {abnormal.mean():.3f}  below 0.5: {(abnormal < 0.5).mean():.1%}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
