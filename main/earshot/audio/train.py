import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from earshot import load_config
from earshot.audio.data import load_splits
from earshot.audio.features import Featurizer
from earshot.audio.models import build_detector
from earshot.audio.score import fit_calibration


def featurize(feat, clips, what="clips"):
    out = []
    start = time.time()
    for i, clip in enumerate(clips):
        out.append(feat.clip(clip))
        if (i + 1) % 500 == 0:
            print(f"  {what}: {i + 1}/{len(clips)}  ({time.time() - start:.0f}s)", flush=True)
    return out


def evaluate(det, feat, test):
    scores = [det.score(x) for x in featurize(feat, test, "test")]
    rows = []
    for mid in sorted({c.machine_id for c in test}):
        idx = [i for i, c in enumerate(test) if c.machine_id == mid]
        y = [test[i].label for i in idx]
        s = [scores[i] for i in idx]
        if len(set(y)) < 2:
            continue
        # sklearn's max_fpr pAUC is McClish-corrected; the DCASE 2020 baseline reports the same number
        rows.append({"machine_id": mid, "n": len(idx), "auc": roc_auc_score(y, s),
                     "pauc": roc_auc_score(y, s, max_fpr=0.1)})
    if rows:
        rows.append({"machine_id": "mean", "n": len(test),
                     "auc": np.mean([r["auc"] for r in rows]), "pauc": np.mean([r["pauc"] for r in rows])})
    return rows, scores


def calibrate(det, feat, val, max_windows):
    by_machine = {}
    for clip in val:
        by_machine.setdefault(clip.machine_id, []).extend(det.score(w) for w in feat.windows(clip, max_windows))
    cals = {mid: fit_calibration(s) for mid, s in by_machine.items()}
    cals["all"] = fit_calibration([x for s in by_machine.values() for x in s])
    return cals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ae", choices=["gmm", "ae", "conv", "knn"])
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--layout", choices=["dcase", "mimii", "own"])
    parser.add_argument("--snr", help="for the mimii layout: 6, 0 or -6")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--out", default="runs/detector")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.layout:
        cfg["data"]["layout"] = args.layout
    if args.snr:
        cfg["data"]["snr"] = args.snr
    if args.epochs and args.model in ("ae", "conv"):
        cfg["models"][args.model]["epochs"] = args.epochs

    fit, val, test = load_splits(cfg["data"], cfg["seed"])
    print(f"{len(fit)} train / {len(val)} val (normal only), {len(test)} test clips")

    det = build_detector(args.model, cfg)
    feat = Featurizer(cfg["features"], det.input, cfg["cache_dir"])
    det.fit(featurize(feat, fit, "train"))
    cals = calibrate(det, feat, val, cfg["calibration_windows"])
    rows, scores = evaluate(det, feat, test)

    layout = cfg["data"]["layout"]
    name = f"{layout}_{args.model}" + (f"_{cfg['data']['snr']}dB" if layout == "mimii" else "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"detector": det, "calibration": cals, "config": cfg, "name": name}, out / f"{name}.joblib")

    with open(out / f"{name}_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["machine_id", "n", "auc", "pauc"])
        writer.writeheader()
        writer.writerows(rows)
    with open(out / f"{name}_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "offset", "machine_id", "label", "condition", "score"])
        for clip, s in zip(test, scores):
            writer.writerow([clip.path, clip.offset, clip.machine_id, clip.label, clip.condition, s])

    print(f"\n{name}")
    print(f"{'machine':>10s} {'AUC':>7s} {'pAUC':>7s}")
    for r in rows:
        print(f"{r['machine_id']:>10s} {r['auc']:7.3f} {r['pauc']:7.3f}")
    print(f"saved {out / name}.joblib")


if __name__ == "__main__":
    main()
