import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pareto(csv_paths, out):
    df = pd.concat([pd.read_csv(p) for p in csv_paths])
    summary = df.groupby("controller").agg(fan_wh=("fan_wh", "mean"), fan_sd=("fan_wh", "std"),
                                           over=("violation_min", "mean"), over_sd=("violation_min", "std"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in summary.iterrows():
        ax.errorbar(r.fan_wh, r.over, xerr=r.fan_sd, yerr=r.over_sd, fmt="o", capsize=3)
        ax.annotate(name, (r.fan_wh, r.over), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("fan energy (Wh / day)")
    ax.set_ylabel("time above T_max (min / day)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def replay(npz_path, out, names=None):
    d = np.load(npz_path)
    dt = float(d["dt"])
    controllers = sorted({k.split("__")[0] for k in d.files if "__" in k})
    names = names or controllers
    faults = json.loads(str(d["faults"]))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for name in names:
        hours = np.arange(len(d[f"{name}__max_temp"])) * dt / 3600
        axes[0].plot(hours, d[f"{name}__max_temp"], label=name, lw=1)
        axes[1].plot(hours, d[f"{name}__fan_w"], label=name, lw=1)
        axes[2].plot(hours, d[f"{name}__min_health"], label=name, lw=1)
    axes[0].axhline(85, color="k", ls="--", lw=0.8)
    for f in faults:
        for ax in axes:
            ax.axvline(f["onset"] * dt / 3600, color="r", alpha=0.4, lw=0.8)
    axes[0].set_ylabel("hottest zone (C)")
    axes[1].set_ylabel("fan power (W)")
    axes[2].set_ylabel("lowest fan health")
    axes[2].set_xlabel("hours")
    axes[0].legend(fontsize=8)
    title = ", ".join(f"fan {f['fan']} {f['kind']}" for f in faults) or "no faults"
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pareto")
    p.add_argument("csv", nargs="+")
    p.add_argument("--out", default="results/pareto.png")
    r = sub.add_parser("replay")
    r.add_argument("npz")
    r.add_argument("--names", nargs="+")
    r.add_argument("--out", default="results/replay.png")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "pareto":
        pareto(args.csv, args.out)
    else:
        replay(args.npz, args.out, args.names)


if __name__ == "__main__":
    main()
