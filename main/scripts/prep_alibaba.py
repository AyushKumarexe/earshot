import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Turn Alibaba cluster-trace-v2018 machine_usage.csv into a (machines, steps) CPU array.")
    parser.add_argument("csv", help="machine_usage.csv from the 2018 trace")
    parser.add_argument("--machines", type=int, default=200)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--out", default="data/alibaba_cpu.npy")
    args = parser.parse_args()

    keep = None
    parts = []
    reader = pd.read_csv(args.csv, header=None, usecols=[0, 1, 2], names=["machine", "t", "cpu"],
                         chunksize=5_000_000)
    for i, chunk in enumerate(reader):
        if keep is None:
            keep = set(chunk["machine"].drop_duplicates().iloc[:args.machines])
        parts.append(chunk[chunk["machine"].isin(keep)])
        print(f"chunk {i}: kept {sum(len(p) for p in parts)} rows", flush=True)

    df = pd.concat(parts).dropna().sort_values(["machine", "t"])
    t0, t1 = df["t"].min(), df["t"].max()
    grid = np.arange(t0, t1, args.dt)
    rows = []
    for machine, g in df.groupby("machine"):
        g = g.drop_duplicates("t")
        if len(g) < 100:
            continue
        rows.append(np.interp(grid, g["t"].to_numpy(), g["cpu"].to_numpy() / 100.0))
    cpu = np.clip(np.stack(rows), 0.0, 1.0).astype(np.float16)
    np.save(args.out, cpu)
    print(f"saved {args.out}: {cpu.shape[0]} machines x {cpu.shape[1]} steps ({cpu.shape[1] * args.dt / 86400:.1f} days)")


if __name__ == "__main__":
    main()
