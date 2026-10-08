import argparse
import itertools

import numpy as np

from earshot import load_config
from earshot.control.baselines import PID
from earshot.control.evaluate import make_scenarios, run_episode
from earshot.sim.env import RackEnv


def main():
    parser = argparse.ArgumentParser(description="Grid search PID gains: least fan energy with no over-temperature.")
    parser.add_argument("--rack", default="configs/rack.yaml")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--hours", type=float, default=8)
    args = parser.parse_args()

    env = RackEnv(load_config(args.rack))
    steps = int(args.hours * 3600 / env.rack.dt)
    scenarios = make_scenarios(env, "faults", args.days, seed=7, steps=steps)

    results = []
    for setpoint, kp, ki in itertools.product([72, 75, 78, 80], [0.02, 0.05, 0.1], [0.0002, 0.0004, 0.001]):
        pid = PID(env, setpoint=setpoint, kp=kp, ki=ki)
        runs = [run_episode(env, pid, s, seed=i)[0] for i, s in enumerate(scenarios)]
        wh = np.mean([r["fan_wh"] for r in runs])
        over = np.sum([r["violation_min"] for r in runs])
        results.append((over, wh, setpoint, kp, ki))
        print(f"setpoint {setpoint}  kp {kp:<5}  ki {ki:<6}  fan {wh:6.1f} Wh  over-temp {over:5.1f} min", flush=True)

    safe = [r for r in results if r[0] == 0] or sorted(results)[:1]
    best = min(safe, key=lambda r: r[1])
    print(f"\nbest: setpoint {best[2]}  kp {best[3]}  ki {best[4]}  ({best[1]:.1f} Wh, {best[0]:.1f} min over)")


if __name__ == "__main__":
    main()
