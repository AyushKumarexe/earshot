import argparse
import csv
import json
from pathlib import Path

import numpy as np

from earshot import load_config
from earshot.control.baselines import PID, FanCurve
from earshot.control.ppo import PolicyController
from earshot.control.supervisor import make_env
from earshot.sim.env import RackEnv

DAY_STEPS = 17280


def make_scenarios(env, kind, days, seed, steps=DAY_STEPS, kinds=None):
    rng = np.random.default_rng(seed)
    p_clean = 1.0 if kind == "clean" else 0.0
    return [env.make_scenario(rng, steps, kinds=kinds, p_clean=p_clean) for _ in range(days)]


def build_controller(spec, rack_cfg, shield=None):
    if spec in ("curve", "pid"):
        env = RackEnv(rack_cfg, shield=shield)
        ctrl = FanCurve(env) if spec == "curve" else PID(env)
        return ctrl, env
    ctrl = PolicyController(spec, name=Path(spec).parent.name)
    env = make_env(rack_cfg, ctrl.action, ctrl.setpoint_range, use_health=ctrl.use_health, shield=shield)
    return ctrl, env


def run_episode(env, ctrl, scenario, seed, record=False):
    obs, _ = env.reset(seed=seed, options=scenario)
    ctrl.reset()
    dt = env.rack.dt
    fan_wh = violation_min = cost = 0.0
    peak = -np.inf
    trace = {"max_temp": [], "fan_w": [], "min_health": [], "speeds": []}
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(ctrl.act(obs))
        done = terminated or truncated
        fan_wh += info["fan_w"] * dt / 3600
        violation_min += info["violation"] * dt / 60
        cost += info["cost"]
        peak = max(peak, info["max_temp"])
        if record:
            trace["max_temp"].append(info["max_temp"])
            trace["fan_w"].append(info["fan_w"])
            trace["min_health"].append(env.health.min())
            trace["speeds"].append(env.speeds.copy())
    metrics = {
        "fan_wh": fan_wh,
        "violation_min": violation_min,
        "peak_temp": peak,
        "mean_cost": cost / scenario["steps"],
    }
    return metrics, {k: np.asarray(v) for k, v in trace.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("controllers", nargs="+", help="curve, pid, or paths to policy.pt")
    parser.add_argument("--rack", default="configs/rack.yaml")
    parser.add_argument("--scenario", choices=["clean", "faults"], default="faults")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=DAY_STEPS)
    parser.add_argument("--kinds", nargs="+")
    parser.add_argument("--shield", type=float, help="force full speed within this many degrees of T_max")
    parser.add_argument("--bridge")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    rack_cfg = load_config(args.rack)
    if args.bridge:
        rack_cfg["bridge"] = args.bridge
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = make_scenarios(RackEnv(rack_cfg), args.scenario, args.days, args.seed, args.steps, args.kinds)
    suffix = f"{args.scenario}" + ("_shield" if args.shield else "")
    rows, traces = [], {}
    for spec in args.controllers:
        ctrl, env = build_controller(spec, rack_cfg, args.shield)
        for day, scenario in enumerate(scenarios):
            metrics, trace = run_episode(env, ctrl, scenario, seed=args.seed + day, record=day == 0)
            rows.append({"controller": ctrl.name, "day": day, **metrics})
            if day == 0:
                traces[ctrl.name] = trace
        summary = [r for r in rows if r["controller"] == ctrl.name]
        print(f"{ctrl.name:>20s}  fan {np.mean([r['fan_wh'] for r in summary]):7.1f} Wh/day  "
              f"over-temp {np.mean([r['violation_min'] for r in summary]):6.1f} min/day  "
              f"peak {max(r['peak_temp'] for r in summary):5.1f} C")

    with open(out / f"eval_{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first = scenarios[0]
    np.savez_compressed(
        out / f"trace_{suffix}.npz",
        dt=rack_cfg["dt"],
        faults=json.dumps([vars(f) for f in first["faults"]]),
        **{f"{name}__{key}": value for name, trace in traces.items() for key, value in trace.items()},
    )


if __name__ == "__main__":
    main()
