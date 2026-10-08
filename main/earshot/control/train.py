import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from earshot import load_config
from earshot.control.lagrangian import Lagrange
from earshot.control.ppo import PPOLag, gae
from earshot.control.supervisor import make_env


def with_kinds(rack_cfg, kinds):
    if not kinds:
        return rack_cfg
    return {**rack_cfg, "faults": {**rack_cfg["faults"], "kinds": list(kinds)}}


class EnvPool:
    def __init__(self, envs, seed):
        self.envs = envs
        self.obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
        self.running = [self._blank() for _ in envs]
        self.finished = []

    @staticmethod
    def _blank():
        return {"return": 0.0, "cost": 0.0, "fan_wh": 0.0, "violation_min": 0.0, "hours": 0.0}

    def step(self, actions):
        n = len(self.envs)
        rewards, costs, dones = np.zeros(n), np.zeros(n), np.zeros(n)
        truncated_obs = {}
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            dt = env.rack.dt
            ep = self.running[i]
            ep["return"] += reward
            ep["cost"] += info["cost"]
            ep["fan_wh"] += info["fan_w"] * dt / 3600
            ep["violation_min"] += info["violation"] * dt / 60
            ep["hours"] += dt / 3600
            rewards[i], costs[i] = reward, info["cost"]
            if terminated or truncated:
                dones[i] = 1.0
                if not terminated:
                    truncated_obs[i] = obs
                self.finished.append(ep)
                self.running[i] = self._blank()
                obs, _ = env.reset()
            self.obs[i] = obs
        return rewards, costs, dones, truncated_obs


def train(cfg, rack_cfg, out, use_health=True, kinds=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    torch.manual_seed(seed)

    rack_cfg = with_kinds(rack_cfg, kinds)
    envs = [make_env(rack_cfg, cfg["action"], cfg["setpoint_range"], use_health=use_health)
            for _ in range(cfg["n_envs"])]
    pool = EnvPool(envs, seed)
    n_obs = envs[0].observation_space.shape[0]
    n_act = envs[0].action_space.shape[0]

    mode = cfg["mode"]
    init = {"lag": cfg["lambda_init"], "penalty": cfg["penalty"], "plain": 0.0}[mode]
    lagrange = Lagrange(cfg["cost_limit"], cfg["lambda_lr"], init=init, learn=mode == "lag")
    agent = PPOLag(n_obs, n_act, cfg, lagrange)
    extra = {"n_act": n_act, "use_health": use_health, "rack": rack_cfg, "mode": mode,
             "action": cfg["action"], "setpoint_range": cfg["setpoint_range"]}

    T, N, gamma = cfg["rollout_steps"], cfg["n_envs"], cfg["gamma"]
    iterations = max(1, cfg["total_steps"] // (T * N))

    log_file = open(out / "progress.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["iteration", "steps", "ep_return", "ep_cost", "fan_wh_per_h", "violation_min_per_h",
                  "lambda", "kl", "entropy"])
    start = time.time()

    for it in range(1, iterations + 1):
        obs_buf = np.zeros((T, N, n_obs), np.float32)
        act_buf = np.zeros((T, N, n_act), np.float32)
        logp_buf, rew_buf, cost_buf, done_buf, raw_cost = (np.zeros((T, N)) for _ in range(5))
        v_r_buf, v_c_buf = np.zeros((T, N)), np.zeros((T, N))

        for t in range(T):
            agent.norm.update(pool.obs)
            x = agent.norm(pool.obs)
            action, logp, v_r, v_c = agent.step(x)
            rewards, costs, dones, truncated_obs = pool.step(action)
            raw_cost[t] = costs
            for i, final in truncated_obs.items():
                f_r, f_c = agent.value(agent.norm(final[None]))
                rewards[i] += gamma * f_r[0]
                costs[i] += gamma * f_c[0]
            obs_buf[t], act_buf[t], logp_buf[t] = x, action, logp
            rew_buf[t], cost_buf[t], done_buf[t] = rewards, costs, dones
            v_r_buf[t], v_c_buf[t] = v_r, v_c

        last_r, last_c = agent.value(agent.norm(pool.obs))
        next_r = np.concatenate([v_r_buf[1:], last_r[None]])
        next_c = np.concatenate([v_c_buf[1:], last_c[None]])
        adv_r, ret_r = gae(rew_buf, v_r_buf, next_r, done_buf, gamma, cfg["gae_lambda"])
        adv_c, ret_c = gae(cost_buf, v_c_buf, next_c, done_buf, gamma, cfg["gae_lambda"])

        lagrange.update(raw_cost.mean())
        flat = lambda a: a.reshape(T * N, *a.shape[2:])
        stats = agent.update({
            "obs": flat(obs_buf), "act": flat(act_buf), "logp": flat(logp_buf).astype(np.float32),
            "adv_r": flat(adv_r), "adv_c": flat(adv_c), "ret_r": flat(ret_r), "ret_c": flat(ret_c),
        })

        recent = pool.finished[-2 * N:]
        if recent:
            hours = sum(ep["hours"] for ep in recent)
            row = [np.mean([ep["return"] for ep in recent]), np.mean([ep["cost"] for ep in recent]),
                   sum(ep["fan_wh"] for ep in recent) / hours, sum(ep["violation_min"] for ep in recent) / hours]
        else:
            row = [np.nan] * 4
        log.writerow([it, it * T * N, *row, lagrange.value, stats["kl"], stats["entropy"]])
        log_file.flush()

        if it % 10 == 0 or it == iterations:
            print(f"it {it:4d}/{iterations}  steps {it * T * N:>8d}  return {row[0]:8.2f}  "
                  f"fan {row[2]:6.1f} Wh/h  over-temp {row[3]:5.2f} min/h  lambda {lagrange.value:6.3f}  "
                  f"{time.time() - start:6.0f}s", flush=True)
        if it % 50 == 0 or it == iterations:
            agent.save(out / "policy.pt", extra)

    log_file.close()
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/ppo_lag.yaml")
    parser.add_argument("--rack", default="configs/rack.yaml")
    parser.add_argument("--mode", choices=["lag", "penalty", "plain"])
    parser.add_argument("--action", choices=["setpoint", "speeds"])
    parser.add_argument("--no-health", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cost-limit", type=float)
    parser.add_argument("--lambda-lr", type=float)
    parser.add_argument("--bridge", help="npz from earshot.audio.score, overrides rack config")
    parser.add_argument("--kinds", nargs="+", help="fault kinds seen in training, e.g. wear obstruction")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rack_cfg = load_config(args.rack)
    for key, value in [("mode", args.mode), ("action", args.action), ("seed", args.seed), ("total_steps", args.steps),
                       ("cost_limit", args.cost_limit), ("lambda_lr", args.lambda_lr)]:
        if value is not None:
            cfg[key] = value
    if args.bridge:
        rack_cfg["bridge"] = args.bridge
    train(cfg, rack_cfg, args.out, use_health=not args.no_health, kinds=args.kinds)


if __name__ == "__main__":
    main()
