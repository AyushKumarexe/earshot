import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningNorm:
    def __init__(self, size, clip=10.0):
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        batch_mean, batch_var, n = x.mean(0), x.var(0), len(x)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


def mlp(n_in, n_out, hidden):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_out),
    )


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_act, hidden=64, init_log_std=-0.5):
        super().__init__()
        self.actor = mlp(n_obs, n_act, hidden)
        self.log_std = nn.Parameter(torch.full((n_act,), init_log_std))
        self.reward_critic = mlp(n_obs, 1, hidden)
        self.cost_critic = mlp(n_obs, 1, hidden)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        return Normal(self.actor(obs), self.log_std.exp())

    def values(self, obs):
        return self.reward_critic(obs).squeeze(-1), self.cost_critic(obs).squeeze(-1)


def gae(rewards, values, next_values, dones, gamma, lam):
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


class PPOLag:
    def __init__(self, n_obs, n_act, cfg, lagrange):
        self.cfg = cfg
        self.model = ActorCritic(n_obs, n_act, cfg["hidden"], cfg["init_log_std"])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], eps=1e-5)
        self.norm = RunningNorm(n_obs)
        self.lagrange = lagrange

    @torch.no_grad()
    def step(self, x):
        x = torch.as_tensor(x)
        dist = self.model.dist(x)
        action = dist.sample()
        v_r, v_c = self.model.values(x)
        return action.numpy(), dist.log_prob(action).sum(-1).numpy(), v_r.numpy(), v_c.numpy()

    @torch.no_grad()
    def value(self, x):
        v_r, v_c = self.model.values(torch.as_tensor(x))
        return v_r.numpy(), v_c.numpy()

    def update(self, batch):
        cfg = self.cfg
        lam = self.lagrange.value
        adv_r = (batch["adv_r"] - batch["adv_r"].mean()) / (batch["adv_r"].std() + 1e-8)
        adv_c = batch["adv_c"] - batch["adv_c"].mean()
        adv = (adv_r - lam * adv_c) / (1.0 + lam)

        obs = torch.as_tensor(batch["obs"])
        act = torch.as_tensor(batch["act"])
        logp_old = torch.as_tensor(batch["logp"])
        adv = torch.as_tensor(adv, dtype=torch.float32)
        ret_r = torch.as_tensor(batch["ret_r"], dtype=torch.float32)
        ret_c = torch.as_tensor(batch["ret_c"], dtype=torch.float32)

        n = len(obs)
        stats = {"kl": 0.0, "entropy": 0.0, "epochs": 0}
        for epoch in range(cfg["epochs"]):
            kls = []
            for idx in torch.randperm(n).split(cfg["minibatch"]):
                dist = self.model.dist(obs[idx])
                logp = dist.log_prob(act[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                clipped = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])
                policy_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()

                v_r, v_c = self.model.values(obs[idx])
                value_loss = ((v_r - ret_r[idx]) ** 2).mean() + ((v_c - ret_c[idx]) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg["max_grad_norm"])
                self.opt.step()

                with torch.no_grad():
                    kls.append(((ratio - 1) - (logp - logp_old[idx])).mean().item())
                stats["entropy"] = entropy.item()
            stats["kl"] = float(np.mean(kls))
            stats["epochs"] = epoch + 1
            if stats["kl"] > cfg["target_kl"]:
                break
        return stats

    def save(self, path, extra=None):
        torch.save({
            "model": self.model.state_dict(),
            "norm": self.norm.state_dict(),
            "lambda": self.lagrange.value,
            "cfg": self.cfg,
            **(extra or {}),
        }, path)


class PolicyController:
    def __init__(self, path, name=None):
        ckpt = torch.load(path, weights_only=False)
        cfg = ckpt["cfg"]
        n_obs = len(ckpt["norm"]["mean"])
        self.model = ActorCritic(n_obs, ckpt["n_act"], cfg["hidden"], cfg["init_log_std"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.norm = RunningNorm(n_obs)
        self.norm.load_state_dict(ckpt["norm"])
        self.use_health = ckpt["use_health"]
        self.action = ckpt.get("action", "speeds")
        self.setpoint_range = ckpt.get("setpoint_range", (70.0, 82.0))
        self.name = name or path

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(self.norm(obs[None]))
        return self.model.actor(x)[0].numpy()
