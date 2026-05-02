from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.env import CameraBudgetEnv
from adaptive_scanning.rollout import eval_policy
from adaptive_scanning.policies import Policy


class MLPPolicy(nn.Module, Policy):
    """Categorical policy for REINFORCE; also usable with greedy argmax."""

    def __init__(self, obs_dim: int, *, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, obs: np.ndarray, info: dict) -> int:
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0)
            logits = self.forward_logits(x)
            return int(torch.argmax(logits, dim=-1).item())

    def act_stochastic(self, obs: np.ndarray) -> tuple[int, torch.Tensor]:
        x = torch.from_numpy(obs).float().unsqueeze(0)
        logits = self.forward_logits(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        lp = dist.log_prob(a)
        return int(a.item()), lp


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    g = 0.0
    out = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


@dataclass
class TrainResult:
    history: list[dict[str, float]]


def train_reinforce(
    cfg: AdaptiveScanningConfig,
    *,
    epochs: int = 40,
    episodes_per_epoch: int = 8,
    lr: float = 3e-4,
    gamma: float = 0.995,
    seed: int = 0,
    device: str | None = None,
) -> tuple[MLPPolicy, TrainResult]:
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = CameraBudgetEnv(cfg, seed=seed)
    obs_dim = env.observation_dim
    policy = MLPPolicy(obs_dim).to(dev)
    opt = optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []

    for ep in range(epochs):
        opt.zero_grad()
        batch_logp: list[torch.Tensor] = []
        batch_ret: list[torch.Tensor] = []
        for _ in range(episodes_per_epoch):
            s = int(rng.integers(0, 2**31 - 1))
            obs, _info = env.reset(seed=s)
            logps: list[torch.Tensor] = []
            rews: list[float] = []
            while True:
                x = torch.from_numpy(obs).float().to(dev).unsqueeze(0)
                logits = policy.forward_logits(x)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logps.append(dist.log_prob(a))
                step = env.step(int(a.item()))
                rews.append(step.reward)
                obs = step.observation
                if step.terminated or step.truncated:
                    break
            returns = _discounted_returns(rews, gamma)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=dev)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)
            for t in range(len(logps)):
                batch_logp.append(logps[t])
                batch_ret.append(returns_t[t])

        if not batch_logp:
            continue
        logp_stack = torch.stack(batch_logp)
        ret_stack = torch.stack(batch_ret)
        loss = -(logp_stack * ret_stack).mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            metrics = eval_policy(env, policy, n_episodes=4, seed0=ep + 1000)
        history.append(
            {
                "loss": float(loss.item()),
                "return_mean": metrics["return_mean"],
                "uncovered_mean": metrics["uncovered_mean"],
                "stale_mean": metrics["stale_mean"],
            }
        )

    return policy, TrainResult(history=history)


def save_policy(path: str, policy: MLPPolicy, cfg: AdaptiveScanningConfig) -> None:
    torch.save({"state_dict": policy.state_dict(), "cfg": cfg.__dict__}, path)


def load_policy(path: str, device: str | None = None) -> tuple[MLPPolicy, AdaptiveScanningConfig]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=dev)
    cfg = AdaptiveScanningConfig(**ckpt["cfg"])
    env = CameraBudgetEnv(cfg)
    pol = MLPPolicy(env.observation_dim).to(dev)
    pol.load_state_dict(ckpt["state_dict"])
    pol.eval()
    return pol, cfg
