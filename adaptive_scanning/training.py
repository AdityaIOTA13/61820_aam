from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
        dev = next(self.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            logits = self.forward_logits(x)
            return int(torch.argmax(logits, dim=-1).item())

    def act_stochastic(self, obs: np.ndarray) -> tuple[int, torch.Tensor]:
        dev = next(self.parameters()).device
        x = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
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


def _config_json_safe(cfg: AdaptiveScanningConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _training_log_start(log_dir: Path, meta: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    (log_dir / "metrics.jsonl").unlink(missing_ok=True)


def _training_log_epoch(log_dir: Path, epoch: int, row: dict[str, float]) -> None:
    payload = {"epoch": epoch, **row}
    with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _training_log_finish(log_dir: Path, history: list[dict[str, float]]) -> None:
    rows = [{"epoch": i + 1, **h} for i, h in enumerate(history)]
    with (log_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _write_training_curves_png(log_dir, rows)


def _write_training_curves_png(log_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    ret = [float(r["return_mean"]) for r in rows]
    unc = [float(r["uncovered_mean"]) for r in rows]
    st = [float(r["stale_mean"]) for r in rows]
    cam = [float(r.get("camera_on_mean", 0.0)) for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flat

    ax0.plot(ep, loss, color="C0", linewidth=1.2)
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("policy loss (total)")
    ax0.set_title("Loss")
    ax0.set_yscale("symlog", linthresh=1e-6)

    ax1.plot(ep, ret, color="C1", linewidth=1.2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mean total return (eval)")
    ax1.set_title("Eval return")

    ax2.plot(ep, cam, color="C4", linewidth=1.2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("fraction steps on (eval)")
    ax2.set_title("Eval camera-on (greedy)")
    ax2.set_ylim(0.0, 1.05)

    ax3.plot(ep, unc, color="C2", linewidth=1.2)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("uncovered fraction")
    ax3.set_title("Eval uncovered (lower is better)")
    ax3.set_ylim(0.0, 1.05)

    ax4.plot(ep, st, color="C3", linewidth=1.2)
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("mean stale (normalized)")
    ax4.set_title("Eval staleness")

    ax5.axis("off")

    fig.suptitle("REINFORCE training curves", fontsize=12)
    out = log_dir / "training_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def train_reinforce(
    cfg: AdaptiveScanningConfig,
    *,
    epochs: int = 40,
    episodes_per_epoch: int = 8,
    lr: float = 3e-4,
    gamma: float = 0.995,
    entropy_coef: float = 0.03,
    seed: int = 0,
    device: str | None = None,
    show_progress: bool = True,
    log_dir: str | Path | None = None,
) -> tuple[MLPPolicy, TrainResult]:
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = CameraBudgetEnv(cfg, seed=seed)
    obs_dim = env.observation_dim
    policy = MLPPolicy(obs_dim).to(dev)
    opt = optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []

    log_path: Path | None = Path(log_dir) if log_dir else None
    if log_path is not None:
        _training_log_start(
            log_path,
            {
                "seed": seed,
                "epochs": epochs,
                "episodes_per_epoch": episodes_per_epoch,
                "lr": lr,
                "gamma": gamma,
                "entropy_coef": entropy_coef,
                "device": dev,
                "config": _config_json_safe(cfg),
            },
        )

    tqdm_mod: Any = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            tqdm_mod = tqdm
        except ImportError:
            tqdm_mod = None

    pbar = None
    if tqdm_mod is not None:
        pbar = tqdm_mod(
            range(epochs),
            desc="REINFORCE",
            unit="epoch",
            dynamic_ncols=True,
            mininterval=0.25,
        )
    epoch_iter = pbar if pbar is not None else range(epochs)

    for ep in epoch_iter:
        if pbar is not None:
            pbar.set_postfix_str(f"ep {ep + 1}/{epochs} rollouts…")
            pbar.refresh()
        opt.zero_grad()
        batch_logp: list[torch.Tensor] = []
        batch_ret: list[torch.Tensor] = []
        batch_entr: list[torch.Tensor] = []
        inner_it = range(episodes_per_epoch)
        if tqdm_mod is not None:
            inner_it = tqdm_mod(
                range(episodes_per_epoch),
                desc=f"rollouts ep {ep + 1}/{epochs}",
                unit="run",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.15,
            )
        for _ in inner_it:
            s = int(rng.integers(0, 2**31 - 1))
            obs, _info = env.reset(seed=s)
            logps: list[torch.Tensor] = []
            entrs: list[torch.Tensor] = []
            rews: list[float] = []
            while True:
                x = torch.from_numpy(obs).float().to(dev).unsqueeze(0)
                logits = policy.forward_logits(x)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logps.append(dist.log_prob(a))
                entrs.append(dist.entropy())
                step = env.step(int(a.item()))
                rews.append(step.reward)
                obs = step.observation
                if step.terminated or step.truncated:
                    break
            returns = _discounted_returns(rews, gamma)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=dev)
            batch_logp.append(torch.stack(logps))
            batch_ret.append(returns_t)
            batch_entr.append(torch.stack(entrs))

        if not batch_logp:
            continue
        if pbar is not None:
            pbar.set_postfix_str(f"ep {ep + 1}/{epochs} backward+step…")
            pbar.refresh()
        logp_stack = torch.cat(batch_logp)
        entr_stack = torch.cat(batch_entr)
        ret_raw = torch.cat(batch_ret)
        ret_stack = (ret_raw - ret_raw.mean()) / (ret_raw.std() + 1e-6)
        loss_pg = -(logp_stack * ret_stack).mean()
        loss = loss_pg - entropy_coef * entr_stack.mean()
        loss.backward()
        opt.step()

        if pbar is not None:
            pbar.set_postfix_str(f"ep {ep + 1}/{epochs} eval…")
            pbar.refresh()
        with torch.no_grad():
            metrics = eval_policy(env, policy, n_episodes=4, seed0=ep + 1000)
        row = {
            "loss": float(loss.item()),
            "return_mean": metrics["return_mean"],
            "uncovered_mean": metrics["uncovered_mean"],
            "stale_mean": metrics["stale_mean"],
            "camera_on_mean": metrics["camera_on_mean"],
        }
        history.append(row)
        if log_path is not None:
            _training_log_epoch(log_path, ep + 1, row)
        if pbar is not None:
            pbar.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                R=f"{float(metrics['return_mean']):.3f}",
                unc=f"{float(metrics['uncovered_mean']):.3f}",
                on=f"{float(metrics['camera_on_mean']):.2f}",
            )

    if pbar is not None:
        pbar.close()

    if log_path is not None and history:
        _training_log_finish(log_path, history)

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
