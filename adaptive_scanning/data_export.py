from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.env import CameraBudgetEnv
from adaptive_scanning.policies import RandomPolicy


def generate_synthetic_episode_batch(
    *,
    cfg: AdaptiveScanningConfig,
    n_episodes: int,
    seed: int = 0,
    policy: Any | None = None,
) -> dict[str, Any]:
    """
    Roll out episodes (default: random actions) and pack observations/rewards/actions
    into ragged lists for saving or offline training.
    """
    rng = np.random.default_rng(seed)
    pol = policy or RandomPolicy(rng=rng)
    env = CameraBudgetEnv(cfg, seed=seed)
    episodes: list[dict[str, Any]] = []
    for i in range(n_episodes):
        s = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=s)
        obs_l: list[np.ndarray] = [obs.copy()]
        act_l: list[int] = []
        rew_l: list[float] = []
        while True:
            a = pol.act(obs, info)
            st = env.step(a)
            act_l.append(a)
            rew_l.append(st.reward)
            obs = st.observation
            info = st.info
            obs_l.append(obs.copy())
            if st.terminated or st.truncated:
                break
        episodes.append(
            {
                "observations": np.stack(obs_l, axis=0),
                "actions": np.array(act_l, dtype=np.int64),
                "rewards": np.array(rew_l, dtype=np.float32),
            }
        )
    return {"config": cfg.__dict__, "episodes": episodes}


def save_episode_npz(path: str | Path, batch: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, Any] = {}
    flat["config_json"] = np.array([json.dumps(batch["config"])], dtype=object)
    for i, ep in enumerate(batch["episodes"]):
        flat[f"ep{i}_obs"] = ep["observations"]
        flat[f"ep{i}_actions"] = ep["actions"]
        flat[f"ep{i}_rewards"] = ep["rewards"]
    np.savez_compressed(path, **flat)
