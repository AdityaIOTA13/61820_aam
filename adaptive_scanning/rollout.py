from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_scanning.env import CameraBudgetEnv
from adaptive_scanning.policies import Policy


@dataclass
class RolloutStats:
    total_reward: float
    steps: int
    final_uncovered_fraction: float
    final_mean_stale_normalized: float
    mean_reward: float
    camera_on_fraction: float


def run_episode(env: CameraBudgetEnv, policy: Policy, *, seed: int | None = None) -> RolloutStats:
    obs, info = env.reset(seed=seed)
    total_r = 0.0
    on_steps = 0
    steps = 0
    while True:
        a = policy.act(obs, info)
        if a == 1:
            on_steps += 1
        step = env.step(a)
        total_r += step.reward
        obs = step.observation
        info = step.info
        steps += 1
        if step.terminated or step.truncated:
            final_u = info["uncovered_fraction"]
            final_s = info["mean_stale_normalized"]
            break
    return RolloutStats(
        total_reward=total_r,
        steps=steps,
        final_uncovered_fraction=final_u,
        final_mean_stale_normalized=final_s,
        mean_reward=total_r / max(steps, 1),
        camera_on_fraction=on_steps / max(steps, 1),
    )


def eval_policy(
    env: CameraBudgetEnv,
    policy: Policy,
    *,
    n_episodes: int,
    seed0: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed0)
    finals_u: list[float] = []
    finals_s: list[float] = []
    rewards: list[float] = []
    on_frac: list[float] = []
    for i in range(n_episodes):
        s = int(rng.integers(0, 2**31 - 1))
        st = run_episode(env, policy, seed=s)
        finals_u.append(st.final_uncovered_fraction)
        finals_s.append(st.final_mean_stale_normalized)
        rewards.append(st.total_reward)
        on_frac.append(st.camera_on_fraction)
    return {
        "uncovered_mean": float(np.mean(finals_u)),
        "uncovered_std": float(np.std(finals_u)),
        "stale_mean": float(np.mean(finals_s)),
        "stale_std": float(np.std(finals_s)),
        "return_mean": float(np.mean(rewards)),
        "return_std": float(np.std(rewards)),
        "camera_on_mean": float(np.mean(on_frac)),
    }


def always_on_reference(env: CameraBudgetEnv, *, seed: int | None = None) -> RolloutStats:
    from adaptive_scanning.policies import AlwaysOnPolicy

    return run_episode(env, AlwaysOnPolicy(), seed=seed)
