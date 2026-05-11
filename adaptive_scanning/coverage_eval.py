"""
Evaluation: coverage vs geometric union, largest connected component (LCC) coverage,
and mean age of scanned cells.

Runs N episodes (seeds) comparing trained MLP, random (uniform ON times, full daily budget),
greedy unseen, greedy unseen with progressive late-day rescan, and an offline oracle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.env import CameraBudgetEnv
from adaptive_scanning.oracle import (
    always_on_reachable_flat_cells,
    geometric_reachable_flat_cells,
    greedy_coverage_oracle_actions,
    random_full_daily_budget_actions,
)
from adaptive_scanning.policies import (
    BudgetAwareGreedyUnseenOnlyPolicy,
    BudgetAwareGreedyUnseenProgressiveRescanPolicy,
    Policy,
)


def largest_connected_component_size(mask: np.ndarray) -> int:
    """
    Size of the largest 4-connected component in a bool grid ``mask`` (shape ``ny, nx``).

    Scanned cells are treated as foreground; diagonal neighbors are not connected.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    ny, nx = int(mask.shape[0]), int(mask.shape[1])
    if not np.any(mask):
        return 0
    visited = np.zeros((ny, nx), dtype=bool)
    best = 0
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or visited[i, j]:
                continue
            stack = [(i, j)]
            visited[i, j] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    y2, x2 = y + dy, x + dx
                    if (
                        0 <= y2 < ny
                        and 0 <= x2 < nx
                        and mask[y2, x2]
                        and not visited[y2, x2]
                    ):
                        visited[y2, x2] = True
                        stack.append((y2, x2))
            if size > best:
                best = size
    return int(best)


@dataclass
class EpisodeMetrics:
    """Per-episode metrics after one full rollout."""

    # Scanned cells / union of sector cells if camera were on at every moving step (no budget cap).
    coverage_vs_theoretic_max: float
    # Scanned cells / (nx * ny).
    coverage_of_full_grid: float
    # Mean sim_time_end - last_seen over finite cells (seconds).
    mean_age_scanned_sec: float

    final_uncovered_fraction: float
    final_mean_stale_normalized: float
    # Sum of ``dt_s`` over moving intervals where ``camera_on_effective`` (actually recording).
    effective_camera_on_sec: float
    n_cells_policy: int
    n_cells_theoretic_max: int
    n_cells_always_on_budgeted: int
    # Largest 4-connected scanned component size / geometric union cell count.
    lcc_cells: int
    lcc_vs_theoretic_max: float
    total_reward: float
    steps: int


class PrecomputedActionPolicy:
    """Plays a fixed action list (one entry per env step index ``env_step_idx``)."""

    def __init__(self, actions: list[int]):
        self._actions = actions

    def act(self, obs: np.ndarray, info: dict) -> int:
        if not bool(info.get("interval_is_moving", True)):
            return 0
        k = int(info.get("env_step_idx", 0))
        if k >= len(self._actions):
            return 0
        return int(self._actions[k])


def _mean_age_scanned_sec(env: CameraBudgetEnv) -> float:
    assert env.last_seen is not None
    ls = env.last_seen
    T = float(env._sim_time_s)
    finite = np.isfinite(ls)
    if not np.any(finite):
        return float("nan")
    ages = T - ls[finite]
    return float(np.mean(ages))


def run_episode_metrics(env: CameraBudgetEnv, policy: Policy, *, seed: int) -> EpisodeMetrics:
    """Single episode; mutates ``env``."""
    n_theory = len(geometric_reachable_flat_cells(env.cfg, seed=seed))
    n_always_budgeted = len(always_on_reachable_flat_cells(env.cfg, seed=seed))

    obs, info = env.reset(seed=seed)
    total_r = 0.0
    effective_on_moving = 0
    moving_steps = 0
    steps = 0
    while True:
        is_moving = bool(info.get("interval_is_moving", True))
        if is_moving:
            moving_steps += 1
        a_req = int(policy.act(obs, info))
        a = a_req if is_moving else 0
        step = env.step(a)
        total_r += step.reward
        obs = step.observation
        info = step.info
        if is_moving and bool(info.get("camera_on_effective", False)):
            effective_on_moving += 1
        steps += 1
        if step.terminated or step.truncated:
            final_u = float(info["uncovered_fraction"])
            final_s = float(info["mean_stale_normalized"])
            break

    assert env.last_seen is not None
    ls = env.last_seen
    scanned_mask = np.isfinite(ls)
    n_pol = int(np.sum(scanned_mask))
    lcc_sz = largest_connected_component_size(scanned_mask)
    nxny = int(env.cfg.nx * env.cfg.ny)
    cov_full = float(n_pol) / max(nxny, 1)
    cov_vs = float(n_pol) / max(n_theory, 1)
    lcc_vs = float(lcc_sz) / max(n_theory, 1)
    age = _mean_age_scanned_sec(env)
    cam_sec = float(effective_on_moving) * float(env.cfg.dt_s)

    return EpisodeMetrics(
        coverage_vs_theoretic_max=cov_vs,
        coverage_of_full_grid=cov_full,
        mean_age_scanned_sec=age,
        final_uncovered_fraction=final_u,
        final_mean_stale_normalized=final_s,
        effective_camera_on_sec=cam_sec,
        n_cells_policy=n_pol,
        n_cells_theoretic_max=n_theory,
        n_cells_always_on_budgeted=n_always_budgeted,
        lcc_cells=lcc_sz,
        lcc_vs_theoretic_max=lcc_vs,
        total_reward=float(total_r),
        steps=int(steps),
    )


def _summarize(rows: list[EpisodeMetrics]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [
        "coverage_vs_theoretic_max",
        "coverage_of_full_grid",
        "mean_age_scanned_sec",
        "lcc_vs_theoretic_max",
        "final_uncovered_fraction",
        "final_mean_stale_normalized",
        "effective_camera_on_sec",
        "total_reward",
    ]
    out: dict[str, float] = {}
    for k in keys:
        v = np.array([getattr(r, k) for r in rows], dtype=np.float64)
        if k == "mean_age_scanned_sec":
            v = v[np.isfinite(v)]
            if v.size == 0:
                out[f"{k}_mean"] = float("nan")
                out[f"{k}_std"] = float("nan")
                continue
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
    return out


def run_benchmark(
    cfg: AdaptiveScanningConfig,
    *,
    checkpoint_path: str | Path | None,
    n_scenarios: int,
    seed0: int,
    device: str | None = None,
    oracle_per_day_budget: bool = True,
) -> dict[str, Any]:
    """
    Run ``n_scenarios`` episodes (seeds ``seed0``, ``seed0+1``, …) per method.

    Methods: ``trained``, ``random`` (uniform random ON steps per day, **full** daily quota),
    ``greedy_unseen``, ``greedy_unseen_progressive`` (``BudgetAwareGreedyUnseenProgressiveRescanPolicy``),
    ``oracle``.

    **Greedy unseen vs oracle:** ``BudgetAwareGreedyUnseenOnlyPolicy`` is an **online**
    heuristic (foot-cell + wedge grace + remaining budget); it does **not** run offline
    set cover on the full trajectory. The oracle uses **future** geometry and greedy
    coverage + staleness + fill, so it typically **≥** greedy on union coverage (same metric).

    **Oracle budget (default):** ``oracle_per_day_budget=True`` matches the simulator’s
    daily reset (≤ ``R`` ON steps per calendar day), so **effective** camera seconds match
    the budget cap. Pass ``oracle_per_day_budget=False`` only for the looser global pool
    (can exceed ``R`` on one day and clamp on replay).
    """
    from adaptive_scanning.training import load_policy

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    trained_policy: Policy | None = None
    base_cfg = cfg
    if checkpoint_path is not None and checkpoint_path.is_file():
        trained_policy, base_cfg = load_policy(str(checkpoint_path), device=device)

    methods: dict[str, list[EpisodeMetrics]] = {
        "trained": [],
        "random": [],
        "greedy_unseen": [],
        "greedy_unseen_progressive": [],
        "oracle": [],
    }

    for i in range(n_scenarios):
        seed = int(seed0 + i)

        if trained_policy is not None:
            env_i = CameraBudgetEnv(base_cfg, seed=seed)
            methods["trained"].append(run_episode_metrics(env_i, trained_policy, seed=seed))

        rnd_actions = random_full_daily_budget_actions(base_cfg, seed=seed)
        env_i = CameraBudgetEnv(base_cfg, seed=seed)
        methods["random"].append(
            run_episode_metrics(env_i, PrecomputedActionPolicy(rnd_actions), seed=seed),
        )

        env_i = CameraBudgetEnv(base_cfg, seed=seed)
        methods["greedy_unseen"].append(
            run_episode_metrics(env_i, BudgetAwareGreedyUnseenOnlyPolicy(), seed=seed),
        )

        env_i = CameraBudgetEnv(base_cfg, seed=seed)
        methods["greedy_unseen_progressive"].append(
            run_episode_metrics(
                env_i, BudgetAwareGreedyUnseenProgressiveRescanPolicy(), seed=seed
            ),
        )

        actions = greedy_coverage_oracle_actions(
            base_cfg, seed=seed, per_day_budget=oracle_per_day_budget
        )
        env_i = CameraBudgetEnv(base_cfg, seed=seed)
        methods["oracle"].append(
            run_episode_metrics(env_i, PrecomputedActionPolicy(actions), seed=seed),
        )

    summary: dict[str, Any] = {}
    for name, rows in methods.items():
        if rows:
            summary[name] = _summarize(rows)
        else:
            summary[name] = None

    return {
        "n_scenarios": n_scenarios,
        "seed0": seed0,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "dt_s": float(base_cfg.dt_s),
        "oracle_per_day_budget": bool(oracle_per_day_budget),
        "oracle_mode": (
            "per_day_feasible" if oracle_per_day_budget else "global_budget_pool"
        ),
        "eval_config_note": "When a checkpoint is provided, all methods use its saved ``AdaptiveScanningConfig`` so the budget and motion match training.",
        "summary": summary,
    }


def write_benchmark_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
