"""
Offline oracle for camera scheduling (non-causal upper baseline).

**Phase 1 — coverage:** greedy set cover on the geometric union.

**Phase 2 — age / budget:** greedy staleness reduction on a virtual ``last_seen`` grid,
then uniform random fill until the pick cap — so the oracle usually **exhausts** its
allotted ON-step budget when moving steps remain.

**Random full daily budget:** ``random_full_daily_budget_actions`` assigns exactly
``min(R, moving_steps_per_day)`` random ON intervals per calendar day.

**Relaxation (global pool):** without ``per_day_budget``, more than ``R`` intervals may
fall on one calendar day.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.env import CameraBudgetEnv, daily_video_budget_seconds


def _max_camera_on_steps_per_day(cfg: AdaptiveScanningConfig) -> int:
    b = float(daily_video_budget_seconds(cfg))
    dt = float(cfg.dt_s)
    return max(0, int(np.floor(b / max(dt, 1e-9) + 1e-9)))


def _flat_indices_from_mask(mask: np.ndarray) -> set[int]:
    flat = np.flatnonzero(np.asarray(mask, dtype=bool))
    return set(int(x) for x in flat.tolist())


def always_on_reachable_flat_cells(cfg: AdaptiveScanningConfig, *, seed: int) -> set[int]:
    """Cells that ever receive a finite ``last_seen`` under always-on (moving steps only, same budget rules)."""
    from adaptive_scanning.policies import AlwaysOnPolicy

    env = CameraBudgetEnv(cfg, seed=seed)
    pol = AlwaysOnPolicy()
    obs, info = env.reset(seed=seed)
    while True:
        mov = bool(info.get("interval_is_moving", True))
        a = pol.act(obs, info) if mov else 0
        step = env.step(a)
        obs = step.observation
        info = step.info
        if step.terminated or step.truncated:
            break
    assert env.last_seen is not None
    ls = env.last_seen
    return set(int(i) for i in np.flatnonzero(np.isfinite(ls)).tolist())


def geometric_reachable_flat_cells(cfg: AdaptiveScanningConfig, *, seed: int) -> set[int]:
    """
    Union of all sector cells along the episode if the camera were on at every *moving*
    interval (ignores budget). Upper bound on distinct cells any schedule can scan.
    """
    env = CameraBudgetEnv(cfg, seed=seed)
    env.reset(seed=seed)
    assert env._traj_x is not None
    T = len(env._traj_x) - 1
    u: set[int] = set()
    for k in range(T):
        mv, m = env.coverage_mask_for_step_index(k)
        if mv:
            u |= _flat_indices_from_mask(m)
    return u


def _refresh_gain_last_seen(
    ls: np.ndarray,
    mask: np.ndarray,
    stamp: float,
    T_end: float,
) -> float:
    """Marginal reduction in end-of-episode age mass from scanning ``mask`` at ``stamp``."""
    gain = 0.0
    flat_ls = ls.ravel()
    flat_m = mask.ravel()
    for i in np.flatnonzero(flat_m):
        v = flat_ls[i]
        if not np.isfinite(v):
            gain += T_end - stamp
        else:
            fv = float(v)
            nv = max(fv, stamp)
            gain += max(0.0, (T_end - fv) - (T_end - nv))
    return float(gain)


def _virtual_last_seen(
    S: set[int],
    masks: list[np.ndarray],
    *,
    ny: int,
    nx: int,
    dt: float,
) -> np.ndarray:
    ls = np.full((ny, nx), -np.inf, dtype=np.float64)
    for k in sorted(S):
        stamp = float(k) * dt
        m = masks[k]
        ls[m] = np.maximum(ls[m], stamp)
    return ls


def greedy_coverage_oracle_actions(
    cfg: AdaptiveScanningConfig,
    *,
    seed: int,
    per_day_budget: bool = True,
) -> list[int]:
    """
    Build an action sequence (length ``len(traj)-1``) for one episode.

    **Phase 1:** greedy set cover for **novel** coverage (cells in geometric union ``U``).

    **Phase 2:** greedily reduce **staleness** (virtual ``last_seen``); then **uniform
    random fill** among eligible moving steps until the pick cap is reached so leftover
    budget slots are still used (rescans).

    **Default (``per_day_budget=True``):** at most ``R`` picks per calendar day; cap is
    ``sum_d min(R, moving_steps_on_day_d)`` — matches the env daily budget reset so replay
    uses the full allotted camera time.

    **``per_day_budget=False``:** global pool ``min(D×R, n_moving)``; may assign more than
    ``R`` intervals on one calendar day, so replay can **clamp** and waste planned ONs.
    """
    rng = np.random.default_rng(int(seed) + 20260207)

    env = CameraBudgetEnv(cfg, seed=seed)
    env.reset(seed=seed)
    assert env._traj_x is not None
    traj_n = len(env._traj_x)
    T = traj_n - 1
    if T <= 0:
        return []

    U = geometric_reachable_flat_cells(cfg, seed=seed)
    if not U:
        return [0] * T

    env.reset(seed=seed)
    masks: list[np.ndarray] = []
    moving: list[bool] = []
    for k in range(T):
        mv, m = env.coverage_mask_for_step_index(k)
        moving.append(mv)
        masks.append(np.asarray(m, dtype=bool))

    mask_sets: list[set[int]] = [_flat_indices_from_mask(masks[k]) & U for k in range(T)]

    R = _max_camera_on_steps_per_day(cfg)
    dt = float(cfg.dt_s)
    ny, nx = int(cfg.ny), int(cfg.nx)

    day_dur = float(cfg.day_duration_s)

    def day_id(k: int) -> int:
        return int((k * dt) // max(day_dur, 1e-9))

    num_calendar_days = day_id(T - 1) + 1
    total_slots_budget = num_calendar_days * R

    moving_per_day: dict[int, int] = defaultdict(int)
    n_moving = 0
    for k in range(T):
        if moving[k]:
            n_moving += 1
            moving_per_day[day_id(k)] += 1

    if per_day_budget:
        pick_cap = sum(min(R, moving_per_day.get(d, 0)) for d in range(num_calendar_days))
        day_remaining: dict[int, int] = {
            d: min(R, moving_per_day.get(d, 0)) for d in range(num_calendar_days)
        }
    else:
        pick_cap = min(total_slots_budget, n_moving)

    # --- Phase 1: coverage ---
    S: set[int] = set()
    covered: set[int] = set()

    def can_add(k: int) -> bool:
        if not moving[k] or k in S:
            return False
        if len(S) >= pick_cap:
            return False
        if per_day_budget:
            return day_remaining.get(day_id(k), 0) > 0
        return True

    while True:
        best_k: int | None = None
        best_gain = 0
        for k in range(T):
            if not can_add(k):
                continue
            gain = len(mask_sets[k] - covered)
            if gain > best_gain:
                best_gain = gain
                best_k = k
        if best_k is None or best_gain == 0:
            break
        S.add(best_k)
        covered |= mask_sets[best_k]
        if per_day_budget:
            d = day_id(best_k)
            day_remaining[d] = day_remaining.get(d, 0) - 1

    T_end = float(T) * dt

    # --- Phase 2a: staleness greedy ---
    while len(S) < pick_cap:
        eligible = [k for k in range(T) if can_add(k)]
        if not eligible:
            break
        ls_v = _virtual_last_seen(S, masks, ny=ny, nx=nx, dt=dt)
        best_k: int | None = None
        best_g = -1.0
        for k in eligible:
            stamp = float(k) * dt
            g = _refresh_gain_last_seen(ls_v, masks[k], stamp, T_end)
            if g > best_g + 1e-12:
                best_g = g
                best_k = k
        if best_k is None or best_g <= 1e-9:
            break
        S.add(best_k)
        if per_day_budget:
            d = day_id(best_k)
            day_remaining[d] = day_remaining.get(d, 0) - 1

    # --- Phase 2b: random fill to exhaust budget ---
    while len(S) < pick_cap:
        eligible = [k for k in range(T) if can_add(k)]
        if not eligible:
            break
        k = int(rng.choice(eligible))
        S.add(k)
        if per_day_budget:
            d = day_id(k)
            day_remaining[d] = day_remaining.get(d, 0) - 1

    actions = [0] * T
    for k in S:
        actions[k] = 1
    return actions


def random_full_daily_budget_actions(
    cfg: AdaptiveScanningConfig,
    *,
    seed: int,
) -> list[int]:
    """
    Per calendar day, choose ``min(R, n_moving_that_day)`` moving steps uniformly at
    random without replacement and set camera ON there — uses the **full** daily camera
    quota whenever enough motion exists that day (same ``R`` and ``day_id`` as env).
    """
    rng = np.random.default_rng(int(seed) + 911)
    env = CameraBudgetEnv(cfg, seed=seed)
    env.reset(seed=seed)
    assert env._traj_x is not None
    T = len(env._traj_x) - 1
    if T <= 0:
        return []

    R = _max_camera_on_steps_per_day(cfg)
    dt = float(cfg.dt_s)
    day_dur = float(cfg.day_duration_s)

    def day_id(k: int) -> int:
        return int((k * dt) // max(day_dur, 1e-9))

    num_calendar_days = day_id(T - 1) + 1
    by_day: dict[int, list[int]] = defaultdict(list)
    for k in range(T):
        mv, _ = env.coverage_mask_for_step_index(k)
        if mv:
            by_day[day_id(k)].append(k)

    actions = [0] * T
    for d in range(num_calendar_days):
        idxs = by_day.get(d, [])
        if not idxs:
            continue
        pick = min(R, len(idxs))
        chosen = rng.choice(idxs, size=pick, replace=False)
        for k in chosen:
            actions[int(k)] = 1
    return actions
