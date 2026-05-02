from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, obs: np.ndarray, info: dict) -> int:
        ...


class RandomPolicy:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def act(self, obs: np.ndarray, info: dict) -> int:
        return int(self.rng.integers(0, 2))


class AlwaysOnPolicy:
    def act(self, obs: np.ndarray, info: dict) -> int:
        return 1


class AlwaysOffPolicy:
    def act(self, obs: np.ndarray, info: dict) -> int:
        return 0


class GreedyLocalStalenessPolicy:
    """
    Turn camera ON when local mean staleness (from patch channel 1) exceeds
    threshold or uncovered mass in patch is high; respect budget via env clamping.
    """

    def __init__(self, *, stale_threshold: float = 0.35, uncovered_threshold: float = 0.08):
        self.stale_threshold = stale_threshold
        self.uncovered_threshold = uncovered_threshold

    def act(self, obs: np.ndarray, info: dict) -> int:
        c = obs.shape[0]
        patch_flat = obs[:-7]
        ch = int(np.sqrt(len(patch_flat) // 2))
        patch = patch_flat.reshape(2, ch, ch)
        ever = patch[0]
        stale_ch = patch[1]
        uncovered_local = float(np.mean(ever < 0.5))
        # stale channel is 0 for never-scanned; among "ever" cells use stale value
        mask = ever > 0.5
        if np.any(mask):
            local_stale = float(np.mean(stale_ch[mask]))
        else:
            local_stale = 0.0
        want_on = (local_stale >= self.stale_threshold) or (uncovered_local >= self.uncovered_threshold)
        return 1 if want_on else 0


class BudgetAwareGreedyPolicy(GreedyLocalStalenessPolicy):
    """Greedy but forces OFF when budget is nearly exhausted (fraction < dt fraction)."""

    def __init__(
        self,
        *,
        stale_threshold: float = 0.35,
        uncovered_threshold: float = 0.08,
        min_budget_frac_to_turn_on: float = 0.08,
    ):
        super().__init__(stale_threshold=stale_threshold, uncovered_threshold=uncovered_threshold)
        self.min_budget_frac_to_turn_on = min_budget_frac_to_turn_on

    def act(self, obs: np.ndarray, info: dict) -> int:
        budget_frac = float(obs[-7])
        if budget_frac < self.min_budget_frac_to_turn_on:
            return 0
        return super().act(obs, info)
