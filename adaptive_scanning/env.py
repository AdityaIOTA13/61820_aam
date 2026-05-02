from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.motion import generate_trajectory


def _wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class CameraBudgetEnv:
    """
    Grid world in metres. Each step: optional sector scan (consumes budget),
    time advances, agent follows a precomputed trajectory. Budget resets each
    ``day_duration_s`` of simulation time.
    """

    def __init__(self, cfg: AdaptiveScanningConfig | None = None, *, seed: int | None = None):
        self.cfg = cfg or AdaptiveScanningConfig()
        self.rng = np.random.default_rng(seed)
        self._traj_x: np.ndarray | None = None
        self._traj_y: np.ndarray | None = None
        self._traj_h: np.ndarray | None = None
        self._step_idx: int = 0
        self._sim_time_s: float = 0.0
        self._budget_s: float = 0.0
        self._day_start_s: float = 0.0
        self.last_seen: np.ndarray | None = None
        self._trajectory_source: str = "box"
        self._polyline_graph_m: np.ndarray | None = None
        self._graph_crs: str | None = None

    @property
    def world_w_m(self) -> float:
        return self.cfg.nx * self.cfg.resolution_m

    @property
    def world_h_m(self) -> float:
        return self.cfg.ny * self.cfg.resolution_m

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        c = self.cfg
        n_steps = int(c.max_sim_time_s / c.dt_s)

        if c.motion_mode == "streets":
            from pathlib import Path

            from adaptive_scanning.street_trajectories import (
                try_build_single_leg_trajectory,
                try_build_street_trajectory,
            )

            place = c.osm_place.strip() or None
            bbox = c.osm_bbox
            if not place and bbox is None:
                from adaptive_scanning.street_trajectories import DEFAULT_OSM_PLACE

                place = DEFAULT_OSM_PLACE
            if c.osm_single_leg:
                tr = try_build_single_leg_trajectory(
                    cache_dir=Path(c.osm_cache_dir),
                    place=place,
                    bbox=bbox,
                    rng=self.rng,
                    world_w_m=self.world_w_m,
                    world_h_m=self.world_h_m,
                    n_steps=n_steps,
                    speed_m_s=c.walk_speed_m_s,
                    dt_s=c.dt_s,
                    n_anchors=c.osm_anchor_nodes,
                    network_type=c.osm_network_type,
                )
            else:
                tr = try_build_street_trajectory(
                    cache_dir=Path(c.osm_cache_dir),
                    place=place,
                    bbox=bbox,
                    rng=self.rng,
                    world_w_m=self.world_w_m,
                    world_h_m=self.world_h_m,
                    n_steps=n_steps,
                    speed_m_s=c.walk_speed_m_s,
                    dt_s=c.dt_s,
                    n_anchors=c.osm_anchor_nodes,
                    anchor_reuse_bias=c.osm_anchor_reuse_bias,
                    network_type=c.osm_network_type,
                )
            if tr is not None:
                if c.osm_single_leg and len(tr) == 5:
                    (
                        self._traj_x,
                        self._traj_y,
                        self._traj_h,
                        self._polyline_graph_m,
                        self._graph_crs,
                    ) = tr
                else:
                    self._traj_x, self._traj_y, self._traj_h = tr  # type: ignore[assignment]
                    self._polyline_graph_m = None
                    self._graph_crs = None
                self._trajectory_source = (
                    "osm_single_leg" if c.osm_single_leg else "osm_streets"
                )
            else:
                warnings.warn(
                    "motion_mode='streets' but OSM trajectory failed (missing osmnx, "
                    "Overpass error, or empty graph). Using box-motion fallback. "
                    "Check: python -m adaptive_scanning.run_sim check-osm",
                    UserWarning,
                    stacklevel=2,
                )
                self._traj_x, self._traj_y, self._traj_h = self._fallback_box_trajectory(n_steps)
                self._trajectory_source = "box_fallback"
                self._polyline_graph_m = None
                self._graph_crs = None
        else:
            self._traj_x, self._traj_y, self._traj_h = self._fallback_box_trajectory(n_steps)
            self._trajectory_source = "box"
            self._polyline_graph_m = None
            self._graph_crs = None
        self._step_idx = 0
        self._sim_time_s = 0.0
        self._budget_s = c.seconds_video_budget_per_day
        self._day_start_s = 0.0
        self.last_seen = np.full((c.ny, c.nx), -np.inf, dtype=np.float64)

        obs = self._observation()
        info = self._info_dict()
        info["motion_mode"] = c.motion_mode
        info["trajectory_source"] = self._trajectory_source
        return obs, info

    def _fallback_box_trajectory(self, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        x0 = float(self.rng.uniform(2.0, self.world_w_m - 2.0))
        y0 = float(self.rng.uniform(2.0, self.world_h_m - 2.0))
        h0 = float(self.rng.uniform(-math.pi, math.pi))
        return generate_trajectory(
            length=n_steps,
            x0=x0,
            y0=y0,
            heading0=h0,
            world_w_m=self.world_w_m,
            world_h_m=self.world_h_m,
            speed_m_s=c.walk_speed_m_s,
            dt_s=c.dt_s,
            heading_noise_std=c.walk_heading_noise_rad,
            rng=self.rng,
        )

    def _maybe_new_day(self) -> None:
        c = self.cfg
        while self._sim_time_s - self._day_start_s >= c.day_duration_s:
            self._day_start_s += c.day_duration_s
            self._budget_s = c.seconds_video_budget_per_day

    def _cell_centers_world(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        ix = np.arange(c.nx, dtype=np.float64)
        iy = np.arange(c.ny, dtype=np.float64)
        wx = (ix + 0.5) * c.resolution_m
        wy = (iy + 0.5) * c.resolution_m
        wx_grid, wy_grid = np.meshgrid(wx, wy)
        return wx_grid, wy_grid

    def _sector_mask(
        self,
        ax: float,
        ay: float,
        heading: float,
    ) -> np.ndarray:
        """Boolean (ny, nx) cells whose centers lie in the forward sector."""
        c = self.cfg
        wx, wy = self._cell_centers_world()
        dx = wx - ax
        dy = wy - ay
        dist = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) - heading
        ang = _wrap_pi(ang)
        half = math.radians(0.5 * c.hfov_deg)
        return (dist <= c.scan_radius_m) & (np.abs(ang) <= half) & (dist >= 1e-3)

    def _uncovered_fraction(self) -> float:
        assert self.last_seen is not None
        never = ~np.isfinite(self.last_seen)
        return float(np.mean(never))

    def _mean_stale_normalized(self) -> float:
        assert self.last_seen is not None
        scanned = np.isfinite(self.last_seen)
        if not np.any(scanned):
            return 0.0
        age = self._sim_time_s - self.last_seen[scanned]
        return float(np.mean(np.clip(age, 0.0, None) / self.cfg.stale_ref_s))

    def _reward(self) -> float:
        c = self.cfg
        u = self._uncovered_fraction()
        s = self._mean_stale_normalized()
        return -c.w_uncovered * u - c.w_stale_scanned * s

    def _observation(self) -> np.ndarray:
        c = self.cfg
        assert self.last_seen is not None
        assert self._traj_x is not None

        ax = float(self._traj_x[self._step_idx])
        ay = float(self._traj_y[self._step_idx])
        hd = float(self._traj_h[self._step_idx])

        pc = c.patch_cells
        half = pc // 2
        ix0 = int(ax // c.resolution_m)
        iy0 = int(ay // c.resolution_m)

        ch0 = np.zeros((pc, pc), dtype=np.float32)
        ch1 = np.zeros((pc, pc), dtype=np.float32)
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                ii = iy0 + di
                jj = ix0 + dj
                pi, pj = di + half, dj + half
                if 0 <= ii < c.ny and 0 <= jj < c.nx:
                    ls = self.last_seen[ii, jj]
                    if np.isfinite(ls):
                        ch0[pi, pj] = 1.0
                        ch1[pi, pj] = min(
                            1.0,
                            max(0.0, (self._sim_time_s - ls) / c.stale_ref_s),
                        )

        frac_day = (self._sim_time_s - self._day_start_s) / c.day_duration_s
        frac_day = float(np.clip(frac_day, 0.0, 1.0))
        tod = 2 * math.pi * frac_day

        bmax = c.seconds_video_budget_per_day
        glo = np.array(
            [
                self._budget_s / max(bmax, 1e-6),
                math.sin(tod),
                math.cos(tod),
                math.sin(hd),
                math.cos(hd),
                ax / max(self.world_w_m, 1e-6),
                ay / max(self.world_h_m, 1e-6),
            ],
            dtype=np.float32,
        )
        patch = np.stack([ch0, ch1], axis=0).astype(np.float32)
        flat_patch = patch.reshape(-1)
        return np.concatenate([flat_patch, glo], axis=0).astype(np.float32)

    def _info_dict(self) -> dict[str, Any]:
        assert self.last_seen is not None
        return {
            "sim_time_s": self._sim_time_s,
            "budget_s": self._budget_s,
            "uncovered_fraction": self._uncovered_fraction(),
            "mean_stale_normalized": self._mean_stale_normalized(),
            "n_scanned_cells": int(np.sum(np.isfinite(self.last_seen))),
        }

    def step(self, action: int) -> StepResult:
        c = self.cfg
        assert self.last_seen is not None
        assert self._traj_x is not None

        ax0 = float(self._traj_x[self._step_idx])
        ay0 = float(self._traj_y[self._step_idx])
        hd0 = float(self._traj_h[self._step_idx])

        on = int(action) == 1
        budget_ok = self._budget_s >= c.dt_s - 1e-9
        actually_on = on and budget_ok

        if actually_on:
            m = self._sector_mask(ax0, ay0, hd0)
            self.last_seen[m] = np.maximum(self.last_seen[m], self._sim_time_s)
            self._budget_s -= c.dt_s

        self._sim_time_s += c.dt_s
        self._maybe_new_day()

        self._step_idx += 1
        reward = self._reward()

        max_steps = len(self._traj_x) - 1
        truncated = self._step_idx >= max_steps
        terminated = False

        obs = self._observation() if not truncated else self._observation()
        info = self._info_dict()
        info["action_clamped"] = on and not budget_ok
        info["camera_on_effective"] = actually_on
        info["agent_x_m"] = ax0
        info["agent_y_m"] = ay0
        info["agent_heading_rad"] = hd0
        info["action_requested"] = int(action)

        return StepResult(
            observation=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    @property
    def observation_dim(self) -> int:
        c = self.cfg
        return 2 * c.patch_cells * c.patch_cells + 7
