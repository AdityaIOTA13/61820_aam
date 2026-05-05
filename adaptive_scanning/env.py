from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_scanning.config import AdaptiveScanningConfig


def daily_video_budget_seconds(cfg: AdaptiveScanningConfig) -> float:
    """
    SI seconds of camera-on budget per daily reset.

    When ``video_budget_reference_walk_speed_m_s`` > 0, scales
    ``seconds_video_budget_per_day`` inversely with ``walk_speed_m_s`` so that
    ``walk_speed_m_s * budget`` (rough path metres recordable at full duty) is
    stable when speed changes. Reference 0 disables scaling.
    """
    b = float(cfg.seconds_video_budget_per_day)
    ref = float(getattr(cfg, "video_budget_reference_walk_speed_m_s", 0.0) or 0.0)
    if ref > 0.0:
        spd = max(float(cfg.walk_speed_m_s), 0.05)
        b *= ref / spd
    return max(b, 1e-9)
from adaptive_scanning.motion import generate_trajectory


def _wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _graph_to_world_scale_from_poly(
    poly: np.ndarray | None,
    world_w_m: float,
    world_h_m: float,
    *,
    margin: float = 1.0,
) -> float:
    """
    Uniform scale s such that delta_world ≈ s * delta_graph, matching
    ``street_trajectories.affine_map_points`` / trajectory letterboxing.

    ``scan_radius_m`` in config is in graph / ground metres (same space as
    Folium wedges); multiply by this scale for distance checks in env world
    coordinates.
    """
    if poly is None:
        return 1.0
    p = np.asarray(poly, dtype=np.float64)
    if p.size < 2 or p.shape[0] < 1:
        return 1.0
    gx = p[:, 0]
    gy = p[:, 1]
    gw = float(np.ptp(gx))
    gh = float(np.ptp(gy))
    if gw < 1e-9 and gh < 1e-9:
        return 1.0
    gw += 1e-6
    gh += 1e-6
    inner_w = float(world_w_m) - 2.0 * float(margin)
    inner_h = float(world_h_m) - 2.0 * float(margin)
    return min(inner_w / gw, inner_h / gh)


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
        self._last_seen_foot: np.ndarray | None = None
        self._trajectory_source: str = "box"
        self._polyline_graph_m: np.ndarray | None = None
        self._graph_crs: str | None = None
        self._osm_home_daily_meta: dict[str, Any] | None = None
        self._graph_to_world_scale: float = 1.0

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
                try_build_home_daily_episode_trajectory,
                try_build_single_leg_trajectory,
                try_build_street_trajectory,
            )

            place = c.osm_place.strip() or None
            bbox = c.osm_bbox
            if not place and bbox is None:
                from adaptive_scanning.street_trajectories import DEFAULT_OSM_PLACE

                place = DEFAULT_OSM_PLACE
            tr = None
            if getattr(c, "osm_daily_home_commute", False):
                tr = try_build_home_daily_episode_trajectory(
                    cache_dir=Path(c.osm_cache_dir),
                    place=place,
                    bbox=bbox,
                    rng=self.rng,
                    world_w_m=self.world_w_m,
                    world_h_m=self.world_h_m,
                    max_sim_time_s=float(c.max_sim_time_s),
                    day_duration_s=float(c.day_duration_s),
                    dt_s=float(c.dt_s),
                    speed_m_s=float(c.walk_speed_m_s),
                    walks_per_day=int(c.osm_walks_per_day),
                    same_home_next_day_p=float(c.osm_same_home_next_day_p),
                    repeat_destination_across_days_p=float(
                        getattr(c, "osm_repeat_destination_across_days_p", 0.35)
                    ),
                    repeat_prior_stops_recency=float(
                        getattr(c, "osm_repeat_prior_stops_recency", 1.0)
                    ),
                    network_type=c.osm_network_type,
                )
            elif c.osm_single_leg:
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
                if len(tr) == 6:
                    (
                        self._traj_x,
                        self._traj_y,
                        self._traj_h,
                        self._polyline_graph_m,
                        self._graph_crs,
                        meta,
                    ) = tr
                    self._osm_home_daily_meta = meta
                    self._trajectory_source = "osm_home_daily"
                elif len(tr) == 5:
                    (
                        self._traj_x,
                        self._traj_y,
                        self._traj_h,
                        self._polyline_graph_m,
                        self._graph_crs,
                    ) = tr
                    self._osm_home_daily_meta = None
                    self._trajectory_source = "osm_single_leg"
                else:
                    self._traj_x, self._traj_y, self._traj_h = tr  # type: ignore[assignment]
                    self._polyline_graph_m = None
                    self._graph_crs = None
                    self._osm_home_daily_meta = None
                    self._trajectory_source = "osm_streets"
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
                self._osm_home_daily_meta = None
        else:
            self._traj_x, self._traj_y, self._traj_h = self._fallback_box_trajectory(n_steps)
            self._trajectory_source = "box"
            self._polyline_graph_m = None
            self._graph_crs = None
            self._osm_home_daily_meta = None
        self._graph_to_world_scale = _graph_to_world_scale_from_poly(
            self._polyline_graph_m,
            self.world_w_m,
            self.world_h_m,
        )
        self._step_idx = 0
        self._sim_time_s = 0.0
        self._budget_s = float(daily_video_budget_seconds(c))
        self._day_start_s = 0.0
        self.last_seen = np.full((c.ny, c.nx), -np.inf, dtype=np.float64)
        self._last_seen_foot = np.full((c.ny, c.nx), -np.inf, dtype=np.float64)
        self._last_day_boundary_reward: float = 0.0

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

    def _maybe_new_day(self) -> float:
        """
        Advance calendar days when ``sim_time`` crosses ``day_duration_s`` boundaries.
        Returns reward adjustment (typically <= 0): unused daily budget penalty per completed day.
        """
        c = self.cfg
        extra = 0.0
        w_unused = float(getattr(c, "w_unused_budget_end_of_day", 0.0))
        bmax = float(daily_video_budget_seconds(c))
        while self._sim_time_s - self._day_start_s >= c.day_duration_s:
            if w_unused != 0.0 and bmax > 1e-9:
                unused_frac = max(0.0, min(1.0, float(self._budget_s) / bmax))
                extra -= w_unused * unused_frac
            self._day_start_s += c.day_duration_s
            self._budget_s = float(daily_video_budget_seconds(c))
        return float(extra)

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
        r_w = float(c.scan_radius_m) * float(self._graph_to_world_scale)
        return (dist <= r_w) & (np.abs(ang) <= half) & (dist >= 1e-3)

    def _sector_union_mask_along_interval(
        self,
        x0: float,
        y0: float,
        h0: float,
        x1: float,
        y1: float,
        h1: float,
    ) -> np.ndarray:
        """OR of forward-sector masks along the motion segment (same sampling as coverage update)."""
        c = self.cfg
        union = np.zeros((c.ny, c.nx), dtype=bool)
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            union |= self._sector_mask(x0, y0, h0)
            return union
        step_m = max(0.5 * float(c.resolution_m), 0.35)
        n = int(math.ceil(dist / step_m)) + 1
        n = max(2, min(40, n))
        dh = float(_wrap_pi(h1 - h0))
        for j in range(n):
            t = j / (n - 1) if n > 1 else 0.0
            ax = x0 + t * dx
            ay = y0 + t * dy
            hd = float(_wrap_pi(h0 + t * dh))
            union |= self._sector_mask(ax, ay, hd)
        return union

    def _sector_never_scanned_stats_next_interval(self) -> tuple[float, int, int]:
        """
        For the forward sector along the *next* motion interval (``_step_idx`` → ``_step_idx+1``):
        ``(fraction_never, n_never_cells, n_union_cells)`` where never means ``last_seen`` not finite.
        """
        assert self.last_seen is not None
        assert self._traj_x is not None
        assert self._traj_y is not None
        assert self._traj_h is not None
        if self._step_idx >= len(self._traj_x) - 1:
            return 0.0, 0, 0
        ax0 = float(self._traj_x[self._step_idx])
        ay0 = float(self._traj_y[self._step_idx])
        hd0 = float(self._traj_h[self._step_idx])
        i1 = min(self._step_idx + 1, len(self._traj_x) - 1)
        ax1 = float(self._traj_x[i1])
        ay1 = float(self._traj_y[i1])
        hd1 = float(self._traj_h[i1])
        if math.hypot(ax1 - ax0, ay1 - ay0) < 1e-6:
            return 0.0, 0, 0
        union = self._sector_union_mask_along_interval(ax0, ay0, hd0, ax1, ay1, hd1)
        if not np.any(union):
            return 0.0, 0, 0
        sub = self.last_seen[union]
        nu = int(sub.size)
        nn = int(np.sum(~np.isfinite(sub)))
        return float(nn / max(nu, 1)), nn, nu

    def _apply_sector_while_moving(
        self,
        x0: float,
        y0: float,
        h0: float,
        x1: float,
        y1: float,
        h1: float,
        stamp: float,
    ) -> None:
        """
        While the camera is on for one ``dt_s`` interval, the agent moves from
        (x0,y0,h0) toward (x1,y1,h1). Integrate sector coverage along that segment
        so ``last_seen`` matches holding the shutter for the full interval.
        """
        assert self.last_seen is not None
        c = self.cfg
        union = self._sector_union_mask_along_interval(x0, y0, h0, x1, y1, h1)
        if np.any(union):
            if bool(getattr(c, "update_last_seen_only_on_first_hit", False)):
                first = union & (~np.isfinite(self.last_seen))
                if np.any(first):
                    self.last_seen[first] = np.maximum(self.last_seen[first], stamp)
            else:
                self.last_seen[union] = np.maximum(self.last_seen[union], stamp)

    def _apply_foot_scan_stamp(self, ax: float, ay: float, stamp: float) -> None:
        """Stamp only the grid cell under the agent (not the full wedge) for foot-based policies."""
        assert self._last_seen_foot is not None
        c = self.cfg
        ix = int(ax // c.resolution_m)
        iy = int(ay // c.resolution_m)
        if not (0 <= iy < c.ny and 0 <= ix < c.nx):
            return
        if bool(getattr(c, "update_last_seen_only_on_first_hit", False)):
            if not np.isfinite(self._last_seen_foot[iy, ix]):
                self._last_seen_foot[iy, ix] = float(stamp)
        else:
            self._last_seen_foot[iy, ix] = float(
                np.maximum(self._last_seen_foot[iy, ix], float(stamp))
            )

    def _foot_cell_never_scanned_for_policy(self) -> bool:
        """
        True iff the agent's current grid cell has no prior **foot** scan stamp.

        Wedge updates to ``last_seen`` can mark cells ahead of the agent; this uses
        ``_last_seen_foot`` so ``greedy_unseen`` can keep scanning along a corridor until
        the foot cell itself has been stamped.
        """
        if self._last_seen_foot is None or self._traj_x is None:
            return True
        if self._step_idx >= len(self._traj_x):
            return False
        c = self.cfg
        ax = float(self._traj_x[self._step_idx])
        ay = float(self._traj_y[self._step_idx])
        ix = int(ax // c.resolution_m)
        iy = int(ay // c.resolution_m)
        if not (0 <= iy < c.ny and 0 <= ix < c.nx):
            return False
        return not np.isfinite(float(self._last_seen_foot[iy, ix]))

    def _greedy_unseen_evening_leniency(self) -> float:
        """
        Weight in [0, 1]: after ``evening_lenient_after_s_since_day_start`` seconds since
        ``_day_start_s`` (and with enough remaining video budget), relax greedy-unseen suppression.
        Ramps linearly over ``evening_lenient_ramp_s``; does not use ``day_duration_s``.
        """
        c = self.cfg
        t0 = float(getattr(c, "evening_lenient_after_s_since_day_start", -1.0))
        if t0 < 0.0:
            return 0.0
        ramp = float(getattr(c, "evening_lenient_ramp_s", 3600.0))
        ramp = max(ramp, 1e-6)
        bmax = float(daily_video_budget_seconds(c))
        bf = float(self._budget_s / max(bmax, 1e-6))
        min_bf = float(getattr(c, "evening_lenient_min_budget_frac", 0.0))
        if bf < min_bf - 1e-12:
            return 0.0
        t_day = float(self._sim_time_s - self._day_start_s)
        if t_day < t0 - 1e-12:
            return 0.0
        return float(np.clip((t_day - t0) / ramp, 0.0, 1.0))

    def _foot_cell_greedy_unseen_on(self) -> bool:
        """
        Per-step greedy camera recommendation: OFF if either:

        - **Foot**: ``_last_seen_foot`` on this cell is older than ``greedy_unseen_coverage_grace_seconds``
          / grace applies so very fresh foot stamps do not immediately suppress.

        - **Wedge revisit**: ``last_seen`` on this cell is older than an effective minimum age (see
          ``greedy_unseen_wedge_suppress_min_age_s``, scaled with ``dt_s``). Same-pass wedge on the
          corridor is typically only ~``dt_s`` old, so it does not suppress; returning days later
          (large ``sim_time`` gap) does.

        After the configured time since day start (see ``evening_lenient_*``), effective foot grace
        and wedge age thresholds are increased so ``prior_*`` triggers less often — more willing to
        scan stale cells when there is budget left to spend.
        """
        if self.last_seen is None or self._last_seen_foot is None or self._traj_x is None:
            return True
        if self._step_idx >= len(self._traj_x):
            return False
        c = self.cfg
        tau0 = max(float(getattr(c, "greedy_unseen_coverage_grace_seconds", 0.0)), 0.0)
        # Floor vs dt: a low user threshold + long dwell / camera-off bursts must not look like "revisit".
        min_wedge0 = max(
            float(getattr(c, "greedy_unseen_wedge_suppress_min_age_s", 3600.0)),
            10.0 * float(c.dt_s),
        )
        w_eve = float(self._greedy_unseen_evening_leniency())
        wm = float(getattr(c, "evening_lenient_wedge_suppress_age_mult", 1.0))
        fm = float(getattr(c, "evening_lenient_foot_grace_mult", 1.0))
        tau = tau0 * (1.0 + w_eve * max(0.0, fm - 1.0))
        min_wedge_s = min_wedge0 * (1.0 + w_eve * max(0.0, wm - 1.0))
        ax = float(self._traj_x[self._step_idx])
        ay = float(self._traj_y[self._step_idx])
        ix = int(ax // c.resolution_m)
        iy = int(ay // c.resolution_m)
        if not (0 <= iy < c.ny and 0 <= ix < c.nx):
            return False
        T = float(self._sim_time_s)
        lf = float(self._last_seen_foot[iy, ix])
        ls = float(self.last_seen[iy, ix])
        prior_foot = bool(np.isfinite(lf) and (T - lf > tau + 1e-9))
        prior_wedge = bool(np.isfinite(ls) and (T - ls > min_wedge_s + 1e-9))
        return not (prior_foot or prior_wedge)

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

    def _current_interval_is_moving(self) -> bool:
        assert self._traj_x is not None
        assert self._traj_y is not None
        if self._step_idx >= len(self._traj_x) - 1:
            return False
        ax0 = float(self._traj_x[self._step_idx])
        ay0 = float(self._traj_y[self._step_idx])
        ax1 = float(self._traj_x[self._step_idx + 1])
        ay1 = float(self._traj_y[self._step_idx + 1])
        return math.hypot(ax1 - ax0, ay1 - ay0) > 1e-6

    def _observation(self) -> np.ndarray:
        c = self.cfg
        assert self.last_seen is not None
        assert self._traj_x is not None

        ax = float(self._traj_x[self._step_idx])
        ay = float(self._traj_y[self._step_idx])
        hd = float(self._traj_h[self._step_idx])

        ix0 = int(ax // c.resolution_m)
        iy0 = int(ay // c.resolution_m)

        frac_day = (self._sim_time_s - self._day_start_s) / c.day_duration_s
        frac_day = float(np.clip(frac_day, 0.0, 1.0))
        tod = 2 * math.pi * frac_day

        bmax = float(daily_video_budget_seconds(c))
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
        if str(getattr(c, "observation_mode", "patch")) == "foot_cell":
            ls = float(self.last_seen[iy0, ix0]) if (0 <= iy0 < c.ny and 0 <= ix0 < c.nx) else float("nan")
            lf = (
                float(self._last_seen_foot[iy0, ix0])
                if (self._last_seen_foot is not None and 0 <= iy0 < c.ny and 0 <= ix0 < c.nx)
                else float("nan")
            )
            T = float(self._sim_time_s)
            min_wedge0 = max(
                float(getattr(c, "greedy_unseen_wedge_suppress_min_age_s", 3600.0)),
                10.0 * float(c.dt_s),
            )
            tau0 = max(float(getattr(c, "greedy_unseen_coverage_grace_seconds", 0.0)), 0.0)
            w_eve = float(self._greedy_unseen_evening_leniency())
            wm = float(getattr(c, "evening_lenient_wedge_suppress_age_mult", 1.0))
            fm = float(getattr(c, "evening_lenient_foot_grace_mult", 1.0))
            tau = tau0 * (1.0 + w_eve * max(0.0, fm - 1.0))
            min_wedge_s = min_wedge0 * (1.0 + w_eve * max(0.0, wm - 1.0))
            prior_foot = bool(np.isfinite(lf) and (T - lf > tau + 1e-9))
            prior_wedge = bool(np.isfinite(ls) and (T - ls > min_wedge_s + 1e-9))
            foot = np.array(
                [
                    1.0 if np.isfinite(ls) else 0.0,
                    float(min(1.0, max(0.0, (T - ls) / max(c.stale_ref_s, 1e-6)))) if np.isfinite(ls) else 0.0,
                    1.0 if np.isfinite(lf) else 0.0,
                    float(min(1.0, max(0.0, (T - lf) / max(c.stale_ref_s, 1e-6)))) if np.isfinite(lf) else 0.0,
                    1.0 if prior_foot else 0.0,
                    1.0 if prior_wedge else 0.0,
                    1.0 if (not (prior_foot or prior_wedge)) else 0.0,
                ],
                dtype=np.float32,
            )
            return np.concatenate([foot, glo], axis=0).astype(np.float32)

        pc = c.patch_cells
        half = pc // 2
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
                else:
                    # Out-of-map patch slots are not "uncovered world"; treat as covered so
                    # local greedy rules are not dominated by padding near bbox edges.
                    ch0[pi, pj] = 1.0
                    ch1[pi, pj] = 0.0
        patch = np.stack([ch0, ch1], axis=0).astype(np.float32)
        flat_patch = patch.reshape(-1)
        return np.concatenate([flat_patch, glo], axis=0).astype(np.float32)

    def _info_dict(self) -> dict[str, Any]:
        assert self.last_seen is not None
        sec_fr, sec_nn, sec_nu = self._sector_never_scanned_stats_next_interval()
        return {
            "sim_time_s": self._sim_time_s,
            "budget_s": self._budget_s,
            "seconds_video_budget_per_day_effective": float(daily_video_budget_seconds(self.cfg)),
            "uncovered_fraction": self._uncovered_fraction(),
            "mean_stale_normalized": self._mean_stale_normalized(),
            "n_scanned_cells": int(np.sum(np.isfinite(self.last_seen))),
            "interval_is_moving": bool(self._current_interval_is_moving()),
            "sector_never_scanned_fraction": float(sec_fr),
            "sector_never_scanned_cells": int(sec_nn),
            "sector_union_cells": int(sec_nu),
            "end_of_day_unused_budget_penalty": float(
                getattr(self, "_last_day_boundary_reward", 0.0)
            ),
            "foot_cell_never_scanned_for_policy": bool(self._foot_cell_never_scanned_for_policy()),
            "foot_cell_greedy_unseen_on": bool(self._foot_cell_greedy_unseen_on()),
            "greedy_unseen_evening_leniency": float(self._greedy_unseen_evening_leniency()),
        }

    def step(self, action: int) -> StepResult:
        c = self.cfg
        assert self.last_seen is not None
        assert self._traj_x is not None

        ax0 = float(self._traj_x[self._step_idx])
        ay0 = float(self._traj_y[self._step_idx])
        hd0 = float(self._traj_h[self._step_idx])
        stamp = float(self._sim_time_s)
        i1 = min(self._step_idx + 1, len(self._traj_x) - 1)
        ax1 = float(self._traj_x[i1])
        ay1 = float(self._traj_y[i1])
        hd1 = float(self._traj_h[i1])
        interval_is_moving = math.hypot(ax1 - ax0, ay1 - ay0) > 1e-6

        on = int(action) == 1
        budget_ok = self._budget_s >= c.dt_s - 1e-9
        actually_on = on and budget_ok and interval_is_moving

        if actually_on:
            self._apply_sector_while_moving(ax0, ay0, hd0, ax1, ay1, hd1, stamp)
            self._apply_foot_scan_stamp(ax0, ay0, stamp)
            self._budget_s -= c.dt_s

        self._sim_time_s += c.dt_s
        day_reward = self._maybe_new_day()
        self._last_day_boundary_reward = float(day_reward)

        self._step_idx += 1
        reward = self._reward()
        bonus = float(getattr(c, "reward_camera_on_bonus", 0.0))
        if actually_on and bonus != 0.0:
            reward += bonus
        reward += day_reward

        max_steps = len(self._traj_x) - 1
        truncated = self._step_idx >= max_steps
        terminated = False

        obs = self._observation() if not truncated else self._observation()
        info = self._info_dict()
        info["action_clamped"] = on and not budget_ok
        info["camera_on_effective"] = actually_on
        info["step_interval_is_moving"] = bool(interval_is_moving)
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
        if str(getattr(c, "observation_mode", "patch")) == "foot_cell":
            return 7 + 7
        return 2 * c.patch_cells * c.patch_cells + 7
