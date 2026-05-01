"""
project_coverage.py

Maps subsampled equirectangular depth to world (x, y) metres and accumulates
per-cell last observation time from frame_positions.json.

Default: **back-project** depth along rays to 3D, then use horizontal (x, y) —
counts walls, poles, ground, etc. Optional **ground_plane** mode uses z=0
intersection + occlusion (stricter floor-only).

Requires outputs/frames/*.jpg, outputs/depth_maps/*_depth.png, and
outputs/frame_positions.json (see README / R2 CDN).

Run from repo root:
  python scripts/project_coverage.py
  python scripts/project_coverage.py --max-frames 20 --resolution 0.25
  python scripts/project_coverage.py --fov-full-360
  python scripts/project_coverage.py --budget-seconds 18
  python scripts/project_coverage.py --hfov-deg 70 --vfov-deg 86
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from courtyard_geom import (
    IMG_H_PX,
    IMG_W_PX,
    PX_PER_M,
    RAYBAN_META_DEFAULT_HFOV_DEG,
    RAYBAN_META_DEFAULT_VFOV_DEG,
    REAL_W_M,
    backproject_xy_hits,
    camera_heading_from_trajectory,
    ground_plane_hits,
    load_depth_norm,
    parse_svg_waypoints,
)

FRAMES_DIR = ROOT / "outputs" / "frames"
DEPTH_DIR = ROOT / "outputs" / "depth_maps"
POSITIONS_JSON = ROOT / "outputs" / "frame_positions.json"
OUT_DIR = ROOT / "outputs" / "coverage"
REPLAYS_PARENT = ROOT / "outputs" / "coverage_replays"

# Optional architectural image (same pixel frame as svg_path.svg)
FLOORPLAN_CANDIDATES = (
    "base_floorplan.jpg",
    "floorplan_walkpath.png",
    "floorplan_grid.png",
)


def load_positions(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def frame_stem(jpg_name: str) -> str:
    return Path(jpg_name).stem


def depth_path_for_frame(jpg_name: str) -> Path:
    return DEPTH_DIR / f"{frame_stem(jpg_name)}_depth.png"


def eligible_timeline_rows(
    positions: dict,
    frame_names: list[str],
) -> list[tuple[str, float, float, float]]:
    """(name, x_m, y_m, t_sec) sorted by time; only frames with JPEG + depth on disk."""
    rows: list[tuple[str, float, float, float]] = []
    for name in frame_names:
        if name not in positions:
            continue
        if not (FRAMES_DIR / name).exists():
            continue
        if not depth_path_for_frame(name).exists():
            continue
        p = positions[name]
        rows.append(
            (name, float(p["x_meters"]), float(p["y_meters"]), float(p["timestamp_sec"]))
        )
    rows.sort(key=lambda r: r[3])
    return rows


def heading_by_name_from_rows(
    rows: list[tuple[str, float, float, float]],
) -> dict[str, float]:
    """Camera heading (rad) per frame name from full timeline (path tangent)."""
    if not rows:
        return {}
    xy = [(r[1], r[2]) for r in rows]
    heads = camera_heading_from_trajectory(xy)
    return {rows[i][0]: heads[i] for i in range(len(rows))}


def select_frames_for_budget(
    rows: list[tuple[str, float, float, float]],
    budget_seconds: float,
    policy: str,
    *,
    seed: int = 0,
) -> tuple[list[str], dict]:
    """
    Budget replay: each *kept* frame costs mean Δt between consecutive timeline samples
    (~1/fps). Choose up to K = floor(budget / mean_dt) frames, then subsample by policy.

    Returns (selected names in time order, info dict).
    """
    n = len(rows)
    if n == 0:
        return [], {"k_target": 0, "mean_dt": 0.0, "budget_seconds": budget_seconds}
    times = [r[3] for r in rows]
    t_lo, t_hi = min(times), max(times)
    if n == 1:
        mean_dt = 0.5
    else:
        mean_dt = (t_hi - t_lo) / float(n - 1)
    if mean_dt <= 0:
        mean_dt = 0.5

    total_span = t_hi - t_lo
    if budget_seconds + 1e-6 >= total_span:
        names_all = [r[0] for r in rows]
        return names_all, {
            "k_target": n,
            "mean_dt": mean_dt,
            "budget_seconds": budget_seconds,
            "note": "budget >= walk span; using all frames",
        }

    k = int(math.floor(budget_seconds / mean_dt + 1e-9))
    k = max(1, min(n, k))

    if policy == "prefix":
        chosen_idx = list(range(k))
    elif policy == "uniform":
        raw = np.linspace(0, n - 1, num=k)
        chosen_idx = sorted({int(round(float(x))) for x in raw})
        j = 0
        while len(chosen_idx) < k and j < n:
            if j not in chosen_idx:
                chosen_idx.append(j)
            j += 1
        chosen_idx = sorted(chosen_idx)[:k]
    elif policy == "random":
        rng = np.random.default_rng(seed)
        chosen_idx = sorted(rng.choice(n, size=k, replace=False).tolist())
    else:
        raise ValueError(f"Unknown replay policy: {policy}")

    selected = [rows[i][0] for i in chosen_idx]
    return selected, {
        "k_target": k,
        "mean_dt": mean_dt,
        "budget_seconds": budget_seconds,
        "policy": policy,
        "seed": seed,
    }


def _wrap_angle_rad(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def floor_grid_layout(resolution_m: float, margin_m: float) -> tuple[float, float, int, int]:
    """Same metre grid as ``run_map_age`` (fixed floorplan width + image aspect height)."""
    real_h_m = IMG_H_PX / PX_PER_M
    w_m = REAL_W_M + 2 * margin_m
    h_m = real_h_m + 2 * margin_m
    nx = max(1, int(math.ceil(w_m / resolution_m)))
    ny = max(1, int(math.ceil(h_m / resolution_m)))
    x0 = -margin_m
    y0 = -margin_m
    return x0, y0, nx, ny


def neighborhood_mean_age_disk(
    last_seen: np.ndarray,
    ts: float,
    *,
    cx: float,
    cy: float,
    x0: float,
    y0: float,
    resolution_m: float,
    nx: int,
    ny: int,
    radius_m: float,
    prior_unseen_age_s: float,
) -> float:
    """Mean map age at time ``ts`` over cells whose centres lie within ``radius_m`` of ``(cx, cy)``."""
    cx_cells = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * resolution_m
    cy_cells = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * resolution_m
    cc, zz = np.meshgrid(cx_cells, cy_cells)
    dist2 = (cc - cx) ** 2 + (zz - cy) ** 2
    mask = dist2 <= radius_m * radius_m + 1e-12
    if not np.any(mask):
        return float(prior_unseen_age_s)
    ls = last_seen[mask]
    seen = np.isfinite(ls)
    ages = np.where(seen, ts - ls, prior_unseen_age_s)
    return float(np.mean(ages))


def greedy_location_needs_coverage(
    last_seen: np.ndarray,
    cx: float,
    cy: float,
    *,
    x0: float,
    y0: float,
    resolution_m: float,
    nx: int,
    ny: int,
    unseen_scope: str,
    radius_m: float,
) -> bool:
    """
    True iff the camera is in a place not yet on the coverage map (never observed by a kept frame).

    ``foot``: the grid cell containing ``(cx, cy)`` has never received a depth hit.
    ``disk``: some cell whose centre lies within ``radius_m`` of ``(cx, cy)`` has never been hit.
    Pose off the grid bounds counts as needing coverage.
    """
    if unseen_scope == "disk":
        cx_cells = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * resolution_m
        cy_cells = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * resolution_m
        cc, zz = np.meshgrid(cx_cells, cy_cells)
        dist2 = (cc - cx) ** 2 + (zz - cy) ** 2
        mask = dist2 <= radius_m * radius_m + 1e-12
        if not np.any(mask):
            return True
        sub = last_seen[mask]
        return bool(np.any(np.isneginf(sub)))

    ix = int(math.floor((cx - x0) / resolution_m))
    iy = int(math.floor((cy - y0) / resolution_m))
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        return True
    return bool(np.isneginf(last_seen[iy, ix]))


def greedy_location_stale(
    last_seen: np.ndarray,
    ts: float,
    cx: float,
    cy: float,
    *,
    x0: float,
    y0: float,
    resolution_m: float,
    nx: int,
    ny: int,
    unseen_scope: str,
    radius_m: float,
    stale_threshold_s: float,
) -> bool:
    """
    True iff **already-covered** geometry is stale: time since last hit ≥ ``stale_threshold_s``.

    Unseen cells are **not** treated as infinitely stale here (use ``greedy_location_needs_coverage``
    for coverage gaps). ``foot`` / ``disk`` use the same regions as the unseen helpers.
    """
    if stale_threshold_s <= 0:
        return False

    if unseen_scope == "foot":
        ix = int(math.floor((cx - x0) / resolution_m))
        iy = int(math.floor((cy - y0) / resolution_m))
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            return False
        ls = last_seen[iy, ix]
        if not np.isfinite(ls) or np.isneginf(ls):
            return False
        return (float(ts) - float(ls)) >= float(stale_threshold_s)

    cx_cells = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * resolution_m
    cy_cells = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * resolution_m
    cc, zz = np.meshgrid(cx_cells, cy_cells)
    dist2 = (cc - cx) ** 2 + (zz - cy) ** 2
    mask = dist2 <= radius_m * radius_m + 1e-12
    if not np.any(mask):
        return False
    sub = last_seen[mask]
    seen = np.isfinite(sub) & ~np.isneginf(sub)
    if not np.any(seen):
        return False
    ages = float(ts) - sub[seen].astype(np.float64)
    return float(np.max(ages)) >= float(stale_threshold_s)


def _accumulate_depth_hits_for_frame(
    last_seen: np.ndarray,
    *,
    name: str,
    cx: float,
    cy: float,
    ts: float,
    h: float,
    x0: float,
    y0: float,
    nx: int,
    ny: int,
    resolution_m: float,
    subsample: int,
    hfov_deg: float,
    vfov_deg: float,
    projection: str,
    heading_offset_rad: float = 0.0,
) -> tuple[bool, int]:
    """
    Merge one frame's depth hits into ``last_seen`` (in-place).

    ``heading_offset_rad`` rotates the camera azimuth used for rays (e.g. π to use the
    rear hemisphere of a 360° capture relative to the path tangent).

    Returns (had_jpeg_and_depth_file, ray_hit_count).
    """
    dp = depth_path_for_frame(name)
    fp = FRAMES_DIR / name
    if not dp.exists() or not fp.exists():
        return False, 0
    depth_norm = load_depth_norm(dp)
    h_ray = _wrap_angle_rad(float(h) + float(heading_offset_rad))
    if projection == "ground_plane":
        gx, gy = ground_plane_hits(
            depth_norm,
            cx,
            cy,
            h_ray,
            subsample=subsample,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
        )
    else:
        gx, gy = backproject_xy_hits(
            depth_norm,
            cx,
            cy,
            h_ray,
            subsample=subsample,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
        )
    if gx.size == 0:
        return True, 0
    ix = ((gx - x0) / resolution_m).astype(np.int64)
    iy = ((gy - y0) / resolution_m).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[valid], iy[valid]
    if ix.size == 0:
        return True, 0
    ts_arr = np.full(ix.shape, ts, dtype=np.float64)
    np.maximum.at(last_seen, (iy, ix), ts_arr)
    return True, int(ix.size)


def select_frames_greedy_local(
    full_rows: list[tuple[str, float, float, float]],
    budget_seconds: float,
    *,
    resolution_m: float,
    margin_m: float,
    subsample: int,
    hfov_deg: float,
    vfov_deg: float,
    projection: str,
    heading_by_name: dict[str, float] | None,
    greedy_trigger: str,
    greedy_unseen_scope: str,
    greedy_radius_m: float,
    greedy_mean_age_threshold_s: float,
    greedy_stale_threshold_s: float = 120.0,
    greedy_hysteresis: bool = True,
    greedy_stale_release_s: float | None = None,
    greedy_mean_age_release_s: float | None = None,
    prior_unseen_age_s: float = 86400.0 * 365.0,
    last_seen_initial: np.ndarray | None = None,
    heading_offset_rad: float = 0.0,
    timestamp_day_offset_sec: float = 0.0,
    after_keep_cb: Callable[[np.ndarray, str, float, float, float], None] | None = None,
    per_eligible_frame_cb: Callable[[np.ndarray, str, float, float, float, bool], None]
    | None = None,
    walk_eligible_tail_after_budget: bool = False,
    max_kept_frames: int | None = None,
) -> tuple[list[str], dict, np.ndarray]:
    """
    Walk the full eligible timeline in order; merge depth hits only for selected frames.

    ``greedy_trigger``:

    - ``unseen``: keep only when **coverage** is missing (see ``greedy_unseen_scope``).
    - ``unseen_or_stale``: keep when **uncovered OR** observed cells in the same scope are
      stale: ``(t - last_hit) ≥ greedy_stale_threshold_s`` (max age in disk, or foot cell).
    - ``mean_age``: legacy mean-age in disk vs ``greedy_mean_age_threshold_s`` (uses
      ``prior_unseen_age_s`` only for unseen cells inside the mean).

    **Hysteresis** (default ``greedy_hysteresis=True``): once capture arms, keep capturing on
    subsequent eligibles until the **local disk** (``greedy_radius_m``) is fully covered and
    staleness there falls below ``greedy_stale_release_s`` (for ``unseen_or_stale``), or mean
    age falls below ``greedy_mean_age_release_s`` (for ``mean_age``). This avoids one-frame ON
    bursts when the trigger flips off immediately after a merge. Disable with
    ``greedy_hysteresis=False`` for legacy stepping.

    Stops after ``K`` kept frames, with ``K`` from the same rule as ``select_frames_for_budget``.

    If ``walk_eligible_tail_after_budget`` is True, keeps walking every eligible frame after
    the budget is full (no more merges) so callers can e.g. animate the full path; invokes
    ``per_eligible_frame_cb`` for each eligible row with ``kept=False`` on those tail steps.

    If ``max_kept_frames`` is set, it caps how many frames may be kept this call (0 allowed);
    ``budget_seconds`` is still recorded in the info dict but does not set ``k_target`` in that
    case (multi-session shared daily budget).

    If ``last_seen_initial`` is set (same ``ny, nx`` as this layout), greedy starts from that
    coverage state instead of an empty map. ``timestamp_day_offset_sec`` is added to each
    frame's ``timestamp_sec`` when writing ``last_seen`` (multi-day simulation). Returns
    ``(selected, info, last_seen_out)``.
    """
    n = len(full_rows)
    if n == 0:
        x0e, y0e, nxe, nye = floor_grid_layout(resolution_m, margin_m)
        empty_ls = np.full((nye, nxe), -np.inf, dtype=np.float64)
        if last_seen_initial is not None and last_seen_initial.shape == empty_ls.shape:
            empty_ls[:, :] = last_seen_initial
        return [], {"k_target": 0, "mean_dt": 0.0, "budget_seconds": budget_seconds}, empty_ls
    times = [r[3] for r in full_rows]
    t_lo, t_hi = min(times), max(times)
    if n == 1:
        mean_dt = 0.5
    else:
        mean_dt = (t_hi - t_lo) / float(n - 1)
    if mean_dt <= 0:
        mean_dt = 0.5

    total_span = t_hi - t_lo
    if heading_by_name is not None:
        headings = [float(heading_by_name.get(name, 0.0)) for name, _, _, _ in full_rows]
    else:
        xy_list = [(r[1], r[2]) for r in full_rows]
        headings = camera_heading_from_trajectory(xy_list)

    x0, y0, nx, ny = floor_grid_layout(resolution_m, margin_m)
    last_seen = np.full((ny, nx), -np.inf, dtype=np.float64)
    if last_seen_initial is not None:
        if last_seen_initial.shape != last_seen.shape:
            raise ValueError("last_seen_initial shape mismatch for floor grid")
        last_seen[:, :] = last_seen_initial

    # Full merge only when budget covers the walk and caller did not cap selections.
    if (
        max_kept_frames is None
        and budget_seconds + 1e-6 >= total_span
    ):
        names_all = [r[0] for r in full_rows]
        rays = 0
        for (name, cx, cy, ts), h in zip(full_rows, headings, strict=True):
            dp = depth_path_for_frame(name)
            fp = FRAMES_DIR / name
            if not dp.exists() or not fp.exists():
                continue
            ts_eff = float(ts) + float(timestamp_day_offset_sec)
            _ok, hits = _accumulate_depth_hits_for_frame(
                last_seen,
                name=name,
                cx=cx,
                cy=cy,
                ts=ts_eff,
                h=float(h),
                x0=x0,
                y0=y0,
                nx=nx,
                ny=ny,
                resolution_m=resolution_m,
                subsample=subsample,
                hfov_deg=hfov_deg,
                vfov_deg=vfov_deg,
                projection=projection,
                heading_offset_rad=heading_offset_rad,
            )
            rays += hits
        return names_all, {
            "k_target": n,
            "mean_dt": mean_dt,
            "budget_seconds": budget_seconds,
            "policy": "greedy_local",
            "greedy_trigger": greedy_trigger,
            "greedy_unseen_scope": greedy_unseen_scope,
            "greedy_radius_m": greedy_radius_m,
            "greedy_mean_age_threshold_s": greedy_mean_age_threshold_s,
            "greedy_stale_threshold_s": greedy_stale_threshold_s,
            "note": "budget >= walk span; merging depth for all frames",
            "rays_accumulated": rays,
            "skipped_no_depth_or_jpeg": 0,
            "prior_unseen_age_s": prior_unseen_age_s,
        }, last_seen

    if budget_seconds + 1e-6 >= total_span:
        k_from_budget = n
    else:
        k_from_budget = max(1, min(n, int(math.floor(budget_seconds / mean_dt + 1e-9))))

    if max_kept_frames is not None:
        k_target = max(0, min(int(max_kept_frames), n))
    else:
        k_target = k_from_budget

    selected: list[str] = []
    rays = 0
    skipped_no_pair = 0
    st_rel_eff = (
        float(greedy_stale_release_s)
        if greedy_stale_release_s is not None
        else max(6.0, 0.28 * float(greedy_stale_threshold_s))
    )
    mean_rel_eff = (
        float(greedy_mean_age_release_s)
        if greedy_mean_age_release_s is not None
        else max(1.0, 0.42 * float(greedy_mean_age_threshold_s))
    )
    capture_latched = False

    for (name, cx, cy, ts), h in zip(full_rows, headings, strict=True):
        if not walk_eligible_tail_after_budget and len(selected) >= k_target:
            break
        dp = depth_path_for_frame(name)
        fp = FRAMES_DIR / name
        if not dp.exists() or not fp.exists():
            skipped_no_pair += 1
            continue

        ts_eff = float(ts) + float(timestamp_day_offset_sec)
        budget_full = len(selected) >= k_target

        trigger_ok = False
        if greedy_trigger == "unseen":
            trigger_ok = greedy_location_needs_coverage(
                last_seen,
                cx,
                cy,
                x0=x0,
                y0=y0,
                resolution_m=resolution_m,
                nx=nx,
                ny=ny,
                unseen_scope=greedy_unseen_scope,
                radius_m=greedy_radius_m,
            )
        elif greedy_trigger == "unseen_or_stale":
            nu = greedy_location_needs_coverage(
                last_seen,
                cx,
                cy,
                x0=x0,
                y0=y0,
                resolution_m=resolution_m,
                nx=nx,
                ny=ny,
                unseen_scope=greedy_unseen_scope,
                radius_m=greedy_radius_m,
            )
            st = greedy_location_stale(
                last_seen,
                ts_eff,
                cx,
                cy,
                x0=x0,
                y0=y0,
                resolution_m=resolution_m,
                nx=nx,
                ny=ny,
                unseen_scope=greedy_unseen_scope,
                radius_m=greedy_radius_m,
                stale_threshold_s=greedy_stale_threshold_s,
            )
            trigger_ok = bool(nu or st)
        else:
            local_mean = neighborhood_mean_age_disk(
                last_seen,
                ts_eff,
                cx=cx,
                cy=cy,
                x0=x0,
                y0=y0,
                resolution_m=resolution_m,
                nx=nx,
                ny=ny,
                radius_m=greedy_radius_m,
                prior_unseen_age_s=prior_unseen_age_s,
            )
            trigger_ok = local_mean >= greedy_mean_age_threshold_s

        nu_disk = greedy_location_needs_coverage(
            last_seen,
            cx,
            cy,
            x0=x0,
            y0=y0,
            resolution_m=resolution_m,
            nx=nx,
            ny=ny,
            unseen_scope="disk",
            radius_m=greedy_radius_m,
        )

        if greedy_hysteresis:
            if greedy_trigger == "unseen":
                release_ok = not nu_disk
            elif greedy_trigger == "unseen_or_stale":
                st_lo = greedy_location_stale(
                    last_seen,
                    ts_eff,
                    cx,
                    cy,
                    x0=x0,
                    y0=y0,
                    resolution_m=resolution_m,
                    nx=nx,
                    ny=ny,
                    unseen_scope=greedy_unseen_scope,
                    radius_m=greedy_radius_m,
                    stale_threshold_s=st_rel_eff,
                )
                release_ok = (not nu_disk) and (not st_lo)
            else:
                lm_rel = neighborhood_mean_age_disk(
                    last_seen,
                    ts_eff,
                    cx=cx,
                    cy=cy,
                    x0=x0,
                    y0=y0,
                    resolution_m=resolution_m,
                    nx=nx,
                    ny=ny,
                    radius_m=greedy_radius_m,
                    prior_unseen_age_s=prior_unseen_age_s,
                )
                release_ok = lm_rel < mean_rel_eff

            if budget_full:
                capture_latched = False
                will_capture = False
            else:
                if capture_latched and release_ok:
                    capture_latched = False
                if (not capture_latched) and trigger_ok:
                    capture_latched = True
                will_capture = capture_latched
        else:
            will_capture = (not budget_full) and trigger_ok

        if will_capture:
            _ok, hits = _accumulate_depth_hits_for_frame(
                last_seen,
                name=name,
                cx=cx,
                cy=cy,
                ts=ts_eff,
                h=float(h),
                x0=x0,
                y0=y0,
                nx=nx,
                ny=ny,
                resolution_m=resolution_m,
                subsample=subsample,
                hfov_deg=hfov_deg,
                vfov_deg=vfov_deg,
                projection=projection,
                heading_offset_rad=heading_offset_rad,
            )
            rays += hits
            selected.append(name)
            if after_keep_cb is not None:
                after_keep_cb(last_seen, name, cx, cy, ts_eff)

        if per_eligible_frame_cb is not None:
            per_eligible_frame_cb(last_seen, name, cx, cy, ts_eff, will_capture)

    return selected, {
        "k_target": k_target,
        "mean_dt": mean_dt,
        "budget_seconds": budget_seconds,
        "policy": "greedy_local",
        "greedy_trigger": greedy_trigger,
        "greedy_unseen_scope": greedy_unseen_scope,
        "greedy_radius_m": greedy_radius_m,
        "greedy_mean_age_threshold_s": greedy_mean_age_threshold_s,
        "greedy_stale_threshold_s": greedy_stale_threshold_s,
        "greedy_hysteresis": bool(greedy_hysteresis),
        "greedy_stale_release_s": float(st_rel_eff),
        "greedy_mean_age_release_s": float(mean_rel_eff),
        "rays_accumulated": rays,
        "skipped_no_depth_or_jpeg": skipped_no_pair,
        "prior_unseen_age_s": prior_unseen_age_s,
    }, last_seen


def load_floorplan_rgb(root: Path) -> tuple[np.ndarray, str]:
    """
    RGB uint8 (H, W, 3) in floorplan pixel coordinates.
    Prefers data/base_floorplan.jpg (or walkpath/grid PNG); else draws SVG path on white.
    """
    from PIL import Image, ImageDraw

    data_dir = root / "data"
    for name in FLOORPLAN_CANDIDATES:
        p = data_dir / name
        if p.exists():
            im = Image.open(p).convert("RGB")
            if im.size != (IMG_W_PX, IMG_H_PX):
                im = im.resize((IMG_W_PX, IMG_H_PX), Image.Resampling.LANCZOS)
            return np.asarray(im, dtype=np.uint8), str(p.relative_to(root))

    svg = data_dir / "svg_path.svg"
    pts = parse_svg_waypoints(svg)
    img = Image.new("RGB", (IMG_W_PX, IMG_H_PX), (248, 248, 252))
    dr = ImageDraw.Draw(img)
    if len(pts) >= 2:
        dr.line([tuple(p) for p in pts], fill=(55, 85, 120), width=6)
    return np.asarray(img, dtype=np.uint8), str(svg.relative_to(root)) + " (path only)"


def resample_grid_to_floorplan_pixels(
    age: np.ndarray,
    ever: np.ndarray,
    x0: float,
    y0: float,
    res: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map metre-grid age/ever to image pixels (v down), same convention as sync_gps_video."""
    vv, uu = np.meshgrid(
        np.arange(IMG_H_PX, dtype=np.float64),
        np.arange(IMG_W_PX, dtype=np.float64),
        indexing="ij",
    )
    x_m = (uu + 0.5) / PX_PER_M
    y_m = (IMG_H_PX - vv - 0.5) / PX_PER_M
    ix = np.floor((x_m - x0) / res).astype(np.int64)
    iy = np.floor((y_m - y0) / res).astype(np.int64)
    ins = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    age_px = np.full((IMG_H_PX, IMG_W_PX), np.nan, dtype=np.float64)
    ever_px = np.zeros((IMG_H_PX, IMG_W_PX), dtype=np.bool_)
    age_px[ins] = age[iy[ins], ix[ins]]
    ever_px[ins] = ever[iy[ins], ix[ins]]
    return age_px, ever_px


def metres_to_floorplan_pixel_xy(cx_m: float, cy_m: float) -> tuple[float, float]:
    """Floorplan pixel coordinates (x right, y down) for a pose in metres."""
    u = float(cx_m) * float(PX_PER_M) - 0.5
    v = float(IMG_H_PX) - float(cy_m) * float(PX_PER_M) - 0.5
    return u, v


def staleness_age_grid(last_seen: np.ndarray, t_now: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell staleness ``t_now - last_hit``; unseen cells are NaN. Returns (age, ever bool)."""
    ever = np.isfinite(last_seen) & (last_seen > -np.inf)
    age = np.full_like(last_seen, np.nan, dtype=np.float64)
    age[ever] = float(t_now) - last_seen[ever]
    return age, ever


def blend_staleness_on_floorplan_rgb(
    floor_rgb: np.ndarray,
    last_seen: np.ndarray,
    t_now: float,
    *,
    x0: float,
    y0: float,
    res: float,
    nx: int,
    ny: int,
    overlay_alpha: float,
    vmax_age_s: float,
) -> np.ndarray:
    """
    RGB uint8 (H,W,3) floorplan with RdYlGn_r staleness overlay on observed cells.
    ``vmax_age_s`` fixes the color scale (values above it saturate red).
    """
    import matplotlib as mpl_mod

    age, ever = staleness_age_grid(last_seen, t_now)
    age_px, ever_px = resample_grid_to_floorplan_pixels(age, ever, x0, y0, res, nx, ny)
    cmap_m = mpl_mod.colormaps["RdYlGn_r"]
    va = max(float(vmax_age_s), 1e-6)
    norm_age = np.zeros_like(age_px, dtype=np.float64)
    mask_age = ever_px & np.isfinite(age_px)
    norm_age[mask_age] = np.clip(age_px[mask_age] / va, 0.0, 1.0)
    rgba = cmap_m(norm_age)
    rgb_ov = (rgba[:, :, :3] * 255.0).astype(np.float32)
    base = floor_rgb.astype(np.float32)
    a = float(np.clip(overlay_alpha, 0.05, 0.95))
    out = base.copy()
    m = mask_age
    out[m] = base[m] * (1.0 - a) + rgb_ov[m] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def lerp_staleness_age_grids_for_viz(
    age0: np.ndarray,
    age1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Blend per-cell staleness ages for animation. NaN = never seen at that endpoint.
    Newly-seen cells (finite in age1 only) ramp in with ``alpha``; overlap uses linear lerp.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    out = np.full_like(age0, np.nan, dtype=np.float64)
    f0 = np.isfinite(age0)
    f1 = np.isfinite(age1)
    both = f0 & f1
    out[both] = (1.0 - a) * age0[both] + a * age1[both]
    only0 = f0 & ~f1
    out[only0] = (1.0 - a) * age0[only0]
    only1 = ~f0 & f1
    out[only1] = a * age1[only1]
    return out


def blend_staleness_grid_values_on_floorplan_rgb(
    floor_rgb: np.ndarray,
    age_grid: np.ndarray,
    ever_grid: np.ndarray,
    *,
    x0: float,
    y0: float,
    res: float,
    nx: int,
    ny: int,
    overlay_alpha: float,
    vmax_age_s: float,
) -> np.ndarray:
    """Staleness overlay from precomputed metre-grid ``age_grid`` (NaN unseen) and ``ever_grid`` bool."""
    import matplotlib as mpl_mod

    age_px, ever_px = resample_grid_to_floorplan_pixels(age_grid, ever_grid, x0, y0, res, nx, ny)
    cmap_m = mpl_mod.colormaps["RdYlGn_r"]
    va = max(float(vmax_age_s), 1e-6)
    norm_age = np.zeros_like(age_px, dtype=np.float64)
    mask_age = ever_px & np.isfinite(age_px)
    norm_age[mask_age] = np.clip(age_px[mask_age] / va, 0.0, 1.0)
    rgba = cmap_m(norm_age)
    rgb_ov = (rgba[:, :, :3] * 255.0).astype(np.float32)
    base = floor_rgb.astype(np.float32)
    a = float(np.clip(overlay_alpha, 0.05, 0.95))
    out = base.copy()
    m = mask_age
    out[m] = base[m] * (1.0 - a) + rgb_ov[m] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_coverage_on_floorplan_rgb(
    floor_rgb: np.ndarray,
    ever: np.ndarray,
    *,
    x0: float,
    y0: float,
    res: float,
    nx: int,
    ny: int,
    overlay_alpha: float,
) -> np.ndarray:
    """Magma coverage (ever seen) blended on floorplan."""
    import matplotlib as mpl_mod

    age_dummy = np.zeros_like(ever, dtype=np.float64)
    _, ever_px = resample_grid_to_floorplan_pixels(age_dummy, ever, x0, y0, res, nx, ny)
    cmap_c = mpl_mod.colormaps["magma"]
    cov_f = ever_px.astype(np.float64)
    rgba_c = cmap_c(cov_f)
    rgb_c = (rgba_c[:, :, :3] * 255.0).astype(np.float32)
    base = floor_rgb.astype(np.float32)
    a = float(np.clip(overlay_alpha, 0.05, 0.95))
    out = base.copy()
    out[ever_px] = base[ever_px] * (1.0 - a) + rgb_c[ever_px] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_pose_dot_on_rgb(
    rgb: np.ndarray,
    cx_m: float,
    cy_m: float,
    *,
    radius_px: int = 11,
    fill: tuple[int, int, int] = (235, 51, 41),
    outline: tuple[int, int, int] = (255, 255, 255),
    outline_width: int = 3,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    u, v = metres_to_floorplan_pixel_xy(cx_m, cy_m)
    im = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(im)
    r = int(radius_px)
    dr.ellipse((u - r, v - r, u + r, v + r), fill=fill, outline=outline, width=outline_width)
    return np.asarray(im)


def draw_caption_lines_on_rgb(
    rgb: np.ndarray,
    lines: list[str],
    *,
    margin_px: int = 12,
    line_px: int = 22,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    y = margin_px
    for line in lines:
        dr.text((margin_px + 1, y + 1), line, fill=(255, 255, 255), font=font)
        dr.text((margin_px, y), line, fill=(28, 28, 32), font=font)
        y += line_px
    return np.asarray(im)


def render_floorplan_staleness_pose_frame(
    last_seen: np.ndarray,
    t_now: float,
    cx_m: float,
    cy_m: float,
    *,
    resolution_m: float,
    margin_m: float,
    overlay_alpha: float = 0.52,
    vmax_age_s: float = 600.0,
    caption_lines: list[str] | None = None,
    include_coverage_panel: bool = False,
    root: Path | None = None,
) -> np.ndarray:
    """
    One RGB image: staleness on floorplan + pose dot. Optionally append a second panel
    with cumulative coverage (magma) for the same ``last_seen`` state.
    """
    root = root or ROOT
    x0, y0, nx, ny = floor_grid_layout(resolution_m, margin_m)
    if last_seen.shape != (ny, nx):
        raise ValueError(f"last_seen shape {last_seen.shape} != grid ({ny}, {nx})")
    floor_rgb, _ = load_floorplan_rgb(root)
    ever = np.isfinite(last_seen) & (last_seen > -np.inf)
    age_rgb = blend_staleness_on_floorplan_rgb(
        floor_rgb,
        last_seen,
        t_now,
        x0=x0,
        y0=y0,
        res=resolution_m,
        nx=nx,
        ny=ny,
        overlay_alpha=overlay_alpha,
        vmax_age_s=vmax_age_s,
    )
    age_rgb = draw_pose_dot_on_rgb(age_rgb, cx_m, cy_m)
    if include_coverage_panel:
        cov_rgb = blend_coverage_on_floorplan_rgb(
            floor_rgb,
            ever,
            x0=x0,
            y0=y0,
            res=resolution_m,
            nx=nx,
            ny=ny,
            overlay_alpha=overlay_alpha,
        )
        cov_rgb = draw_pose_dot_on_rgb(cov_rgb, cx_m, cy_m)
        out = np.concatenate([age_rgb, cov_rgb], axis=1)
    else:
        out = age_rgb
    if caption_lines:
        out = draw_caption_lines_on_rgb(out, caption_lines)
    return out


def render_floorplan_staleness_pose_frame_with_viz_age(
    last_seen: np.ndarray,
    age_viz: np.ndarray,
    ever_viz: np.ndarray,
    cx_m: float,
    cy_m: float,
    *,
    resolution_m: float,
    margin_m: float,
    overlay_alpha: float = 0.52,
    vmax_age_s: float = 600.0,
    caption_lines: list[str] | None = None,
    include_coverage_panel: bool = False,
    root: Path | None = None,
) -> np.ndarray:
    """Staleness panel from pre-blended ``age_viz`` / ``ever_viz``; coverage panel from ``last_seen``."""
    root = root or ROOT
    x0, y0, nx, ny = floor_grid_layout(resolution_m, margin_m)
    if last_seen.shape != (ny, nx):
        raise ValueError(f"last_seen shape {last_seen.shape} != grid ({ny}, {nx})")
    if age_viz.shape != (ny, nx) or ever_viz.shape != (ny, nx):
        raise ValueError("age_viz / ever_viz must match last_seen grid shape")
    floor_rgb, _ = load_floorplan_rgb(root)
    ever = np.isfinite(last_seen) & (last_seen > -np.inf)
    age_rgb = blend_staleness_grid_values_on_floorplan_rgb(
        floor_rgb,
        age_viz,
        ever_viz.astype(bool),
        x0=x0,
        y0=y0,
        res=resolution_m,
        nx=nx,
        ny=ny,
        overlay_alpha=overlay_alpha,
        vmax_age_s=vmax_age_s,
    )
    age_rgb = draw_pose_dot_on_rgb(age_rgb, cx_m, cy_m)
    if include_coverage_panel:
        cov_rgb = blend_coverage_on_floorplan_rgb(
            floor_rgb,
            ever,
            x0=x0,
            y0=y0,
            res=resolution_m,
            nx=nx,
            ny=ny,
            overlay_alpha=overlay_alpha,
        )
        cov_rgb = draw_pose_dot_on_rgb(cov_rgb, cx_m, cy_m)
        out = np.concatenate([age_rgb, cov_rgb], axis=1)
    else:
        out = age_rgb
    if caption_lines:
        out = draw_caption_lines_on_rgb(out, caption_lines)
    return out


def expand_revisit_staleness_keyframes_to_frames(
    keyframes: list[tuple[np.ndarray, float, float, float, list[str]]],
    substeps: int,
    *,
    resolution_m: float,
    margin_m: float,
    overlay_alpha: float,
    vmax_age_s: float,
    include_coverage_panel: bool,
    root: Path | None,
) -> list[np.ndarray]:
    """
    Turn discrete simulation keyframes into a longer GIF by linearly blending staleness age
    grids and pose between consecutive eligibles (time jumps no longer snap the colormap in one frame).
    """
    if not keyframes:
        return []
    sub = max(1, int(substeps))
    if sub == 1:
        return [
            render_floorplan_staleness_pose_frame(
                ls,
                t,
                cx,
                cy,
                resolution_m=resolution_m,
                margin_m=margin_m,
                overlay_alpha=overlay_alpha,
                vmax_age_s=vmax_age_s,
                caption_lines=cap,
                include_coverage_panel=include_coverage_panel,
                root=root,
            )
            for ls, t, cx, cy, cap in keyframes
        ]

    n = len(keyframes)
    n_out = (n - 1) * sub + 1
    frames: list[np.ndarray] = []
    for fi in range(n_out):
        lo, rem = divmod(fi, sub)
        if lo >= n - 1:
            ls, t, cx, cy, cap = keyframes[-1]
            frames.append(
                render_floorplan_staleness_pose_frame(
                    ls,
                    t,
                    cx,
                    cy,
                    resolution_m=resolution_m,
                    margin_m=margin_m,
                    overlay_alpha=overlay_alpha,
                    vmax_age_s=vmax_age_s,
                    caption_lines=cap,
                    include_coverage_panel=include_coverage_panel,
                    root=root,
                )
            )
        elif rem == 0:
            ls, t, cx, cy, cap = keyframes[lo]
            frames.append(
                render_floorplan_staleness_pose_frame(
                    ls,
                    t,
                    cx,
                    cy,
                    resolution_m=resolution_m,
                    margin_m=margin_m,
                    overlay_alpha=overlay_alpha,
                    vmax_age_s=vmax_age_s,
                    caption_lines=cap,
                    include_coverage_panel=include_coverage_panel,
                    root=root,
                )
            )
        else:
            ls0, t0, x0, y0, c0 = keyframes[lo]
            ls1, t1, x1, y1, c1 = keyframes[lo + 1]
            a = rem / sub
            age0, ever0 = staleness_age_grid(ls0, t0)
            age1, ever1 = staleness_age_grid(ls1, t1)
            age_v = lerp_staleness_age_grids_for_viz(age0, age1, a)
            ever_v = ever0 | ever1
            cx_ = (1.0 - a) * x0 + a * x1
            cy_ = (1.0 - a) * y0 + a * y1
            cap = c1 if a >= 0.5 else c0
            ls_cov = ls1 if a >= 0.5 else ls0
            frames.append(
                render_floorplan_staleness_pose_frame_with_viz_age(
                    ls_cov,
                    age_v,
                    ever_v,
                    cx_,
                    cy_,
                    resolution_m=resolution_m,
                    margin_m=margin_m,
                    overlay_alpha=overlay_alpha,
                    vmax_age_s=vmax_age_s,
                    caption_lines=cap,
                    include_coverage_panel=include_coverage_panel,
                    root=root,
                )
            )
    return frames


def save_animation_gif(frames: list[np.ndarray], path: Path, *, frame_ms: int = 220) -> None:
    from PIL import Image

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to save")
    imgs = [Image.fromarray(f) for f in frames]
    duration = max(int(frame_ms), 1)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def run_map_age(
    positions: dict,
    frame_names: list[str],
    *,
    resolution_m: float = 0.2,
    subsample: int = 8,
    margin_m: float = 1.0,
    hfov_deg: float = RAYBAN_META_DEFAULT_HFOV_DEG,
    vfov_deg: float = RAYBAN_META_DEFAULT_VFOV_DEG,
    projection: str = "backproject",
    heading_by_name: dict[str, float] | None = None,
    timeline_end_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Returns (last_seen_sec, ever_seen_mask, t_end_sec, meta).

    last_seen_sec: (ny, nx) max timestamp observing each cell; -inf if never.
    ever_seen_mask: bool (ny, nx)
    """
    x0, y0, nx, ny = floor_grid_layout(resolution_m, margin_m)

    rows = []
    for name in frame_names:
        if name not in positions:
            continue
        p = positions[name]
        rows.append((name, float(p["x_meters"]), float(p["y_meters"]), float(p["timestamp_sec"])))

    if not rows:
        raise ValueError("No frames matched frame_positions.json")
    if projection not in ("backproject", "ground_plane"):
        raise ValueError(f"Unknown projection: {projection}")

    rows.sort(key=lambda r: r[3])
    if heading_by_name is not None:
        headings = [float(heading_by_name.get(name, 0.0)) for name, _, _, _ in rows]
    else:
        xy_list = [(r[1], r[2]) for r in rows]
        headings = camera_heading_from_trajectory(xy_list)

    last_seen = np.full((ny, nx), -np.inf, dtype=np.float64)
    t_end = float(timeline_end_sec) if timeline_end_sec is not None else float(rows[-1][3])

    meta_hits = 0
    frames_depth_ok = 0
    for i, ((name, cx, cy, ts), h) in enumerate(zip(rows, headings)):
        ok, hits = _accumulate_depth_hits_for_frame(
            last_seen,
            name=name,
            cx=cx,
            cy=cy,
            ts=ts,
            h=h,
            x0=x0,
            y0=y0,
            nx=nx,
            ny=ny,
            resolution_m=resolution_m,
            subsample=subsample,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            projection=projection,
        )
        if ok:
            frames_depth_ok += 1
        meta_hits += hits

    ever = np.isfinite(last_seen) & (last_seen > -np.inf)
    meta = {
        "frames_used": len(rows),
        "frames_with_depth_files": frames_depth_ok,
        "rays_accumulated": meta_hits,
        "grid_nx": nx,
        "grid_ny": ny,
        "resolution_m": resolution_m,
        "origin_x0_m": x0,
        "origin_y0_m": y0,
        "t_end_sec": t_end,
        "frac_cells_ever_seen": float(ever.mean()) if ever.size else 0.0,
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "projection": projection,
        "last_frame_timestamp_sec": float(rows[-1][3]) if rows else t_end,
    }
    return last_seen, ever.astype(np.bool_), t_end, meta


def metadata_from_last_seen(
    last_seen: np.ndarray,
    *,
    resolution_m: float,
    margin_m: float,
    t_end: float,
    hfov_deg: float,
    vfov_deg: float,
    projection: str,
) -> dict:
    """Minimal ``meta`` dict for ``save_outputs`` from an in-memory ``last_seen`` grid."""
    x0, y0, nx, ny = floor_grid_layout(resolution_m, margin_m)
    if last_seen.shape != (ny, nx):
        raise ValueError(
            f"last_seen shape {last_seen.shape} != grid ({ny}, {nx}) for "
            f"resolution={resolution_m} margin={margin_m}"
        )
    ever = np.isfinite(last_seen) & (last_seen > -np.inf)
    return {
        "frames_used": 0,
        "frames_with_depth_files": 0,
        "rays_accumulated": 0,
        "grid_nx": nx,
        "grid_ny": ny,
        "resolution_m": resolution_m,
        "origin_x0_m": x0,
        "origin_y0_m": y0,
        "t_end_sec": float(t_end),
        "frac_cells_ever_seen": float(ever.mean()) if ever.size else 0.0,
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "projection": projection,
        "last_frame_timestamp_sec": float(t_end),
    }


def save_outputs(
    last_seen: np.ndarray,
    ever: np.ndarray,
    t_end: float,
    meta: dict,
    *,
    overlay_alpha: float = 0.52,
    skip_floorplan_overlay: bool = False,
    output_dir: Path | None = None,
    file_prefix: str = "",
    title_suffix: str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib as mpl_mod
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for PNG export. Install: pip install matplotlib"
        ) from e

    out = output_dir if output_dir is not None else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    pref = file_prefix.strip()
    res = meta["resolution_m"]
    nx, ny = meta["grid_nx"], meta["grid_ny"]
    x0, y0 = meta["origin_x0_m"], meta["origin_y0_m"]
    extent = [x0, x0 + nx * res, y0, y0 + ny * res]

    age = np.full_like(last_seen, np.nan, dtype=np.float64)
    seen = last_seen > -np.inf
    age[seen] = t_end - last_seen[seen]

    vmax_age = 1.0
    if np.any(seen):
        vmax_age = float(np.nanpercentile(age[seen], 95))
        if vmax_age <= 0:
            vmax_age = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(
        age,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=vmax_age,
    )
    ax.set_xlabel("x (m, floorplan frame)")
    ax.set_ylabel("y (m, floorplan frame)")
    ax.set_title(f"Map age at end of walk (s since last observation){title_suffix}")
    plt.colorbar(im, ax=ax, label="seconds")
    p_age = out / f"{pref}map_age_end.png"
    fig.tight_layout()
    fig.savefig(p_age, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    cov = ever.astype(np.float32)
    ax2.imshow(cov, origin="lower", extent=extent, aspect="equal", cmap="viridis", vmin=0, vmax=1)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title(f"Coverage (1 = cell observed at least once){title_suffix}")
    p_cov = out / f"{pref}coverage_ever.png"
    fig2.tight_layout()
    fig2.savefig(p_cov, dpi=150)
    plt.close(fig2)

    meta_path = out / f"{pref}coverage_meta.json"
    np.save(out / f"{pref}last_seen_sec.npy", last_seen)
    print(f"Wrote {p_age.relative_to(ROOT)}")
    print(f"Wrote {p_cov.relative_to(ROOT)}")
    print(f"Wrote {(out / f'{pref}last_seen_sec.npy').relative_to(ROOT)}")

    if skip_floorplan_overlay:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {meta_path.relative_to(ROOT)}")
        return

    from PIL import Image

    floor_rgb, fp_src = load_floorplan_rgb(ROOT)
    meta["floorplan_source"] = fp_src
    age_px, ever_px = resample_grid_to_floorplan_pixels(
        age, ever, x0, y0, res, nx, ny
    )
    a = float(np.clip(overlay_alpha, 0.05, 0.95))

    # --- Map age on floorplan (PIL + matplotlib colormap, native 1400x819) ---
    cmap_m = mpl_mod.colormaps["RdYlGn_r"]
    norm_age = np.zeros_like(age_px, dtype=np.float64)
    mask_age = ever_px & np.isfinite(age_px)
    norm_age[mask_age] = np.clip((age_px[mask_age] - 0.0) / vmax_age, 0.0, 1.0)
    rgba = cmap_m(norm_age)
    rgb_ov = (rgba[:, :, :3] * 255.0).astype(np.float32)
    base = floor_rgb.astype(np.float32)
    out_age = base.copy()
    m = mask_age
    out_age[m] = base[m] * (1.0 - a) + rgb_ov[m] * a
    p_overlay = out / f"{pref}map_age_on_floorplan.png"
    Image.fromarray(np.clip(out_age, 0, 255).astype(np.uint8)).save(p_overlay, quality=95)
    print(f"Wrote {p_overlay.relative_to(ROOT)}  (floorplan: {fp_src})")

    # --- Coverage on floorplan ---
    cmap_c = mpl_mod.colormaps["magma"]
    cov_f = ever_px.astype(np.float64)
    rgba_c = cmap_c(cov_f)
    rgb_c = (rgba_c[:, :, :3] * 255.0).astype(np.float32)
    out_cov = base.copy()
    out_cov[ever_px] = base[ever_px] * (1.0 - a) + rgb_c[ever_px] * a
    p_cov_fp = out / f"{pref}coverage_on_floorplan.png"
    Image.fromarray(np.clip(out_cov, 0, 255).astype(np.uint8)).save(p_cov_fp, quality=95)
    print(f"Wrote {p_cov_fp.relative_to(ROOT)}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Floorplane map-age from depth + frame_positions.json")
    ap.add_argument("--resolution", type=float, default=0.2, help="Grid cell size in metres")
    ap.add_argument("--subsample", type=int, default=8, help="Depth pixel stride (larger = faster)")
    ap.add_argument("--max-frames", type=int, default=0, help="If >0, only first N frames (by time)")
    ap.add_argument("--margin", type=float, default=1.0, help="Extra margin (m) around full floorplan bbox")
    ap.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.52,
        help="Opacity of heatmap over floorplan (0-1)",
    )
    ap.add_argument(
        "--no-floorplan-overlay",
        action="store_true",
        help="Skip map_age_on_floorplan.png / coverage_on_floorplan.png",
    )
    ap.add_argument(
        "--hfov-deg",
        type=float,
        default=RAYBAN_META_DEFAULT_HFOV_DEG,
        help="Horizontal FOV (deg) in camera +x forward frame; >=360 = no mask (full 360)",
    )
    ap.add_argument(
        "--vfov-deg",
        type=float,
        default=RAYBAN_META_DEFAULT_VFOV_DEG,
        help="Vertical FOV (deg) from horizontal plane (asin dz); default 179 is wide for equirect ground; >=180 disables",
    )
    ap.add_argument(
        "--fov-full-360",
        action="store_true",
        help="Use full equirectangular sphere (disable FOV masking)",
    )
    ap.add_argument(
        "--projection",
        choices=("backproject", "ground_plane"),
        default="backproject",
        help="backproject: 3D hit (x,y) incl. vertical structure; ground_plane: z=0 + occlusion",
    )
    ap.add_argument(
        "--budget-seconds",
        type=float,
        default=0.0,
        help="Budget replay: max ~seconds of capture at mean frame spacing (0 = use all eligible frames)",
    )
    ap.add_argument(
        "--replay-policy",
        choices=("greedy_local", "random", "uniform", "prefix"),
        default="greedy_local",
        help="How to pick frames under --budget-seconds (default: greedy_local; ignored if budget is 0)",
    )
    ap.add_argument("--replay-seed", type=int, default=0, help="RNG seed for replay-policy=random")
    ap.add_argument(
        "--greedy-trigger",
        choices=("unseen", "unseen_or_stale", "mean_age"),
        default="unseen",
        help="greedy_local: unseen = coverage gaps only; unseen_or_stale = coverage OR "
        "staleness (see --greedy-stale-threshold-s); mean_age = legacy mean in disk",
    )
    ap.add_argument(
        "--greedy-unseen-scope",
        choices=("foot", "disk"),
        default="foot",
        help="greedy_local + unseen: foot = cell under camera; disk = any unseen cell within --greedy-radius-m",
    )
    ap.add_argument(
        "--greedy-radius-m",
        type=float,
        default=4.0,
        help="Disk radius (m): mean_age trigger, or unseen+disk neighbourhood",
    )
    ap.add_argument(
        "--greedy-mean-age-threshold-s",
        type=float,
        default=8.0,
        help="greedy_local + mean_age: record when mean map age in the disk exceeds this (seconds)",
    )
    ap.add_argument(
        "--greedy-stale-threshold-s",
        type=float,
        default=120.0,
        help="greedy_local + unseen_or_stale: record if max (t - last_hit) in scope ≥ this (seconds); "
        "unseen cells do not count as stale (handled separately)",
    )
    ap.add_argument(
        "--no-greedy-hysteresis",
        action="store_true",
        help="greedy_local: one-frame capture steps only (no sustained burst until neighborhood is fresh)",
    )
    ap.add_argument(
        "--greedy-stale-release-s",
        type=float,
        default=None,
        help="greedy_local hysteresis: stop burst when disk staleness is below this (default auto)",
    )
    ap.add_argument(
        "--greedy-mean-age-release-s",
        type=float,
        default=None,
        help="greedy_local hysteresis for mean_age trigger (default auto)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output folder (default: outputs/coverage, or outputs/coverage_replays/... when budget>0)",
    )
    args = ap.parse_args()

    hfov = 400.0 if args.fov_full_360 else float(args.hfov_deg)
    vfov = 200.0 if args.fov_full_360 else float(args.vfov_deg)

    if not POSITIONS_JSON.exists():
        sys.exit(f"Missing {POSITIONS_JSON.relative_to(ROOT)}")

    positions = load_positions(POSITIONS_JSON)
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        sys.exit(
            f"No frames in {FRAMES_DIR.relative_to(ROOT)}/. "
            "Run extract_frames.py or download from R2 (README)."
        )

    names = [p.name for p in frame_paths if p.name in positions]
    names.sort(key=lambda n: positions[n]["timestamp_sec"])
    if args.max_frames > 0:
        names = names[: args.max_frames]

    full_rows = eligible_timeline_rows(positions, names)
    if not full_rows:
        sys.exit(
            f"No frames with both JPEG and depth in {FRAMES_DIR.relative_to(ROOT)}/ "
            f"and {DEPTH_DIR.relative_to(ROOT)}/."
        )

    heading_map = heading_by_name_from_rows(full_rows)

    if args.budget_seconds > 0:
        if args.replay_policy == "greedy_local":
            selected, rinfo, _greedy_last = select_frames_greedy_local(
                full_rows,
                args.budget_seconds,
                resolution_m=args.resolution,
                margin_m=args.margin,
                subsample=args.subsample,
                hfov_deg=hfov,
                vfov_deg=vfov,
                projection=args.projection,
                heading_by_name=heading_map,
                greedy_trigger=args.greedy_trigger,
                greedy_unseen_scope=args.greedy_unseen_scope,
                greedy_radius_m=args.greedy_radius_m,
                greedy_mean_age_threshold_s=args.greedy_mean_age_threshold_s,
                greedy_stale_threshold_s=args.greedy_stale_threshold_s,
                greedy_hysteresis=not args.no_greedy_hysteresis,
                greedy_stale_release_s=args.greedy_stale_release_s,
                greedy_mean_age_release_s=args.greedy_mean_age_release_s,
            )
        else:
            selected, rinfo = select_frames_for_budget(
                full_rows,
                args.budget_seconds,
                args.replay_policy,
                seed=args.replay_seed,
            )
        names_used = selected
        replay_subdir = f"budget_{args.budget_seconds:g}s_{args.replay_policy}"
        if args.replay_policy == "greedy_local":
            replay_subdir += f"_{args.greedy_trigger}"
            if args.greedy_trigger in ("unseen", "unseen_or_stale"):
                replay_subdir += f"_{args.greedy_unseen_scope}"
        if args.replay_policy == "random":
            replay_subdir += f"_seed{args.replay_seed}"
        out_dir = (
            Path(args.out_dir)
            if args.out_dir.strip()
            else (REPLAYS_PARENT / replay_subdir)
        )
    else:
        names_used = [r[0] for r in full_rows]
        rinfo = {}
        out_dir = Path(args.out_dir) if args.out_dir.strip() else OUT_DIR

    print(
        f"Grid res={args.resolution}m subsample={args.subsample} projection={args.projection} "
        f"frames={len(names_used)}/{len(full_rows)} eligible  FOV h={hfov:g} v={vfov:g} deg"
        + (" (full 360)" if args.fov_full_360 else "")
        + (
            (
                f"  replay budget={args.budget_seconds:g}s policy={args.replay_policy}"
                + (
                    f" seed={args.replay_seed}"
                    if args.replay_policy == "random"
                    else ""
                )
            )
            if args.budget_seconds > 0
            else ""
        )
    )
    timeline_end = float(full_rows[-1][3])
    last_seen, ever, t_end, meta = run_map_age(
        positions,
        names_used,
        resolution_m=args.resolution,
        subsample=args.subsample,
        margin_m=args.margin,
        hfov_deg=hfov,
        vfov_deg=vfov,
        projection=args.projection,
        heading_by_name=heading_map,
        timeline_end_sec=timeline_end,
    )
    if args.budget_seconds > 0:
        meta["replay_budget_seconds"] = float(args.budget_seconds)
        meta["replay_policy"] = args.replay_policy
        meta["replay_seed"] = int(args.replay_seed)
        meta["eligible_frames"] = len(full_rows)
        meta.update({f"replay_{k}": v for k, v in rinfo.items()})
        if args.replay_policy == "greedy_local":
            meta["greedy_trigger"] = args.greedy_trigger
            meta["greedy_unseen_scope"] = args.greedy_unseen_scope
            meta["greedy_radius_m"] = float(args.greedy_radius_m)
            meta["greedy_mean_age_threshold_s"] = float(args.greedy_mean_age_threshold_s)
            meta["greedy_stale_threshold_s"] = float(args.greedy_stale_threshold_s)
    print(
        f"Cells ever seen: {meta['frac_cells_ever_seen']*100:.2f}%  "
        f"ray updates: {meta['rays_accumulated']}"
    )
    save_outputs(
        last_seen,
        ever,
        t_end,
        meta,
        overlay_alpha=args.overlay_alpha,
        skip_floorplan_overlay=args.no_floorplan_overlay,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
