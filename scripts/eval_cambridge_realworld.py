#!/usr/bin/env python3
"""
eval_cambridge_realworld.py  (v2 — fixed coords + timestamps)
==============================================================
Evaluate scanning policies on Aditya's real Cambridge Commons walk
(data_02_outputs/) and emit a JSON compatible with plot_coverage_eval.py.

Two bugs fixed vs v1
--------------------
1. timestamp_sec is 0 for every frame in the JSON (process_cambridge.py
   couldn't read video creation time via ffprobe).  We reconstruct a
   monotonic walk timeline from clip_id order + clip_offset_sec.

2. x_meters/y_meters are GPS-local (~-95 to -104 m) so they land entirely
   outside the courtyard floor_grid_layout() which starts at x0=-1 m.
   We shift all positions so the walk centroid = (0, 0) and build a
   Cambridge-specific grid that covers the actual coordinate range.

Usage
-----
  python scripts/eval_cambridge_realworld.py
  python scripts/eval_cambridge_realworld.py --n-random-seeds 5
  python scripts/eval_cambridge_realworld.py --subsample 16   # faster
  python scripts/eval_cambridge_realworld.py --skip-oracle
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Repo paths ─────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

DATA_OUT       = ROOT / "data_02_outputs"
POSITIONS_JSON = DATA_OUT / "frame_positions.json"
FRAMES_DIR     = DATA_OUT / "frames"
DEPTH_DIR      = DATA_OUT / "depth_maps"
DEFAULT_OUT    = ROOT / "outputs" / "cambridge_coverage_eval.json"

# ── Import project_coverage and redirect its path globals ──────────────────
import project_coverage as pc
pc.FRAMES_DIR     = FRAMES_DIR
pc.DEPTH_DIR      = DEPTH_DIR
pc.POSITIONS_JSON = POSITIONS_JSON

from project_coverage import (
    camera_heading_from_trajectory,
    load_depth_norm,
    RAYBAN_META_DEFAULT_HFOV_DEG,
    RAYBAN_META_DEFAULT_VFOV_DEG,
)

# ── Cambridge-specific grid ─────────────────────────────────────────────────

def cambridge_grid_layout(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    resolution_m: float,
    margin_m: float,
) -> tuple[float, float, int, int]:
    """Grid covering the actual Cambridge walk bbox (centred coords)."""
    x0 = float(x_coords.min()) - margin_m
    y0 = float(y_coords.min()) - margin_m
    x1 = float(x_coords.max()) + margin_m
    y1 = float(y_coords.max()) + margin_m
    nx = max(1, int(math.ceil((x1 - x0) / resolution_m)))
    ny = max(1, int(math.ceil((y1 - y0) / resolution_m)))
    return x0, y0, nx, ny


# ── Data loading & normalisation ────────────────────────────────────────────

class FrameRecord(NamedTuple):
    name: str
    x_m: float        # centred coordinates
    y_m: float
    t_sec: float      # monotonic walk time from walk start
    heading_deg: float


def load_cambridge_frames(pos_path: Path) -> list[FrameRecord]:
    """
    Load frame_positions.json, fix timestamps, centre coordinates.

    Timestamp fix: timestamp_sec=0 for all frames because ffprobe couldn't
    read mp4 creation time.  Reconstruct from clip_id sort order +
    clip_offset_sec with a 2-second inter-clip gap.

    Coordinate fix: GPS-local x/y have large offsets; subtract centroid.
    """
    with pos_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    frames_raw: dict = raw["frames"] if "frames" in raw else raw

    # Group by clip, sort within each clip by offset
    clips: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, p in frames_raw.items():
        clips[p["clip_id"]].append((name, p))
    for cid in clips:
        clips[cid].sort(key=lambda x: float(x[1]["clip_offset_sec"]))

    INTER_CLIP_GAP_S = 2.0
    records: list[FrameRecord] = []
    prev_clip_end = 0.0

    for cid in sorted(clips.keys()):
        frames_in_clip = clips[cid]
        offsets = [float(p["clip_offset_sec"]) for _, p in frames_in_clip]
        t_clip_start = prev_clip_end + INTER_CLIP_GAP_S if records else 0.0

        for (name, p), off in zip(frames_in_clip, offsets):
            records.append(FrameRecord(
                name=name,
                x_m=float(p["x_meters"]),
                y_m=float(p["y_meters"]),
                t_sec=t_clip_start + off,
                heading_deg=float(p.get("heading_deg", 0.0)),
            ))

        prev_clip_end = t_clip_start + (max(offsets) if offsets else 0.0)

    # Centre coordinates on walk centroid
    xs = np.array([r.x_m for r in records])
    ys = np.array([r.y_m for r in records])
    cx, cy = float(xs.mean()), float(ys.mean())
    records = [r._replace(x_m=r.x_m - cx, y_m=r.y_m - cy) for r in records]

    print(f"  Coord centroid removed: ({cx:.2f}, {cy:.2f}) m")
    xs_c = xs - cx
    ys_c = ys - cy
    print(f"  Centred walk bbox: x=[{xs_c.min():.1f}, {xs_c.max():.1f}]  "
          f"y=[{ys_c.min():.1f}, {ys_c.max():.1f}] m")
    return records


def filter_eligible(records: list[FrameRecord]) -> list[FrameRecord]:
    eligible = []
    for r in records:
        jpg = FRAMES_DIR / r.name
        dep = DEPTH_DIR / f"{Path(r.name).stem}_depth.png"
        if jpg.exists() and dep.exists():
            eligible.append(r)
    return eligible


# ── Depth accumulation ──────────────────────────────────────────────────────

def accumulate_one(
    last_seen: np.ndarray,
    r: FrameRecord,
    heading_rad: float,
    *,
    x0: float, y0: float, nx: int, ny: int,
    resolution_m: float, subsample: int,
    hfov_deg: float, vfov_deg: float, projection: str,
) -> int:
    dep = DEPTH_DIR / f"{Path(r.name).stem}_depth.png"
    jpg = FRAMES_DIR / r.name
    if not dep.exists() or not jpg.exists():
        return 0

    depth_norm = load_depth_norm(dep)

    if projection == "ground_plane":
        from project_coverage import ground_plane_hits
        gx, gy = ground_plane_hits(
            depth_norm, r.x_m, r.y_m, heading_rad,
            subsample=subsample, hfov_deg=hfov_deg, vfov_deg=vfov_deg,
        )
    else:
        from project_coverage import backproject_xy_hits
        gx, gy = backproject_xy_hits(
            depth_norm, r.x_m, r.y_m, heading_rad,
            subsample=subsample, hfov_deg=hfov_deg, vfov_deg=vfov_deg,
        )

    if gx.size == 0:
        return 0

    ix = ((gx - x0) / resolution_m).astype(np.int64)
    iy = ((gy - y0) / resolution_m).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[valid], iy[valid]
    if ix.size == 0:
        return 0

    np.maximum.at(last_seen, (iy, ix), np.full(ix.shape, r.t_sec))
    return int(ix.size)


# ── Metrics ─────────────────────────────────────────────────────────────────

def largest_connected_component_size(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    ny, nx = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
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
                    if 0 <= y2 < ny and 0 <= x2 < nx and mask[y2, x2] and not visited[y2, x2]:
                        visited[y2, x2] = True
                        stack.append((y2, x2))
            best = max(best, size)
    return best


class EpisodeResult(NamedTuple):
    coverage_vs_theoretic_max: float
    mean_age_scanned_sec: float
    lcc_vs_theoretic_max: float
    effective_camera_on_sec: float
    n_kept: int
    n_reachable: int


def measure(
    kept: list[FrameRecord],
    headings_rad: dict[str, float],
    reachable: int,
    mean_dt: float,
    **grid_kw,
) -> EpisodeResult:
    if not kept:
        return EpisodeResult(0.0, float("nan"), 0.0, 0.0, 0, reachable)

    ny, nx = grid_kw["ny"], grid_kw["nx"]
    last_seen = np.full((ny, nx), -np.inf, dtype=np.float64)
    for r in kept:
        accumulate_one(last_seen, r, headings_rad[r.name], **grid_kw)

    ever = np.isfinite(last_seen) & (last_seen > -np.inf)
    n_scanned = int(np.sum(ever))
    lcc = largest_connected_component_size(ever)
    t_end = max(r.t_sec for r in kept)
    ages = t_end - last_seen[ever].astype(np.float64)
    mean_age = float(np.mean(ages)) if ages.size > 0 else float("nan")

    return EpisodeResult(
        coverage_vs_theoretic_max=float(n_scanned) / max(reachable, 1),
        mean_age_scanned_sec=mean_age,
        lcc_vs_theoretic_max=float(lcc) / max(reachable, 1),
        effective_camera_on_sec=len(kept) * mean_dt,
        n_kept=len(kept),
        n_reachable=reachable,
    )


# ── Policies ────────────────────────────────────────────────────────────────

def policy_random(rows: list[FrameRecord], k: int, seed: int) -> list[FrameRecord]:
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rows), size=min(k, len(rows)), replace=False).tolist())
    return [rows[i] for i in idx]


def _foot_cell(r: FrameRecord, x0: float, y0: float, nx: int, ny: int, res: float):
    ix = int(math.floor((r.x_m - x0) / res))
    iy = int(math.floor((r.y_m - y0) / res))
    return ix, iy


def policy_greedy_unseen(rows, k, headings_rad, **grid_kw) -> list[FrameRecord]:
    ny, nx = grid_kw["ny"], grid_kw["nx"]
    x0, y0, res = grid_kw["x0"], grid_kw["y0"], grid_kw["resolution_m"]
    last_seen = np.full((ny, nx), -np.inf, dtype=np.float64)
    selected = []
    for r in rows:
        if len(selected) >= k:
            break
        ix, iy = _foot_cell(r, x0, y0, nx, ny, res)
        unseen = (ix < 0 or ix >= nx or iy < 0 or iy >= ny
                  or bool(np.isneginf(last_seen[iy, ix])))
        if unseen:
            accumulate_one(last_seen, r, headings_rad[r.name], **grid_kw)
            selected.append(r)
    return selected


def policy_greedy_unseen_progressive(
    rows, k, headings_rad,
    ramp_start_frac=0.65, ramp_end_frac=1.0,
    rescan_max_stale_s=120.0, rescan_min_stale_s=15.0,
    **grid_kw,
) -> list[FrameRecord]:
    if not rows:
        return []
    t_lo, t_hi = rows[0].t_sec, rows[-1].t_sec
    walk_span = max(t_hi - t_lo, 1.0)
    ramp_s = t_lo + ramp_start_frac * walk_span
    ramp_e = t_lo + ramp_end_frac   * walk_span

    ny, nx = grid_kw["ny"], grid_kw["nx"]
    x0, y0, res = grid_kw["x0"], grid_kw["y0"], grid_kw["resolution_m"]
    last_seen = np.full((ny, nx), -np.inf, dtype=np.float64)
    selected = []

    for r in rows:
        if len(selected) >= k:
            break
        ix, iy = _foot_cell(r, x0, y0, nx, ny, res)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            foot_unseen, foot_ls = True, -np.inf
        else:
            foot_ls = float(last_seen[iy, ix])
            foot_unseen = np.isneginf(foot_ls)

        want = foot_unseen
        if not want and r.t_sec >= ramp_s:
            span = max(ramp_e - ramp_s, 1e-9)
            u = min(1.0, (r.t_sec - ramp_s) / span)
            req = rescan_max_stale_s * (1 - u) + rescan_min_stale_s * u
            want = math.isfinite(foot_ls) and (r.t_sec - foot_ls) >= req

        if want:
            accumulate_one(last_seen, r, headings_rad[r.name], **grid_kw)
            selected.append(r)
    return selected


def policy_oracle(rows, k, headings_rad, **grid_kw) -> list[FrameRecord]:
    ny, nx = grid_kw["ny"], grid_kw["nx"]
    print(f"  [oracle] pre-computing {len(rows)} hit sets ...", flush=True)
    frame_hits = []
    for i, r in enumerate(rows):
        tmp = np.full((ny, nx), -np.inf, dtype=np.float64)
        accumulate_one(tmp, r, headings_rad[r.name], **grid_kw)
        iys, ixs = np.where(np.isfinite(tmp) & (tmp > -np.inf))
        frame_hits.append((r, set(zip(iys.tolist(), ixs.tolist()))))
        if i % 200 == 0:
            print(f"  [oracle]  {i}/{len(rows)}", end="\r", flush=True)
    print()

    covered: set = set()
    selected = []
    remaining = list(range(len(frame_hits)))
    for step in range(k):
        if not remaining:
            break
        bi = max(remaining, key=lambda i: len(frame_hits[i][1] - covered))
        r_b, cells_b = frame_hits[bi]
        new = cells_b - covered
        if not new:
            break
        covered |= new
        selected.append(r_b)
        remaining.remove(bi)
        if step % 20 == 0:
            print(f"  [oracle] step {step+1}/{k}  covered={len(covered)}", flush=True)
    return selected


def compute_reachable(rows, headings_rad, **grid_kw) -> int:
    ny, nx = grid_kw["ny"], grid_kw["nx"]
    last_seen = np.full((ny, nx), -np.inf, dtype=np.float64)
    for i, r in enumerate(rows):
        accumulate_one(last_seen, r, headings_rad[r.name], **grid_kw)
        if i % 200 == 0:
            print(f"  reachable: {i}/{len(rows)}", end="\r", flush=True)
    print()
    return int(np.sum(np.isfinite(last_seen) & (last_seen > -np.inf)))


# ── Summary ──────────────────────────────────────────────────────────────────

def summarize(results: list[EpisodeResult]) -> dict:
    if not results:
        return {}
    keys = ["coverage_vs_theoretic_max", "mean_age_scanned_sec",
            "lcc_vs_theoretic_max", "effective_camera_on_sec"]
    out: dict = {}
    for k in keys:
        vals = np.array([getattr(r, k) for r in results], dtype=np.float64)
        if k == "mean_age_scanned_sec":
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                out[f"{k}_mean"] = float("nan")
                out[f"{k}_std"]  = float("nan")
                continue
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"]  = float(np.std(vals))
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions",      default=str(POSITIONS_JSON))
    ap.add_argument("--frames-dir",     default=str(FRAMES_DIR))
    ap.add_argument("--depth-dir",      default=str(DEPTH_DIR))
    ap.add_argument("--out",            default=str(DEFAULT_OUT))
    ap.add_argument("--resolution",     type=float, default=0.2)
    ap.add_argument("--subsample",      type=int,   default=8)
    ap.add_argument("--margin",         type=float, default=5.0)
    ap.add_argument("--hfov-deg",       type=float, default=RAYBAN_META_DEFAULT_HFOV_DEG)
    ap.add_argument("--vfov-deg",       type=float, default=RAYBAN_META_DEFAULT_VFOV_DEG)
    ap.add_argument("--fov-full-360",   action="store_true")
    ap.add_argument("--projection",     choices=("backproject", "ground_plane"),
                    default="backproject")
    ap.add_argument("--budget-seconds", type=float, default=0.0)
    ap.add_argument("--n-random-seeds", type=int,   default=10)
    ap.add_argument("--skip-oracle",    action="store_true")
    args = ap.parse_args()

    pc.FRAMES_DIR     = Path(args.frames_dir)
    pc.DEPTH_DIR      = Path(args.depth_dir)
    pc.POSITIONS_JSON = Path(args.positions)

    hfov = 400.0 if args.fov_full_360 else float(args.hfov_deg)
    vfov = 200.0 if args.fov_full_360 else float(args.vfov_deg)

    print(f"Loading {args.positions} ...")
    all_records = load_cambridge_frames(Path(args.positions))
    print(f"  {len(all_records)} frames total")

    rows = filter_eligible(all_records)
    print(f"  {len(rows)} eligible (JPEG + depth on disk)")
    if not rows:
        sys.exit(
            f"No eligible frames.\n"
            f"  JPEGs expected in: {pc.FRAMES_DIR}\n"
            f"  Depth PNGs in:     {pc.DEPTH_DIR}  (<stem>_depth.png)"
        )

    times = [r.t_sec for r in rows]
    t_lo, t_hi = min(times), max(times)
    walk_span = t_hi - t_lo
    mean_dt = walk_span / max(len(rows) - 1, 1)
    print(f"  Walk span: {walk_span:.1f} s ({walk_span/60:.1f} min)  "
          f"mean Δt: {mean_dt:.3f} s")

    # Headings: use GPS course when available, else derive from trajectory
    xy = [(r.x_m, r.y_m) for r in rows]
    traj_h = camera_heading_from_trajectory(xy)
    headings_rad: dict[str, float] = {
        r.name: (math.radians(r.heading_deg) if r.heading_deg != 0.0 else float(th))
        for r, th in zip(rows, traj_h)
    }

    xs = np.array([r.x_m for r in rows])
    ys = np.array([r.y_m for r in rows])
    x0, y0, nx, ny = cambridge_grid_layout(xs, ys, args.resolution, args.margin)
    print(f"  Grid: {nx}×{ny}  origin=({x0:.1f},{y0:.1f}) m  "
          f"size={nx*args.resolution:.0f}×{ny*args.resolution:.0f} m")

    grid_kw = dict(
        x0=x0, y0=y0, nx=nx, ny=ny,
        resolution_m=args.resolution,
        subsample=args.subsample,
        hfov_deg=hfov, vfov_deg=vfov,
        projection=args.projection,
    )

    print("\nComputing reachable cells ...")
    reachable = compute_reachable(rows, headings_rad, **grid_kw)
    print(f"  Reachable: {reachable} cells")
    if reachable == 0:
        sys.exit("Zero reachable cells — check depth maps and coordinate alignment.")

    print("\nGreedy unseen (sets budget K) ...")
    gu_kept = policy_greedy_unseen(rows, len(rows), headings_rad, **grid_kw)
    k_budget = (
        max(1, int(math.floor(args.budget_seconds / mean_dt)))
        if args.budget_seconds > 0 else len(gu_kept)
    )
    print(f"  greedy_unseen kept {len(gu_kept)} → k_budget={k_budget}")

    all_results: dict[str, list[EpisodeResult]] = {
        "random": [], "greedy_unseen": [],
        "greedy_unseen_progressive": [], "oracle": [],
    }

    print(f"\nRandom ({args.n_random_seeds} seeds) ...")
    for seed in range(args.n_random_seeds):
        kept = policy_random(rows, k_budget, seed)
        r = measure(kept, headings_rad, reachable, mean_dt, **grid_kw)
        all_results["random"].append(r)
        print(f"  seed {seed:2d}: cov={r.coverage_vs_theoretic_max:.3f} "
              f"age={r.mean_age_scanned_sec:.0f}s kept={r.n_kept}", flush=True)

    print("\nGreedy unseen metrics ...")
    r_gu = measure(gu_kept, headings_rad, reachable, mean_dt, **grid_kw)
    all_results["greedy_unseen"].append(r_gu)
    print(f"  cov={r_gu.coverage_vs_theoretic_max:.3f} "
          f"age={r_gu.mean_age_scanned_sec:.0f}s kept={r_gu.n_kept}")

    print("\nGreedy unseen + progressive rescan ...")
    gup = policy_greedy_unseen_progressive(rows, k_budget, headings_rad, **grid_kw)
    r_gup = measure(gup, headings_rad, reachable, mean_dt, **grid_kw)
    all_results["greedy_unseen_progressive"].append(r_gup)
    print(f"  cov={r_gup.coverage_vs_theoretic_max:.3f} "
          f"age={r_gup.mean_age_scanned_sec:.0f}s kept={r_gup.n_kept}")

    if not args.skip_oracle:
        print("\nOracle ...")
        ok = policy_oracle(rows, k_budget, headings_rad, **grid_kw)
        r_ok = measure(ok, headings_rad, reachable, mean_dt, **grid_kw)
        all_results["oracle"].append(r_ok)
        print(f"  cov={r_ok.coverage_vs_theoretic_max:.3f} "
              f"age={r_ok.mean_age_scanned_sec:.0f}s kept={r_ok.n_kept}")
    else:
        print("\n[oracle skipped]")

    summary = {m: summarize(v) if v else None for m, v in all_results.items()}
    summary["trained"] = None

    payload = {
        "n_scenarios": args.n_random_seeds,
        "seed0": 0,
        "checkpoint": None,
        "dt_s": float(mean_dt),
        "oracle_per_day_budget": True,
        "oracle_mode": "offline_set_cover",
        "eval_config_note": (
            f"Cambridge Commons real walk. resolution={args.resolution}m "
            f"subsample={args.subsample} hfov={hfov:.0f} vfov={vfov:.0f} "
            f"projection={args.projection} k_budget={k_budget} "
            f"reachable_cells={reachable} walk_span_s={walk_span:.1f}"
        ),
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {out_path}")
    print(f"Plot:  python scripts/plot_coverage_eval.py -i {out_path} --layout full")

    print("\n── Summary ──────────────────────────────────────────")
    for m, s in summary.items():
        if not s:
            continue
        print(f"  {m:<30} "
              f"cov={s.get('coverage_vs_theoretic_max_mean', float('nan')):.3f} "
              f"age={s.get('mean_age_scanned_sec_mean', float('nan'))/60:.1f}min "
              f"lcc={s.get('lcc_vs_theoretic_max_mean', float('nan')):.3f} "
              f"cam={s.get('effective_camera_on_sec_mean', float('nan')):.0f}s")


if __name__ == "__main__":
    main()