"""
project_coverage.py

Projects subsampled equirectangular depth rays onto the floorplane (z=0),
accumulates per-cell last observation time from frame_positions.json, and
writes map-age and coverage heatmaps (metres, same frame as x_meters/y_meters).

Requires outputs/frames/*.jpg, outputs/depth_maps/*_depth.png, and
outputs/frame_positions.json (see README / R2 CDN).

Run from repo root:
  python scripts/project_coverage.py
  python scripts/project_coverage.py --max-frames 20 --resolution 0.25
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    REAL_W_M,
    camera_heading_from_trajectory,
    ground_plane_hits,
    load_depth_norm,
    parse_svg_waypoints,
)

FRAMES_DIR = ROOT / "outputs" / "frames"
DEPTH_DIR = ROOT / "outputs" / "depth_maps"
POSITIONS_JSON = ROOT / "outputs" / "frame_positions.json"
OUT_DIR = ROOT / "outputs" / "coverage"

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


def run_map_age(
    positions: dict,
    frame_names: list[str],
    *,
    resolution_m: float = 0.2,
    subsample: int = 8,
    margin_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Returns (last_seen_sec, ever_seen_mask, t_end_sec, meta).

    last_seen_sec: (ny, nx) max timestamp observing each cell; -inf if never.
    ever_seen_mask: bool (ny, nx)
    """
    real_h_m = IMG_H_PX / PX_PER_M
    w_m = REAL_W_M + 2 * margin_m
    h_m = real_h_m + 2 * margin_m
    nx = max(1, int(math.ceil(w_m / resolution_m)))
    ny = max(1, int(math.ceil(h_m / resolution_m)))
    x0 = -margin_m
    y0 = -margin_m

    rows = []
    for name in frame_names:
        if name not in positions:
            continue
        p = positions[name]
        rows.append((name, float(p["x_meters"]), float(p["y_meters"]), float(p["timestamp_sec"])))

    if not rows:
        raise ValueError("No frames matched frame_positions.json")

    rows.sort(key=lambda r: r[3])
    xy_list = [(r[1], r[2]) for r in rows]
    headings = camera_heading_from_trajectory(xy_list)

    last_seen = np.full((ny, nx), -np.inf, dtype=np.float64)
    t_end = rows[-1][3]

    meta_hits = 0
    frames_depth_ok = 0
    for i, ((name, cx, cy, ts), h) in enumerate(zip(rows, headings)):
        dp = depth_path_for_frame(name)
        fp = FRAMES_DIR / name
        if not dp.exists() or not fp.exists():
            continue
        frames_depth_ok += 1
        depth_norm = load_depth_norm(dp)
        gx, gy = ground_plane_hits(depth_norm, cx, cy, h, subsample=subsample)
        if gx.size == 0:
            continue
        ix = ((gx - x0) / resolution_m).astype(np.int64)
        iy = ((gy - y0) / resolution_m).astype(np.int64)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy = ix[valid], iy[valid]
        if ix.size == 0:
            continue
        ts_arr = np.full(ix.shape, ts, dtype=np.float64)
        np.maximum.at(last_seen, (iy, ix), ts_arr)
        meta_hits += int(ix.size)

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
    }
    return last_seen, ever.astype(np.bool_), t_end, meta


def save_outputs(
    last_seen: np.ndarray,
    ever: np.ndarray,
    t_end: float,
    meta: dict,
    *,
    overlay_alpha: float = 0.52,
    skip_floorplan_overlay: bool = False,
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    ax.set_title("Map age at end of walk (s since last ground observation)")
    plt.colorbar(im, ax=ax, label="seconds")
    p_age = OUT_DIR / "map_age_end.png"
    fig.tight_layout()
    fig.savefig(p_age, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    cov = ever.astype(np.float32)
    ax2.imshow(cov, origin="lower", extent=extent, aspect="equal", cmap="viridis", vmin=0, vmax=1)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Coverage (1 = cell observed on ground plane at least once)")
    p_cov = OUT_DIR / "coverage_ever.png"
    fig2.tight_layout()
    fig2.savefig(p_cov, dpi=150)
    plt.close(fig2)

    meta_path = OUT_DIR / "coverage_meta.json"
    np.save(OUT_DIR / "last_seen_sec.npy", last_seen)
    print(f"Wrote {p_age.relative_to(ROOT)}")
    print(f"Wrote {p_cov.relative_to(ROOT)}")
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}/last_seen_sec.npy")

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
    p_overlay = OUT_DIR / "map_age_on_floorplan.png"
    Image.fromarray(np.clip(out_age, 0, 255).astype(np.uint8)).save(p_overlay, quality=95)
    print(f"Wrote {p_overlay.relative_to(ROOT)}  (floorplan: {fp_src})")

    # --- Coverage on floorplan ---
    cmap_c = mpl_mod.colormaps["magma"]
    cov_f = ever_px.astype(np.float64)
    rgba_c = cmap_c(cov_f)
    rgb_c = (rgba_c[:, :, :3] * 255.0).astype(np.float32)
    out_cov = base.copy()
    out_cov[ever_px] = base[ever_px] * (1.0 - a) + rgb_c[ever_px] * a
    p_cov_fp = OUT_DIR / "coverage_on_floorplan.png"
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
    args = ap.parse_args()

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

    missing_depth = sum(1 for n in names if not depth_path_for_frame(n).exists())
    if missing_depth == len(names):
        sys.exit(
            f"No depth maps in {DEPTH_DIR.relative_to(ROOT)}/ for selected frames. "
            "Run estimate_depth.py or download depth_maps from R2."
        )
    if missing_depth:
        print(f"Note: {missing_depth} frames skipped (missing depth PNG).")

    print(
        f"Grid res={args.resolution}m subsample={args.subsample} frames={len(names)} "
        f"(of {len(frame_paths)} on disk)"
    )
    last_seen, ever, t_end, meta = run_map_age(
        positions,
        names,
        resolution_m=args.resolution,
        subsample=args.subsample,
        margin_m=args.margin,
    )
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
    )


if __name__ == "__main__":
    main()
