"""
prepare_colmap.py

Selects frames from the 360° courtyard walk using a greedy staleness policy
(matching the budget constraint used in the coverage simulation), then crops
a forward-facing perspective patch from each equirectangular frame.

The cropped images + PINHOLE camera file are written into outputs/colmap_input/
in a layout that COLMAP's feature_extractor can read directly.

Two modes
---------
--all        Use all 183 frames (always-on baseline for 3D recon)
--budget N   Greedy-select N frames whose local map age exceeds a threshold
             (default N = 36, ≈18s budget at 2fps over a 91s walk → ~20% frames)

Perspective crop geometry
-------------------------
HFOV = 100° centred on the camera heading from frame_positions.json.
Equirectangular → pinhole using per-pixel ray remapping (numpy/cv2).
Output resolution: 1280 × 720 (16:9, suitable for COLMAP).

COLMAP commands (printed at the end)
-------------------------------------
  colmap feature_extractor --database_path outputs/colmap_input/database.db \\
      --image_path outputs/colmap_input/images/ \\
      --ImageReader.camera_model PINHOLE \\
      --ImageReader.single_camera 1

  colmap exhaustive_matcher --database_path outputs/colmap_input/database.db

  colmap mapper --database_path outputs/colmap_input/database.db \\
      --image_path outputs/colmap_input/images/ \\
      --output_path outputs/colmap_input/sparse/

Run from repo root:  python scripts/prepare_colmap.py [--all | --budget N]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent

FRAMES_DIR      = ROOT / "outputs" / "frames"
COVERAGE_NPY    = ROOT / "outputs" / "coverage" / "last_seen_sec.npy"
POSITIONS_JSON  = ROOT / "outputs" / "frame_positions.json"
OUT_DIR         = ROOT / "outputs" / "colmap_input"

# Crop parameters
HFOV_DEG   = 100.0
OUT_W, OUT_H = 1280, 720
# Equirectangular source
SRC_W, SRC_H = 3840, 1920
# Coverage grid origin (from coverage_meta.json)
GRID_ORIGIN_X = -1.0   # metres
GRID_ORIGIN_Y = -1.0
GRID_RES_M    = 0.2

# Computed focal length for the perspective crop
focal_px = OUT_W / (2.0 * math.tan(math.radians(HFOV_DEG / 2.0)))
VFOV_DEG = math.degrees(2 * math.atan(OUT_H / (2 * focal_px)))


# ── Equirectangular → perspective remap ───────────────────────────────────────

def build_remap(heading_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cv2 remap maps for a perspective crop centred on heading_rad.
    Returns (map_x, map_y) in source (equirectangular) pixel coordinates.
    """
    # Output pixel grid
    u_out = np.arange(OUT_W, dtype=np.float64)
    v_out = np.arange(OUT_H, dtype=np.float64)
    uu, vv = np.meshgrid(u_out, v_out)  # shape (OUT_H, OUT_W)

    # Camera-frame ray (x=right, y=up, z=forward)
    x_c = (uu - OUT_W / 2.0) / focal_px
    y_c = -(vv - OUT_H / 2.0) / focal_px
    z_c = np.ones_like(x_c)

    # Rotate ray from camera frame into world frame by heading_rad
    # Forward = (cos h, sin h, 0), Right = (sin h, -cos h, 0), Up = (0,0,1)
    cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)

    dx = x_c * sin_h  + z_c * cos_h   # world X (east)
    dy = x_c * (-cos_h) + z_c * sin_h # world Y (north)
    dz = y_c                            # world Z (up)

    # Equirectangular angles
    theta = np.arctan2(dy, dx)                              # azimuth [-π, π]
    phi   = np.arctan2(dz, np.sqrt(dx**2 + dy**2))         # elevation

    # Map to source pixel coordinates
    map_x = ((theta + math.pi) / (2 * math.pi) * SRC_W).astype(np.float32)
    map_y = ((math.pi / 2 - phi) / math.pi * SRC_H).astype(np.float32)

    return map_x, map_y


def crop_frame(src_path: Path, heading_rad: float) -> np.ndarray:
    """Load equirectangular JPEG and return perspective crop as (H,W,3) BGR array."""
    img = cv2.imread(str(src_path))
    # Resize source to expected resolution if needed
    if img.shape[1] != SRC_W or img.shape[0] != SRC_H:
        img = cv2.resize(img, (SRC_W, SRC_H), interpolation=cv2.INTER_LINEAR)
    mx, my = build_remap(heading_rad)
    return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


# ── Frame selection ────────────────────────────────────────────────────────────

def heading_from_positions(positions: dict, ordered: list[str], idx: int) -> float:
    """Estimate camera heading from consecutive frame positions."""
    for di in [1, -1]:
        j = idx + di
        if 0 <= j < len(ordered):
            xi, yi = positions[ordered[idx]]["x_meters"], positions[ordered[idx]]["y_meters"]
            xj, yj = positions[ordered[j]]["x_meters"],  positions[ordered[j]]["y_meters"]
            if abs(xj - xi) > 1e-4 or abs(yj - yi) > 1e-4:
                return math.atan2(yj - yi, xj - xi)
    return 0.0


def grid_cell(x_m: float, y_m: float) -> tuple[int, int]:
    cx = int((x_m - GRID_ORIGIN_X) / GRID_RES_M)
    cy = int((y_m - GRID_ORIGIN_Y) / GRID_RES_M)
    return cx, cy


def select_greedy_budget(positions: dict, ordered: list[str], n_budget: int,
                         last_seen: np.ndarray | None) -> list[str]:
    """
    Greedy frame selection: pick frames where local map age is highest,
    respecting a total budget of n_budget frames.
    Falls back to evenly-spaced if last_seen grid is unavailable.
    """
    if last_seen is None:
        step = max(1, len(ordered) // n_budget)
        return ordered[::step][:n_budget]

    scores = []
    for name in ordered:
        p = positions[name]
        cx, cy = grid_cell(p["x_meters"], p["y_meters"])
        # Local 3×3 patch mean age (NaN = never seen → treat as very stale)
        ny, nx = last_seen.shape
        x0, x1 = max(0, cx - 1), min(nx, cx + 2)
        y0, y1 = max(0, cy - 1), min(ny, cy + 2)
        patch = last_seen[y0:y1, x0:x1]
        age = np.nanmean(np.where(np.isfinite(patch), patch, 1e6))
        scores.append((age, name))

    scores.sort(key=lambda x: -x[0])  # highest age first
    selected = set(s[1] for s in scores[:n_budget])
    # Return in temporal order
    return [n for n in ordered if n in selected]


# ── COLMAP cameras.txt ────────────────────────────────────────────────────────

def write_cameras_txt(out_dir: Path, n_images: int) -> None:
    """Write a PINHOLE cameras.txt for COLMAP (single shared camera)."""
    cx, cy = OUT_W / 2.0, OUT_H / 2.0
    with open(out_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {OUT_W} {OUT_H} {focal_px:.4f} {focal_px:.4f} {cx:.1f} {cy:.1f}\n")
    print(f"  cameras.txt  (PINHOLE f={focal_px:.1f}px, HFOV={HFOV_DEG}°, VFOV={VFOV_DEG:.1f}°)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all",    action="store_true", help="Use all available frames")
    mode.add_argument("--budget", type=int, default=36,
                      help="Number of frames to select via greedy staleness (default: 36)")
    args = ap.parse_args()

    with open(POSITIONS_JSON) as f:
        positions = json.load(f)

    ordered = sorted(positions.keys(), key=lambda n: positions[n]["timestamp_sec"])
    print(f"Total frames available: {len(ordered)}")

    # Load map-age grid for greedy selection
    last_seen: np.ndarray | None = None
    if COVERAGE_NPY.exists():
        last_seen = np.load(str(COVERAGE_NPY))
        print(f"Loaded coverage grid: {last_seen.shape}")
    else:
        print("Warning: coverage grid not found, falling back to uniform sampling")

    if args.all:
        selected = ordered
        tag = "all"
    else:
        selected = select_greedy_budget(positions, ordered, args.budget, last_seen)
        tag = f"budget{args.budget}"

    print(f"Selected {len(selected)} frames ({tag} mode)")

    img_dir = OUT_DIR / tag / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    write_cameras_txt(OUT_DIR / tag, len(selected))

    manifest = []
    for i, name in enumerate(selected, 1):
        src = FRAMES_DIR / name
        if not src.exists():
            print(f"  [{i:>3}/{len(selected)}] MISSING {name}, skipping")
            continue

        p = positions[name]
        h_idx = ordered.index(name)
        heading = heading_from_positions(positions, ordered, h_idx)

        crop = crop_frame(src, heading)
        out_path = img_dir / name
        cv2.imwrite(str(out_path), crop)

        manifest.append({
            "filename": name,
            "timestamp_sec": p["timestamp_sec"],
            "x_meters": p["x_meters"],
            "y_meters": p["y_meters"],
            "heading_rad": heading,
            "lat": p.get("lat"),
            "lon": p.get("lon"),
        })
        print(f"  [{i:>3}/{len(selected)}] {name}  heading={math.degrees(heading):.1f}°", end="\r")

    with open(OUT_DIR / tag / "frame_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n\nOutput: {(OUT_DIR / tag).relative_to(ROOT)}/")
    print(f"  {len(manifest)} perspective crops ({OUT_W}×{OUT_H}, HFOV={HFOV_DEG}°)")
    print(f"  cameras.txt, frame_manifest.json")
    print(f"\nCOLMAP commands:")
    rel = (OUT_DIR / tag).relative_to(ROOT)
    print(f"""
  colmap feature_extractor \\
      --database_path {rel}/database.db \\
      --image_path {rel}/images/ \\
      --ImageReader.camera_model PINHOLE \\
      --ImageReader.single_camera 1

  colmap exhaustive_matcher \\
      --database_path {rel}/database.db

  mkdir -p {rel}/sparse
  colmap mapper \\
      --database_path {rel}/database.db \\
      --image_path {rel}/images/ \\
      --output_path {rel}/sparse/

  # View result:
  colmap gui --import_path {rel}/sparse/0/ --database_path {rel}/database.db \\
             --image_path {rel}/images/
""")


if __name__ == "__main__":
    main()
