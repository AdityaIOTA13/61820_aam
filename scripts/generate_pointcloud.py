"""
generate_pointcloud.py

Generates a metric coloured point cloud from all 183 depth maps + colour frames.

Depth calibration — two complementary methods:
  1. Floor constraint: pixels pointing below -65° elevation see the ground at a
     predictable depth (CAMERA_HEIGHT_M / sin(elevation)). The ratio of expected
     to measured depth gives a per-frame metric scale factor.
  2. SVG waypoint constraint: for each frame, the nearest SVG waypoint is at a
     known world position. The direction from the camera to that waypoint maps to
     a specific equirectangular pixel; its depth value cross-checks the floor
     scale. A weighted average of both is used.

Camera positions come from outputs/frame_positions.json (derived from svg_path.svg
calibrated to the floorplan: 1400px = 67m). Camera heading is estimated from
consecutive frame positions.

Output: outputs/pointcloud.ply  (RGB, voxel-downsampled)

Run from repo root:  python scripts/generate_pointcloud.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

ROOT = Path(__file__).parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from courtyard_geom import (
    CAMERA_HEIGHT_M,
    MAX_DEPTH_M,
    combined_scale,
)

FRAMES_DIR = ROOT / "outputs" / "frames"
DEPTH_DIR = ROOT / "outputs" / "depth_maps"
POSITIONS_JSON = ROOT / "outputs" / "frame_positions.json"
OUTPUT_PLY = ROOT / "outputs" / "pointcloud.ply"

SUBSAMPLE = 6  # use every Nth pixel (6 → 640×320 per frame, ~205K pts)
VOXEL_SIZE_M = 0.06  # voxel grid downsampling (metres)


# ── Per-frame processing ───────────────────────────────────────────────────────

def process_frame(
    frame_path: Path,
    depth_path: Path,
    cam_x: float, cam_y: float,
    cam_heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Nx3 world points, Nx3 RGB colours 0-1)."""
    colour_img = np.array(Image.open(frame_path).convert("RGB"), dtype=np.float32)
    depth_img  = np.array(Image.open(depth_path).convert("L"),   dtype=np.float32)

    H, W = depth_img.shape
    depth_norm = depth_img / 255.0

    scale   = combined_scale(depth_norm, cam_x, cam_y, cam_heading, W, H)
    depth_m = depth_norm * scale
    depth_m = np.clip(depth_m, 0.1, MAX_DEPTH_M)

    # Subsampled grid
    v_idx = np.arange(0, H, SUBSAMPLE)
    u_idx = np.arange(0, W, SUBSAMPLE)
    vv, uu = np.meshgrid(v_idx, u_idx, indexing="ij")

    # Equirectangular → spherical
    theta = (uu / W) * 2 * math.pi        # azimuth 0→2π
    phi   = math.pi / 2 - (vv / H) * math.pi  # elevation π/2→-π/2

    d = depth_m[vv, uu]

    # Ray in camera-local frame (x=forward, y=left, z=up)
    dx_c = np.cos(phi) * np.cos(theta)
    dy_c = np.cos(phi) * np.sin(theta)
    dz_c = np.sin(phi)

    # Rotate by camera heading into world frame (x=east, y=north)
    cos_h, sin_h = math.cos(cam_heading), math.sin(cam_heading)
    dx_w = cos_h * dx_c - sin_h * dy_c
    dy_w = sin_h * dx_c + cos_h * dy_c
    dz_w = dz_c

    x_w = cam_x + d * dx_w
    y_w = cam_y + d * dy_w
    z_w = CAMERA_HEIGHT_M + d * dz_w

    # Keep plausible points (above ground, below ceiling, not at max clip)
    valid = (z_w > -0.3) & (z_w < 12.0) & (d < MAX_DEPTH_M * 0.92)
    pts  = np.stack([x_w[valid], y_w[valid], z_w[valid]], axis=1)
    cols = (colour_img[vv, uu][valid] / 255.0).astype(np.float64)

    return pts, cols


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    with open(POSITIONS_JSON) as f:
        positions = json.load(f)

    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}. Run extract_frames.py first.")

    # Build lookup: frame filename → (cam_x, cam_y, timestamp)
    pos_map = {
        k: (v["x_meters"], v["y_meters"], v["timestamp_sec"])
        for k, v in positions.items()
    }

    # Pre-compute camera headings from consecutive positions
    frames_with_pos = [(fp, pos_map[fp.name]) for fp in frame_paths if fp.name in pos_map]

    def heading(i):
        if i < len(frames_with_pos) - 1:
            x0, y0 = frames_with_pos[i][1][:2]
            x1, y1 = frames_with_pos[i + 1][1][:2]
            if abs(x1 - x0) > 1e-6 or abs(y1 - y0) > 1e-6:
                return math.atan2(y1 - y0, x1 - x0)
        return heading(max(0, i - 1)) if i > 0 else 0.0

    headings = [heading(i) for i in range(len(frames_with_pos))]

    all_pts  = []
    all_cols = []
    n = len(frames_with_pos)

    OUTPUT_PLY.parent.mkdir(parents=True, exist_ok=True)

    for i, ((fp, (cx, cy, _)), h) in enumerate(zip(frames_with_pos, headings), 1):
        stem  = fp.stem  # e.g. frame_00000_0.000s
        dp    = DEPTH_DIR / f"{stem}_depth.png"
        if not dp.exists():
            print(f"  [{i:>3}/{n}] missing depth map, skipping")
            continue

        pts, cols = process_frame(fp, dp, cx, cy, h)
        all_pts.append(pts)
        all_cols.append(cols)
        print(f"  [{i:>3}/{n}] +{len(pts):>7,} pts  cam=({cx:.1f},{cy:.1f})m", end="\r")

    print(f"\nAggregating {sum(len(p) for p in all_pts):,} raw points...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))

    print(f"Voxel downsampling at {VOXEL_SIZE_M}m ...")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)

    print(f"Statistical outlier removal ...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    n_pts = len(pcd.points)
    print(f"Final point cloud: {n_pts:,} points")

    o3d.io.write_point_cloud(str(OUTPUT_PLY), pcd, write_ascii=False, compressed=True)
    size_mb = OUTPUT_PLY.stat().st_size / 1e6
    print(f"Saved: {OUTPUT_PLY.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print(f"\nTo view:  python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{OUTPUT_PLY}')])\"")


if __name__ == "__main__":
    main()
