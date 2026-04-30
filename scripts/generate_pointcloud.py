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
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).parent.parent

FRAMES_DIR    = ROOT / "outputs" / "frames"
DEPTH_DIR     = ROOT / "outputs" / "depth_maps"
POSITIONS_JSON = ROOT / "outputs" / "frame_positions.json"
OUTPUT_PLY    = ROOT / "outputs" / "pointcloud.ply"

CAMERA_HEIGHT_M = 1.6   # assumed camera height above floor (metres)
SUBSAMPLE       = 6     # use every Nth pixel (6 → 640×320 per frame, ~205K pts)
VOXEL_SIZE_M    = 0.06  # voxel grid downsampling (metres)
MAX_DEPTH_M     = 18.0  # clip anything beyond this
FLOOR_ELEV_DEG  = -65   # elevation threshold for floor constraint (degrees)

# SVG waypoints in pixel space (parsed from data/svg_path.svg, viewBox 1400×819)
SVG_WAYPOINTS_PX = [
    (373.5, 475.5), (1126.5, 475.5), (1180.0, 503.5), (1193.0, 554.5),
    (1180.0, 604.0), (1133.0, 635.0), (1015.0, 635.0), (901.5, 635.0),
    (796.5, 635.0), (697.5, 626.0), (601.0, 602.5), (498.0, 563.5),
    (415.0, 518.5), (368.5, 483.5),
]
IMG_W_PX, IMG_H_PX = 1400, 819
REAL_W_M = 67.0
PX_PER_M = IMG_W_PX / REAL_W_M
REAL_H_M = IMG_H_PX / PX_PER_M

# Convert waypoints to world metres (flip Y: SVG top-left → world bottom-left)
SVG_WAYPOINTS_M = [
    (x / PX_PER_M, (IMG_H_PX - y) / PX_PER_M)
    for x, y in SVG_WAYPOINTS_PX
]


# ── Calibration helpers ────────────────────────────────────────────────────────

def floor_scale(depth_norm: np.ndarray) -> float:
    """
    Estimate metric scale factor from the bottom strip of the depth map.
    Pixels at elevation φ < FLOOR_ELEV_DEG see the ground at depth = CAMERA_HEIGHT_M / |sin(φ)|.
    """
    H, W = depth_norm.shape
    phi_thresh = FLOOR_ELEV_DEG * math.pi / 180.0          # negative radians
    # pixel row where elevation == phi_thresh
    v_thresh = int((0.5 - phi_thresh / math.pi) * H)
    v_thresh = max(0, min(v_thresh, H - 1))

    v_rows = np.arange(v_thresh, H)
    if len(v_rows) == 0:
        return MAX_DEPTH_M

    phi_rows = math.pi / 2 - v_rows / H * math.pi          # elevation per row
    expected = CAMERA_HEIGHT_M / np.abs(np.sin(phi_rows))   # expected floor depth (m)
    measured = depth_norm[v_rows, :].mean(axis=1)           # mean norm depth per row

    valid = measured > 0.02
    if not valid.any():
        return MAX_DEPTH_M

    scales = expected[valid] / measured[valid]
    return float(np.median(scales))


def svg_waypoint_scale(
    depth_norm: np.ndarray,
    cam_x: float, cam_y: float, cam_heading: float,
    W: int, H: int,
) -> float | None:
    """
    For the closest SVG waypoint (world XY on the floor), compute the expected
    depth from the camera, find the corresponding equirectangular pixel, read the
    depth value there, and return the implied scale factor.
    Returns None if the waypoint is behind the camera or too close.
    """
    best_scale = None
    best_dist = float("inf")

    for wx, wy in SVG_WAYPOINTS_M:
        dx = wx - cam_x
        dy = wy - cam_y
        dz = -CAMERA_HEIGHT_M            # waypoint is on the floor, camera is above
        real_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if real_dist < 0.5:              # skip waypoints right under the camera
            continue

        # Direction in world → rotate by -heading to get camera-local azimuth
        world_az = math.atan2(dy, dx)
        cam_az = world_az - cam_heading  # relative azimuth
        cam_el = math.atan2(dz, math.sqrt(dx**2 + dy**2))  # elevation (negative)

        # Map to equirectangular pixel
        u = ((cam_az % (2 * math.pi)) / (2 * math.pi)) * W
        v = (math.pi / 2 - cam_el) / math.pi * H
        u = int(np.clip(u, 0, W - 1))
        v = int(np.clip(v, 0, H - 1))

        d_norm = float(depth_norm[v, u])
        if d_norm < 0.02:
            continue

        implied_scale = real_dist / d_norm
        if implied_scale < MAX_DEPTH_M * 0.5 and real_dist < best_dist:
            best_dist = real_dist
            best_scale = implied_scale

    return best_scale


def combined_scale(depth_norm, cam_x, cam_y, cam_heading, W, H) -> float:
    s_floor = floor_scale(depth_norm)
    s_svg   = svg_waypoint_scale(depth_norm, cam_x, cam_y, cam_heading, W, H)

    if s_svg is None:
        return s_floor
    # Weight SVG more heavily (it's a hard geometric constraint)
    return 0.35 * s_floor + 0.65 * s_svg


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
