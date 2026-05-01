"""
Shared equirectangular depth -> world (metres) geometry for courtyard_360.

Aligned with generate_pointcloud.py: same scale, heading, and ray model.
Used by project_coverage.py for floorplane hits.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# Floorplan / SVG (viewBox 1400 x 819)
IMG_W_PX, IMG_H_PX = 1400, 819
REAL_W_M = 67.0
PX_PER_M = IMG_W_PX / REAL_W_M

CAMERA_HEIGHT_M = 1.6
MAX_DEPTH_M = 18.0
FLOOR_ELEV_DEG = -65

# SVG waypoints in pixel space (same order as data/svg_path.svg)
SVG_WAYPOINTS_PX = [
    (373.5, 475.5),
    (1126.5, 475.5),
    (1180.0, 503.5),
    (1193.0, 554.5),
    (1180.0, 604.0),
    (1133.0, 635.0),
    (1015.0, 635.0),
    (901.5, 635.0),
    (796.5, 635.0),
    (697.5, 626.0),
    (601.0, 602.5),
    (498.0, 563.5),
    (415.0, 518.5),
    (368.5, 483.5),
]

SVG_WAYPOINTS_M = [
    (x / PX_PER_M, (IMG_H_PX - y) / PX_PER_M) for x, y in SVG_WAYPOINTS_PX
]


def floor_scale(depth_norm: np.ndarray) -> float:
    H, W = depth_norm.shape
    phi_thresh = FLOOR_ELEV_DEG * math.pi / 180.0
    v_thresh = int((0.5 - phi_thresh / math.pi) * H)
    v_thresh = max(0, min(v_thresh, H - 1))

    v_rows = np.arange(v_thresh, H)
    if len(v_rows) == 0:
        return MAX_DEPTH_M

    phi_rows = math.pi / 2 - v_rows / H * math.pi
    expected = CAMERA_HEIGHT_M / np.abs(np.sin(phi_rows))
    measured = depth_norm[v_rows, :].mean(axis=1)

    valid = measured > 0.02
    if not valid.any():
        return MAX_DEPTH_M

    scales = expected[valid] / measured[valid]
    return float(np.median(scales))


def svg_waypoint_scale(
    depth_norm: np.ndarray,
    cam_x: float,
    cam_y: float,
    cam_heading: float,
    W: int,
    H: int,
) -> float | None:
    best_scale = None
    best_dist = float("inf")

    for wx, wy in SVG_WAYPOINTS_M:
        dx = wx - cam_x
        dy = wy - cam_y
        dz = -CAMERA_HEIGHT_M
        real_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if real_dist < 0.5:
            continue

        world_az = math.atan2(dy, dx)
        cam_az = world_az - cam_heading
        cam_el = math.atan2(dz, math.sqrt(dx**2 + dy**2))

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


def combined_scale(
    depth_norm: np.ndarray,
    cam_x: float,
    cam_y: float,
    cam_heading: float,
    W: int,
    H: int,
) -> float:
    s_floor = floor_scale(depth_norm)
    s_svg = svg_waypoint_scale(depth_norm, cam_x, cam_y, cam_heading, W, H)
    if s_svg is None:
        return s_floor
    return 0.35 * s_floor + 0.65 * s_svg


def camera_heading_from_trajectory(
    frames_xy: list[tuple[float, float]],
) -> list[float]:
    """Heading (rad) from motion; atan2(dy, dx) in world metres (same as pointcloud)."""
    n = len(frames_xy)
    out = []
    for i in range(n):
        if i < n - 1:
            x0, y0 = frames_xy[i]
            x1, y1 = frames_xy[i + 1]
            if abs(x1 - x0) > 1e-6 or abs(y1 - y0) > 1e-6:
                out.append(math.atan2(y1 - y0, x1 - x0))
                continue
        out.append(out[-1] if out else 0.0)
    return out


def ground_plane_hits(
    depth_norm: np.ndarray,
    cam_x: float,
    cam_y: float,
    cam_heading: float,
    subsample: int,
    occlusion_margin_m: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For subsampled equirect pixels, return (gx, gy, valid) world floor (z=0) intersections
    in metres when the ray reaches the ground and depth indicates the ground is visible.

    World XY matches frame_positions.json (y = metres from bottom of plan).
    """
    H, W = depth_norm.shape
    scale = combined_scale(depth_norm, cam_x, cam_y, cam_heading, W, H)
    depth_m = np.clip(depth_norm * scale, 0.1, MAX_DEPTH_M)

    v_idx = np.arange(0, H, subsample)
    u_idx = np.arange(0, W, subsample)
    vv, uu = np.meshgrid(v_idx, u_idx, indexing="ij")

    theta = (uu / W) * 2 * math.pi
    phi = math.pi / 2 - (vv / H) * math.pi

    dx_c = np.cos(phi) * np.cos(theta)
    dy_c = np.cos(phi) * np.sin(theta)
    dz_c = np.sin(phi)

    cos_h, sin_h = math.cos(cam_heading), math.sin(cam_heading)
    dx_w = cos_h * dx_c - sin_h * dy_c
    dy_w = sin_h * dx_c + cos_h * dy_c
    dz_w = dz_c

    d = depth_m[vv, uu]

    # Ray: P(t) = C + t * u, C_z = CAMERA_HEIGHT_M; ground at z=0 -> t = -C_z / dz_w
    t_ground = np.full_like(dz_w, np.inf, dtype=np.float64)
    downward = dz_w < -1e-5
    t_ground[downward] = -CAMERA_HEIGHT_M / dz_w[downward]

    visible_ground = (
        downward
        & np.isfinite(t_ground)
        & (t_ground > 0.2)
        & (t_ground < MAX_DEPTH_M * 0.95)
        & (d + occlusion_margin_m >= t_ground)
    )

    gx = np.empty_like(dx_w, dtype=np.float64)
    gy = np.empty_like(dy_w, dtype=np.float64)
    gx[visible_ground] = cam_x + t_ground[visible_ground] * dx_w[visible_ground]
    gy[visible_ground] = cam_y + t_ground[visible_ground] * dy_w[visible_ground]

    return gx[visible_ground], gy[visible_ground]


def load_depth_norm(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def parse_svg_waypoints(svg_path: Path) -> List[Tuple[float, float]]:
    """Polyline vertices in SVG pixel space (same logic as sync_gps_video.py)."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    path_el = root.find(".//{http://www.w3.org/2000/svg}path")
    if path_el is None:
        path_el = root.find(".//path")
    if path_el is None:
        raise ValueError(f"No <path> in {svg_path}")

    d = path_el.attrib.get("d", "")
    segments = re.findall(r"([MLHVZmlhvz])([^MLHVZmlhvz]*)", d)

    pts: List[Tuple[float, float]] = []
    cx, cy = 0.0, 0.0
    for cmd, args_str in segments:
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", args_str)]
        if cmd == "M":
            cx, cy = nums[0], nums[1]
            pts.append((cx, cy))
            for i in range(2, len(nums) - 1, 2):
                cx, cy = nums[i], nums[i + 1]
                pts.append((cx, cy))
        elif cmd == "L":
            for i in range(0, len(nums) - 1, 2):
                cx, cy = nums[i], nums[i + 1]
                pts.append((cx, cy))
        elif cmd == "H":
            for val in nums:
                cx = val
                pts.append((cx, cy))
        elif cmd == "V":
            for val in nums:
                cy = val
                pts.append((cx, cy))

    deduped = [pts[0]] if pts else []
    for pt in pts[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped
