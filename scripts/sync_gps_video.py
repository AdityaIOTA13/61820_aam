"""
sync_gps_video.py

Matches every extracted frame in outputs/frames/ to a real-world GPS position
on the floorplan using linear interpolation between SVG waypoints.

ASSUMPTION: The SVG path contains no per-waypoint time metadata. Waypoints are
therefore assumed to be evenly distributed across the full video duration. For a
14-waypoint path over a T-second video, waypoint i is assigned timestamp
t_i = i * T / (N-1). Frame timestamps are read from filenames produced by
extract_frames.py ("frame_NNNNN_T.TTTs.jpg"). Replace uniform distribution
below with actual GPS timestamps if they become available.

Floorplan scale: 1400 px = 67 m  →  1 px = 0.04786 m
SVG origin: top-left (y↓). Output origin: bottom-left (y↑), matching grid PNGs.

Output
------
outputs/frame_positions.json — maps each frame filename to:
    { x_px, y_px, x_meters, y_meters, timestamp_sec }

Run from repo root:  python scripts/sync_gps_video.py
"""

import re
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

ROOT       = Path(__file__).parent.parent
SVG_PATH   = ROOT / "data" / "svg_path.svg"
FRAMES_DIR = ROOT / "outputs" / "frames"
OUTPUT_JSON = ROOT / "outputs" / "frame_positions.json"

IMG_W_PX  = 1400
IMG_H_PX  = 819
REAL_W_M  = 67.0
PX_PER_M  = IMG_W_PX / REAL_W_M


# ── SVG parser ─────────────────────────────────────────────────────────────────

def parse_svg_waypoints(svg_path: Path) -> List[Tuple[float, float]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    path_el = root.find(".//{http://www.w3.org/2000/svg}path")
    if path_el is None:
        path_el = root.find(".//path")
    if path_el is None:
        sys.exit(f"ERROR: No <path> element in {svg_path}")

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


# ── Interpolation helpers ──────────────────────────────────────────────────────

def cumulative_distances(pts: List[Tuple[float, float]]) -> List[float]:
    dists = [0.0]
    for i in range(1, len(pts)):
        dx, dy = pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]
        dists.append(dists[-1] + (dx**2 + dy**2) ** 0.5)
    return dists


def interpolate_position(pts, cum_dists, frac: float) -> Tuple[float, float]:
    target = frac * cum_dists[-1]
    for i in range(1, len(cum_dists)):
        if cum_dists[i] >= target:
            seg = cum_dists[i] - cum_dists[i-1]
            t = (target - cum_dists[i-1]) / seg if seg > 0 else 0.0
            x = pts[i-1][0] + t * (pts[i][0] - pts[i-1][0])
            y = pts[i-1][1] + t * (pts[i][1] - pts[i-1][1])
            return x, y
    return pts[-1]


def frame_timestamp(filename: str) -> float:
    m = re.search(r"_(\d+\.\d+)s", filename)
    return float(m.group(1)) if m else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    waypoints = parse_svg_waypoints(SVG_PATH)
    print(f"Parsed {len(waypoints)} waypoints from {SVG_PATH.relative_to(ROOT)}")
    cum_dists = cumulative_distances(waypoints)

    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        sys.exit(f"ERROR: No frames in '{FRAMES_DIR.relative_to(ROOT)}/'. Run extract_frames.py first.")
    print(f"Found {len(frame_paths)} frames")

    timestamps = [frame_timestamp(fp.name) for fp in frame_paths]
    video_duration = max(timestamps) if max(timestamps) > 0 else len(frame_paths) / 2.0
    print(f"Video duration (from frames): {video_duration:.2f}s")
    print("NOTE: Waypoints distributed evenly over duration (no GPS timestamps in SVG).\n")

    results = {}
    for fp in frame_paths:
        t = frame_timestamp(fp.name)
        frac = t / video_duration
        x_px, y_px = interpolate_position(waypoints, cum_dists, frac)
        results[fp.name] = {
            "x_px":          round(x_px, 2),
            "y_px":          round(y_px, 2),
            "x_meters":      round(x_px / PX_PER_M, 4),
            "y_meters":      round((IMG_H_PX - y_px) / PX_PER_M, 4),
            "timestamp_sec": round(t, 4),
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    vals = list(results.values())
    print(f"Matched {len(results)} frames → {OUTPUT_JSON.relative_to(ROOT)}")
    print(f"  X: {min(v['x_meters'] for v in vals):.2f}m – {max(v['x_meters'] for v in vals):.2f}m")
    print(f"  Y: {min(v['y_meters'] for v in vals):.2f}m – {max(v['y_meters'] for v in vals):.2f}m")
    print(f"  T: 0.0s – {max(v['timestamp_sec'] for v in vals):.2f}s")


if __name__ == "__main__":
    main()
