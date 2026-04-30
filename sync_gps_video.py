"""
sync_gps_video.py

Matches every extracted frame in frames/ to a real-world GPS position on the
floorplan using linear interpolation between SVG waypoints.

ASSUMPTION: The SVG path (data/svg_path.svg) contains no per-waypoint time
metadata. Waypoints are therefore assumed to be evenly distributed across the
full video duration. For a 14-waypoint path over a T-second video, waypoint i
is assigned timestamp t_i = i * T / (N-1). Frame timestamps are read from
their filenames (produced by extract_frames.py as "frame_NNNNN_T.TTTs.jpg").
If GPS time metadata becomes available in the SVG, replace the uniform
distribution below with the actual timestamps.

Floorplan scale: 1400 px = 67 m  →  1 px = 0.04786 m
SVG coordinate origin: top-left (y increases downward).
Output origin: bottom-left (y increases upward), matching floorplan_grid.png.

Outputs
-------
frame_positions.json  —  maps each frame filename to:
    {
        "x_px": float,       # position in SVG/floorplan pixel space
        "y_px": float,
        "x_meters": float,   # position in real-world metres (origin bottom-left)
        "y_meters": float,
        "timestamp_sec": float
    }
"""

import os
import re
import sys
import json
import glob
import xml.etree.ElementTree as ET
from typing import List, Tuple

SVG_PATH = os.path.join("data", "svg_path.svg")
FRAMES_DIR = "frames"
OUTPUT_JSON = "frame_positions.json"

# Floorplan scale constants
IMG_W_PX = 1400
IMG_H_PX = 819
REAL_W_M = 67.0
PX_PER_M = IMG_W_PX / REAL_W_M  # 20.896 px/m


# ── SVG path parser ────────────────────────────────────────────────────────────

def parse_svg_waypoints(svg_path: str) -> List[Tuple[float, float]]:
    """
    Parse the <path d="..."> element from the SVG and return a list of
    (x_px, y_px) waypoints in SVG pixel space (origin top-left).
    Handles M, L, H, V, Z commands (absolute only).
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    path_el = root.find(".//svg:path", ns) or root.find(".//path")
    if path_el is None:
        sys.exit(f"ERROR: No <path> element found in {svg_path}")

    d = path_el.attrib.get("d", "")
    tokens = re.findall(r"[MLHVZmlhvz]|[-+]?\d*\.?\d+", d)

    waypoints: List[Tuple[float, float]] = []
    cx, cy = 0.0, 0.0
    cmd = None

    for tok in tokens:
        if tok in "MLHVZmlhvz":
            cmd = tok
            continue
        val = float(tok)
        if cmd == "M":
            cx, cy = val, float(tokens[tokens.index(tok) + 1])
            waypoints.append((cx, cy))
            tokens.pop(tokens.index(tok) + 1)
            cmd = "L"
        elif cmd == "L":
            if not waypoints or (cx, cy) != waypoints[-1]:
                waypoints.append((cx, cy))
            cy_new = float(tokens[tokens.index(tok) + 1])
            cx, cy = val, cy_new
            waypoints.append((cx, cy))
            tokens.pop(tokens.index(tok) + 1)
        elif cmd == "H":
            cx = val
            waypoints.append((cx, cy))
        elif cmd == "V":
            cy = val
            waypoints.append((cx, cy))

    # Deduplicate consecutive identical points
    deduped = [waypoints[0]]
    for pt in waypoints[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped


def parse_svg_waypoints_robust(svg_path: str) -> List[Tuple[float, float]]:
    """
    Robust token-stream SVG path parser that handles all absolute commands.
    Returns list of (x_px, y_px) waypoints.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    path_el = root.find(".//svg:path", ns) or root.find(".//path")
    if path_el is None:
        sys.exit(f"ERROR: No <path> element found in {svg_path}")

    d = path_el.attrib.get("d", "")

    # Split into (command, [numbers]) segments
    segments = re.findall(r"([MLHVZmlhvz])([^MLHVZmlhvz]*)", d)

    waypoints: List[Tuple[float, float]] = []
    cx, cy = 0.0, 0.0

    for cmd, args_str in segments:
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", args_str)]
        if cmd == "M":
            cx, cy = nums[0], nums[1]
            waypoints.append((cx, cy))
            for i in range(2, len(nums) - 1, 2):
                cx, cy = nums[i], nums[i + 1]
                waypoints.append((cx, cy))
        elif cmd == "L":
            for i in range(0, len(nums) - 1, 2):
                cx, cy = nums[i], nums[i + 1]
                waypoints.append((cx, cy))
        elif cmd == "H":
            for val in nums:
                cx = val
                waypoints.append((cx, cy))
        elif cmd == "V":
            for val in nums:
                cy = val
                waypoints.append((cx, cy))
        elif cmd == "Z":
            pass  # close path; no new point needed

    # Deduplicate
    deduped = [waypoints[0]] if waypoints else []
    for pt in waypoints[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped


# ── Path length helpers ────────────────────────────────────────────────────────

def cumulative_distances(pts: List[Tuple[float, float]]) -> List[float]:
    """Return cumulative Euclidean distances along the path (starting at 0)."""
    dists = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        dists.append(dists[-1] + (dx ** 2 + dy ** 2) ** 0.5)
    return dists


def interpolate_position(pts, cum_dists, frac: float) -> Tuple[float, float]:
    """Return (x, y) at fractional distance `frac` (0–1) along the path."""
    target = frac * cum_dists[-1]
    for i in range(1, len(cum_dists)):
        if cum_dists[i] >= target:
            seg_len = cum_dists[i] - cum_dists[i - 1]
            t = (target - cum_dists[i - 1]) / seg_len if seg_len > 0 else 0.0
            x = pts[i - 1][0] + t * (pts[i][0] - pts[i - 1][0])
            y = pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1])
            return x, y
    return pts[-1]


# ── Frame timestamp extraction ─────────────────────────────────────────────────

def frame_timestamp(filename: str) -> float:
    """
    Extract timestamp (seconds) from frame filename.
    Expected format: frame_NNNNN_T.TTTs.jpg  (produced by extract_frames.py)
    Falls back to 0.0 if pattern not found.
    """
    m = re.search(r"_(\d+\.\d+)s", filename)
    return float(m.group(1)) if m else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # --- Load waypoints ---
    waypoints_px = parse_svg_waypoints_robust(SVG_PATH)
    print(f"Parsed {len(waypoints_px)} waypoints from {SVG_PATH}")

    cum_dists = cumulative_distances(waypoints_px)

    # --- Load frames ---
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        sys.exit(f"ERROR: No frames in '{FRAMES_DIR}/'. Run extract_frames.py first.")
    print(f"Found {len(frame_paths)} frames in '{FRAMES_DIR}/'")

    # --- Determine video duration from last frame timestamp ---
    timestamps = [frame_timestamp(os.path.basename(fp)) for fp in frame_paths]
    video_duration = max(timestamps) if max(timestamps) > 0 else len(frame_paths) / 2.0

    if video_duration == 0:
        sys.exit("ERROR: Could not determine video duration from frame filenames.")

    print(f"Video duration (from frames): {video_duration:.2f}s")
    print("NOTE: Waypoints assumed evenly distributed over video duration (no GPS time in SVG).\n")

    # --- Build frame_positions ---
    results = {}
    matched = 0

    for fp in frame_paths:
        fname = os.path.basename(fp)
        t = frame_timestamp(fname)
        frac = t / video_duration  # 0.0 at start, 1.0 at end

        x_px, y_px = interpolate_position(waypoints_px, cum_dists, frac)

        # Convert to metres (flip Y: SVG top-left → output bottom-left)
        x_m = x_px / PX_PER_M
        y_m = (IMG_H_PX - y_px) / PX_PER_M

        results[fname] = {
            "x_px": round(x_px, 2),
            "y_px": round(y_px, 2),
            "x_meters": round(x_m, 4),
            "y_meters": round(y_m, 4),
            "timestamp_sec": round(t, 4),
        }
        matched += 1

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Matched {matched}/{len(frame_paths)} frames → written to '{OUTPUT_JSON}'")
    print(f"  X range: {min(v['x_meters'] for v in results.values()):.2f}m – "
          f"{max(v['x_meters'] for v in results.values()):.2f}m")
    print(f"  Y range: {min(v['y_meters'] for v in results.values()):.2f}m – "
          f"{max(v['y_meters'] for v in results.values()):.2f}m")
    print(f"  Time  : 0.0s – {max(v['timestamp_sec'] for v in results.values()):.2f}s")


if __name__ == "__main__":
    main()
