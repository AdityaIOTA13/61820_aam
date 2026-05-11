"""
process_cambridge.py

Full pipeline for the Cambridge Commons multi-clip dataset (data_02/).

Steps
-----
1. Extract frames at EXTRACT_FPS from each video clip
   (uses local file; downloads from R2 CDN if missing).
2. Sync each frame to real GPS using the paired GPX file.
   frame_abs_time = video_creation_time_utc + frame_offset_sec → interpolate GPX.
3. Write data_02_outputs/frame_positions.json (clip_id, lat, lon, x_m, y_m,
   heading, speed, elevation, timestamp).
4. Run Depth Anything V2 Small → data_02_outputs/depth_maps/
5. Run coverage projection → data_02_outputs/coverage/

Frames are NOT uploaded to R2. Teammates regenerate with:
  python scripts/process_cambridge.py --extract-only
Videos are at: https://assets02.aitkena.com/cambridge_commons_360/

Run from repo root:
  python scripts/process_cambridge.py
  python scripts/process_cambridge.py --extract-only
  python scripts/process_cambridge.py --skip-depth
"""

from __future__ import annotations

import argparse, json, math, subprocess, sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data_02" / "insta360_export"
DATA_OUT  = ROOT / "data_02_outputs"
FRAMES_DIR   = DATA_OUT / "frames"
DEPTH_DIR    = DATA_OUT / "depth_maps"
RAW_DIR      = DATA_OUT / "raw_depths"
OUT_JSON     = DATA_OUT / "frame_positions.json"
COVERAGE_DIR = DATA_OUT / "coverage"

CDN_BASE    = "https://assets02.aitkena.com/cambridge_commons_360"
EXTRACT_FPS = 2
NS = {"gpx": "http://www.topografix.com/GPX/1/1"}
M_PER_DEG_LAT = 111_319.0


# ── GPX ───────────────────────────────────────────────────────────────────────

def parse_gpx(path: Path) -> list[dict]:
    tree = ET.parse(path)
    pts = []
    for tp in tree.getroot().findall(".//gpx:trkpt", NS):
        t_el = tp.find("gpx:time", NS)
        if t_el is None:
            continue
        pts.append({
            "t": datetime.fromisoformat(t_el.text.replace("Z", "+00:00")).timestamp(),
            "lat": float(tp.attrib["lat"]),
            "lon": float(tp.attrib["lon"]),
            "ele": float(tp.findtext("gpx:ele", "0", NS)),
            "speed":  float(tp.findtext(".//speed", "0")),
            "course": float(tp.findtext(".//course", "0")),
        })
    return sorted(pts, key=lambda p: p["t"])


def interp_gps(pts: list[dict], t: float) -> dict | None:
    if not pts:
        return None
    t = max(pts[0]["t"], min(pts[-1]["t"], t))
    for i in range(len(pts) - 1):
        if pts[i]["t"] <= t <= pts[i+1]["t"]:
            dt = pts[i+1]["t"] - pts[i]["t"]
            a = (t - pts[i]["t"]) / dt if dt > 0 else 0.0
            return {k: pts[i][k] + a * (pts[i+1][k] - pts[i][k])
                    for k in ("lat", "lon", "ele", "speed", "course")}
    return pts[-1].copy()


def gps_origin(all_pts: list[dict]) -> tuple[float, float]:
    lats = [p["lat"] for p in all_pts]
    lons = [p["lon"] for p in all_pts]
    return sum(lats)/len(lats), sum(lons)/len(lons)


def to_local_m(lat, lon, olat, olon):
    x = (lon - olon) * M_PER_DEG_LAT * math.cos(math.radians(olat))
    y = (lat - olat) * M_PER_DEG_LAT
    return x, y


# ── Video ─────────────────────────────────────────────────────────────────────

def video_creation_utc(mp4: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(mp4)],
        capture_output=True, text=True, check=True)
    tags = json.loads(r.stdout)["format"].get("tags", {})
    s = tags.get("creation_time", "")
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else 0.0


def resolve_video(gpx_path: Path) -> Path | None:
    local = gpx_path.with_suffix(".mp4")
    if local.exists():
        return local
    cache = DATA_OUT / "video_cache" / local.name
    if cache.exists():
        return cache
    url = f"{CDN_BASE}/{local.name}"
    print(f"  Downloading {local.name} ...")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(cache, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"    {done/1e6:.0f}/{total/1e6:.0f} MB", end="\r")
    print()
    return cache


def extract_frames(mp4: Path, out_dir: Path, fps: int, clip_id: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(src_fps / fps))
    records, saved, idx = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            t = idx / src_fps
            fname = f"clip{clip_id}_frame_{saved:05d}_{t:.3f}s.jpg"
            cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append({"filename": fname, "clip_id": clip_id, "offset_sec": t})
            saved += 1
        idx += 1
    cap.release()
    return records


# ── Depth estimation (inline, no hardcoded-path dependency) ───────────────────

def run_depth(frames_dir: Path, depth_dir: Path, raw_dir: Path) -> None:
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        from PIL import Image
    except ImportError:
        sys.exit("Run: pip install torch transformers pillow")

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print("  No frames found, skipping depth."); return

    depth_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading Depth Anything V2 Small on {device} ...")
    pipe = hf_pipeline(task="depth-estimation",
                       model="depth-anything/Depth-Anything-V2-Small-hf",
                       device=device)
    print(f"  Processing {len(frame_paths)} frames ...")
    for i, fp in enumerate(frame_paths, 1):
        npy_out = raw_dir   / f"{fp.stem}_depth.npy"
        png_out = depth_dir / f"{fp.stem}_depth.png"
        if npy_out.exists() and png_out.exists():
            print(f"  [{i:>4}/{len(frame_paths)}] skip {fp.name}", end="\r")
            continue
        img = Image.open(fp).convert("RGB")
        raw = np.array(pipe(img)["depth"], dtype=np.float32)
        np.save(str(npy_out), raw)
        d_min, d_max = raw.min(), raw.max()
        norm = ((raw - d_min) / (d_max - d_min) * 255).astype(np.uint8) if d_max > d_min \
               else np.zeros_like(raw, dtype=np.uint8)
        Image.fromarray(norm).save(str(png_out))
        print(f"  [{i:>4}/{len(frame_paths)}] {fp.name}", end="\r")
    print(f"\n  Depth maps saved to {depth_dir.relative_to(ROOT)}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", action="store_true",
                    help="Only extract frames + GPS sync, skip depth + coverage")
    ap.add_argument("--skip-depth", action="store_true",
                    help="Skip depth estimation step")
    args = ap.parse_args()

    gpx_files = sorted(DATA_DIR.glob("*.gpx"))
    if not gpx_files:
        sys.exit(f"No GPX files in {DATA_DIR}")
    print(f"Found {len(gpx_files)} clips\n")

    # ── Extract frames + parse GPX ─────────────────────────────────────────
    all_gpx_pts: list[dict] = []
    clips: list[tuple] = []

    for gpx_path in gpx_files:
        clip_id = gpx_path.stem.split("_")[-1]
        mp4 = resolve_video(gpx_path)
        if mp4 is None:
            print(f"  SKIP clip {clip_id}: video unavailable"); continue

        gpx_pts = parse_gpx(gpx_path)
        all_gpx_pts.extend(gpx_pts)
        print(f"Clip {clip_id}: {len(gpx_pts)} GPS pts", end=" ... ")

        frame_recs = extract_frames(mp4, FRAMES_DIR, EXTRACT_FPS, clip_id)
        print(f"{len(frame_recs)} frames")
        clips.append((clip_id, gpx_pts, frame_recs, mp4))

    # ── Build frame_positions.json ─────────────────────────────────────────
    olat, olon = gps_origin(all_gpx_pts)
    print(f"\nLocal origin: {olat:.6f}°N  {olon:.6f}°W")

    positions: dict = {}
    for clip_id, gpx_pts, frame_recs, mp4 in clips:
        t0_utc = video_creation_utc(mp4)
        for rec in frame_recs:
            abs_t = t0_utc + rec["offset_sec"]
            gps = interp_gps(gpx_pts, abs_t)
            if gps is None:
                continue
            xm, ym = to_local_m(gps["lat"], gps["lon"], olat, olon)
            positions[rec["filename"]] = {
                "clip_id":         clip_id,
                "clip_offset_sec": round(rec["offset_sec"], 4),
                "abs_utc_sec":     round(abs_t, 3),
                "lat":             round(gps["lat"], 8),
                "lon":             round(gps["lon"], 8),
                "ele_m":           round(gps["ele"], 3),
                "x_meters":        round(xm, 4),
                "y_meters":        round(ym, 4),
                "speed_m_s":       round(gps["speed"], 4),
                "heading_deg":     round(gps["course"], 2),
            }

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    meta = {"origin_lat": olat, "origin_lon": olon,
            "n_clips": len(clips), "n_frames": len(positions),
            "gps_source": "real GPX 1Hz (Insta360 insv_to_gpx)",
            "cdn_video_base": CDN_BASE}
    with open(OUT_JSON, "w") as f:
        json.dump({"meta": meta, "frames": positions}, f, indent=2)

    total_frames = sum(len(c[2]) for c in clips)
    print(f"\nframe_positions.json: {len(positions)}/{total_frames} frames matched")
    print(f"  Saved: {OUT_JSON.relative_to(ROOT)}\n")

    if args.extract_only:
        print("--extract-only: done."); return

    # ── Depth estimation ────────────────────────────────────────────────────
    if not args.skip_depth:
        print("Running depth estimation ...")
        run_depth(FRAMES_DIR, DEPTH_DIR, RAW_DIR)

    # ── Coverage projection ─────────────────────────────────────────────────
    print("\nRunning coverage projection ...")
    # project_coverage.py reads from outputs/ by default; override via env
    import os
    env = os.environ.copy()
    result = subprocess.run([
        sys.executable, "-c",
        f"""
import sys; sys.path.insert(0, '{ROOT}/scripts')
import project_coverage as pc
pc.FRAMES_DIR      = pc.Path('{FRAMES_DIR}')
pc.DEPTH_DIR       = pc.Path('{DEPTH_DIR}')
pc.POSITIONS_JSON  = pc.Path('{OUT_JSON}')
pc.OUT_DIR         = pc.Path('{COVERAGE_DIR}')
pc.main()
"""
    ], env=env, check=False)
    if result.returncode != 0:
        print("  Coverage projection failed — run manually:")
        print(f"  python scripts/project_coverage.py (after symlinking outputs/ to data_02_outputs/)")


if __name__ == "__main__":
    main()
