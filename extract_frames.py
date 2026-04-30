"""
extract_frames.py

Downloads the 360° courtyard video from R2 (or uses a local copy) and
extracts frames at 2 fps into frames/.
"""

import os
import sys
import cv2
import requests

VIDEO_URL = "https://assets02.aitkena.com/courtyard_360/VID_20260429_143550_00_014.mp4"
LOCAL_VIDEO = os.path.join("data", "equirectangular export insta360", "VID_20260429_143550_00_014.mp4")
DOWNLOAD_PATH = os.path.join("data", "video_cache.mp4")
FRAMES_DIR = "frames"
EXTRACT_FPS = 2


def download_video(url: str, dest: str) -> None:
    print(f"Downloading video from {url} ...")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.1f}%)", end="\r")
    print(f"\nDownload complete → {dest}")


def resolve_video() -> str:
    """Return path to a local copy of the video, downloading if needed."""
    if os.path.exists(LOCAL_VIDEO):
        print(f"Using local video: {LOCAL_VIDEO}")
        return LOCAL_VIDEO
    if os.path.exists(DOWNLOAD_PATH):
        print(f"Using cached download: {DOWNLOAD_PATH}")
        return DOWNLOAD_PATH
    download_video(VIDEO_URL, DOWNLOAD_PATH)
    return DOWNLOAD_PATH


def extract_frames(video_path: str, out_dir: str, fps: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open video at {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / src_fps
    frame_interval = max(1, round(src_fps / fps))

    print(f"Video: {total_frames} frames @ {src_fps:.2f} fps  →  {duration_sec:.1f}s")
    print(f"Extracting every {frame_interval} frames ({fps} fps output) into '{out_dir}/'")

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / src_fps
            filename = f"frame_{saved:05d}_{timestamp_sec:.3f}s.jpg"
            cv2.imwrite(os.path.join(out_dir, filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            print(f"  Saved {saved:>4} frames  (t={timestamp_sec:.2f}s)", end="\r")
        frame_idx += 1

    cap.release()
    print(f"\nDone. {saved} frames saved to '{out_dir}/'")


if __name__ == "__main__":
    video_path = resolve_video()
    extract_frames(video_path, FRAMES_DIR, EXTRACT_FPS)
