"""
Download a few frames + depth maps from the public R2 CDN for local smoke tests.

  python scripts/fetch_sample_frames.py --count 3

Writes into outputs/frames/ and outputs/depth_maps/ (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "outputs" / "frames"
DEPTH_DIR = ROOT / "outputs" / "depth_maps"
POSITIONS_JSON = ROOT / "outputs" / "frame_positions.json"

CDN = "https://assets02.aitkena.com/courtyard_360"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of frames from start of walk (ignored if --all)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Download every frame + depth listed in frame_positions.json (full walk)",
    )
    args = ap.parse_args()

    if not POSITIONS_JSON.exists():
        sys.exit(f"Missing {POSITIONS_JSON} (need frame filenames).")

    import json

    with open(POSITIONS_JSON, encoding="utf-8") as f:
        positions = json.load(f)

    ordered = sorted(positions.keys(), key=lambda n: positions[n]["timestamp_sec"])
    if args.all:
        names = ordered
    else:
        names = ordered[: args.count]
    if not names:
        sys.exit("No frame keys in frame_positions.json")

    for name in names:
        stem = Path(name).stem
        fu = f"{CDN}/frames/{name}"
        du = f"{CDN}/depth_maps/{stem}_depth.png"
        fp = FRAMES_DIR / name
        dp = DEPTH_DIR / f"{stem}_depth.png"
        if not fp.exists():
            print(f"GET {fu}")
            download(fu, fp)
        else:
            print(f"skip exists {fp.name}")
        if not dp.exists():
            print(f"GET {du}")
            download(du, dp)
        else:
            print(f"skip exists {dp.name}")

    n = len(names)
    print(f"Done ({n} frames). Run: python scripts/project_coverage.py")


if __name__ == "__main__":
    main()
