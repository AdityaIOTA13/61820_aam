"""
estimate_depth.py

Runs Depth Anything V2 Small on every frame in outputs/frames/ and saves
normalised grayscale depth maps as PNGs into outputs/depth_maps/.

Model: depth-anything/Depth-Anything-V2-Small-hf (HuggingFace)

Run from repo root:  python scripts/estimate_depth.py
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import torch
    from transformers import pipeline
except ImportError:
    sys.exit("ERROR: Run `pip install torch transformers` first.")

ROOT = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "outputs" / "frames"
DEPTH_DIR  = ROOT / "outputs" / "depth_maps"
MODEL_ID   = "depth-anything/Depth-Anything-V2-Small-hf"


def load_pipeline():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading {MODEL_ID} on {device} ...")
    pipe = pipeline(task="depth-estimation", model=MODEL_ID, device=device)
    print("Model loaded.\n")
    return pipe


def save_depth(depth_array: np.ndarray, out_path: Path) -> None:
    d_min, d_max = depth_array.min(), depth_array.max()
    if d_max > d_min:
        normalised = ((depth_array - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalised = np.zeros_like(depth_array, dtype=np.uint8)
    Image.fromarray(normalised).save(str(out_path))


def main():
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        sys.exit(f"ERROR: No frames in '{FRAMES_DIR.relative_to(ROOT)}/'. Run extract_frames.py first.")

    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline()
    n = len(frame_paths)
    print(f"Processing {n} frames → '{DEPTH_DIR.relative_to(ROOT)}/'")

    for i, fp in enumerate(frame_paths, 1):
        out_path = DEPTH_DIR / f"{fp.stem}_depth.png"
        if out_path.exists():
            print(f"  [{i:>4}/{n}] skip (exists): {fp.name}", end="\r")
            continue
        image = Image.open(fp).convert("RGB")
        result = pipe(image)
        save_depth(np.array(result["depth"]), out_path)
        print(f"  [{i:>4}/{n}] {out_path.name}", end="\r")

    print(f"\nDone. Depth maps saved to '{DEPTH_DIR.relative_to(ROOT)}/'")


if __name__ == "__main__":
    main()
