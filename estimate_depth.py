"""
estimate_depth.py

Runs Depth Anything V2 (Small, HuggingFace) on every frame in frames/
and saves grayscale depth maps as PNGs into depth_maps/.

Model: depth-anything/Depth-Anything-V2-Small-hf
  - Input:  RGB frame
  - Output: relative depth map (normalised 0–255 grayscale)
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import pipeline
except ImportError:
    sys.exit("ERROR: Run `pip install torch transformers` first.")

FRAMES_DIR = "frames"
DEPTH_DIR = "depth_maps"
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} ...")
    pipe = pipeline(
        task="depth-estimation",
        model=MODEL_ID,
        device=device,
    )
    print("Model loaded.\n")
    return pipe


def save_depth(depth_array: np.ndarray, out_path: str) -> None:
    """Normalise depth array to 0–255 and save as grayscale PNG."""
    d_min, d_max = depth_array.min(), depth_array.max()
    if d_max > d_min:
        normalised = ((depth_array - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalised = np.zeros_like(depth_array, dtype=np.uint8)
    Image.fromarray(normalised).save(out_path)


def main():
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        sys.exit(f"ERROR: No frames found in '{FRAMES_DIR}/'. Run extract_frames.py first.")

    os.makedirs(DEPTH_DIR, exist_ok=True)
    pipe = load_pipeline()

    print(f"Processing {len(frame_paths)} frames → '{DEPTH_DIR}/'")
    for i, fp in enumerate(frame_paths, 1):
        basename = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(DEPTH_DIR, f"{basename}_depth.png")

        if os.path.exists(out_path):
            print(f"  [{i:>4}/{len(frame_paths)}] Skipping (exists): {basename}", end="\r")
            continue

        image = Image.open(fp).convert("RGB")
        result = pipe(image)
        depth_array = np.array(result["depth"])
        save_depth(depth_array, out_path)

        print(f"  [{i:>4}/{len(frame_paths)}] {basename}_depth.png", end="\r")

    print(f"\nDone. Depth maps saved to '{DEPTH_DIR}/'")


if __name__ == "__main__":
    main()
