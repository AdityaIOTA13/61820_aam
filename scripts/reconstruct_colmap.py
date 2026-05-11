"""
Run a simple COLMAP sparse reconstruction from extracted 360 frames.

The pipeline converts equirectangular frames in outputs/frames/ into
forward-facing perspective crops, then runs COLMAP feature extraction,
matching, mapping, and PLY export.

Example:
  python scripts/reconstruct_colmap.py --max-frames 30
  python scripts/reconstruct_colmap.py --selection-list selected_frames.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
DEFAULT_FRAMES_DIR = ROOT / "outputs" / "frames"
DEFAULT_POSITIONS_JSON = ROOT / "outputs" / "frame_positions.json"
DEFAULT_OUT_DIR = ROOT / "outputs" / "colmap"

DEFAULT_HFOV_DEG = 100.0
DEFAULT_OUT_W = 1280
DEFAULT_OUT_H = 720


def parse_timestamp(name: str) -> float | None:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)s(?:\.[^.]+)?$", name)
    if match:
        return float(match.group(1))
    return None


def load_positions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    positions = raw.get("frames", raw) if isinstance(raw, dict) else {}
    if not isinstance(positions, dict):
        return {}

    times = [
        v.get("timestamp_sec", v.get("abs_utc_sec"))
        for v in positions.values()
        if isinstance(v, dict) and (v.get("timestamp_sec") is not None or v.get("abs_utc_sec") is not None)
    ]
    t0 = min(times) if times else 0.0
    for value in positions.values():
        if not isinstance(value, dict):
            continue
        if "timestamp_sec" not in value:
            value["timestamp_sec"] = float(value.get("abs_utc_sec", t0) - t0)
        if "heading_rad" not in value and "heading_deg" in value:
            value["heading_rad"] = math.radians(float(value["heading_deg"]))
    return positions


def frame_timestamp(path: Path, positions: dict[str, dict]) -> float:
    if path.name in positions and "timestamp_sec" in positions[path.name]:
        return float(positions[path.name]["timestamp_sec"])
    parsed = parse_timestamp(path.name)
    return parsed if parsed is not None else 0.0


def heading_from_positions(positions: dict[str, dict], ordered_names: list[str], idx: int) -> float:
    if ordered_names[idx] in positions and "heading_rad" in positions[ordered_names[idx]]:
        return float(positions[ordered_names[idx]]["heading_rad"])

    here = positions.get(ordered_names[idx])
    if not here:
        return 0.0
    for offset in (1, -1):
        j = idx + offset
        if not 0 <= j < len(ordered_names):
            continue
        there = positions.get(ordered_names[j])
        if not there:
            continue
        dx = float(there.get("x_meters", 0.0)) - float(here.get("x_meters", 0.0))
        dy = float(there.get("y_meters", 0.0)) - float(here.get("y_meters", 0.0))
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return math.atan2(dy, dx)
    return 0.0


def load_selection(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        with open(path) as f:
            data = json.load(f)
        values: list[str] = []

        def collect(node: object) -> None:
            if isinstance(node, list):
                for item in node:
                    collect(item)
            elif isinstance(node, dict):
                for key in ("filename", "source_frame", "frame", "frame_name", "image", "image_name"):
                    value = node.get(key)
                    if isinstance(value, str):
                        values.append(value)
                frames = node.get("frames")
                if isinstance(frames, dict):
                    values.extend(str(name) for name in frames.keys())
                elif isinstance(frames, list):
                    collect(frames)
                for key in ("selected_frames", "selected_frame_names", "selected"):
                    selected = node.get(key)
                    if isinstance(selected, list):
                        collect(selected)
            elif isinstance(node, str):
                values.append(node)

        collect(data)
        return list(dict.fromkeys(values))

    values: list[str] = []
    with open(path) as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            values.append(value)
    return values


def select_frames(
    frames_dir: Path,
    positions: dict[str, dict],
    selection_list: Path | None,
    max_frames: int | None,
    stride: int,
) -> list[Path]:
    frames = sorted(
        frames_dir.glob("*.jpg"),
        key=lambda p: (frame_timestamp(p, positions), p.name),
    )
    if not frames:
        raise FileNotFoundError(f"No JPEG frames found in {frames_dir}")

    if selection_list:
        requested = load_selection(selection_list)
        by_name = {p.name: p for p in frames}
        by_stem = {p.stem: p for p in frames}
        selected: list[Path] = []
        missing: list[str] = []
        for item in requested:
            if item in by_name:
                selected.append(by_name[item])
                continue
            if item in by_stem:
                selected.append(by_stem[item])
                continue
            try:
                target_t = float(item.rstrip("s"))
            except ValueError:
                missing.append(item)
                continue
            nearest = min(frames, key=lambda p: abs(frame_timestamp(p, positions) - target_t))
            selected.append(nearest)
        if missing:
            print(f"Warning: {len(missing)} selection entries did not match a frame: {missing[:5]}")
        frames = list(dict.fromkeys(selected))

    stride = max(1, stride)
    frames = frames[::stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def build_remap(
    source_w: int,
    source_h: int,
    out_w: int,
    out_h: int,
    hfov_deg: float,
    heading_rad: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    focal_px = out_w / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    u_out = np.arange(out_w, dtype=np.float64)
    v_out = np.arange(out_h, dtype=np.float64)
    uu, vv = np.meshgrid(u_out, v_out)

    x_c = (uu - out_w / 2.0) / focal_px
    y_c = -(vv - out_h / 2.0) / focal_px
    z_c = np.ones_like(x_c)

    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    dx = x_c * sin_h + z_c * cos_h
    dy = x_c * (-cos_h) + z_c * sin_h
    dz = y_c

    theta = np.arctan2(dy, dx)
    phi = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    map_x = ((theta + math.pi) / (2 * math.pi) * source_w).astype(np.float32)
    map_y = ((math.pi / 2 - phi) / math.pi * source_h).astype(np.float32)
    return map_x, map_y, focal_px


def crop_frame(src_path: Path, out_path: Path, heading_rad: float, out_w: int, out_h: int, hfov_deg: float) -> float:
    img = cv2.imread(str(src_path))
    if img is None:
        raise ValueError(f"Could not read image {src_path}")
    source_h, source_w = img.shape[:2]
    map_x, map_y, focal_px = build_remap(source_w, source_h, out_w, out_h, hfov_deg, heading_rad)
    crop = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return focal_px


def run_colmap(args: list[str], log_path: Path) -> None:
    printable = " ".join(args)
    print(f"Running: {printable}")
    with open(log_path, "a") as log:
        log.write(f"\n$ {printable}\n")
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"COLMAP command failed with exit code {result.returncode}: {printable}")


def write_manifest(path: Path, manifest: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def prepare_output(out_dir: Path, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    images_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse"
    db_path = out_dir / "database.db"
    ply_path = out_dir / "reconstruction.ply"

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, sparse_dir, db_path, ply_path


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct sparse 3D geometry from 360 frames with COLMAP.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--positions-json", type=Path, default=DEFAULT_POSITIONS_JSON)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selection-list", type=Path, help="Text file of selected frame filenames, stems, or timestamps.")
    parser.add_argument("--max-frames", type=int, help="Use only the first N selected/available frames.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth selected/available frame.")
    parser.add_argument("--hfov-deg", type=float, default=DEFAULT_HFOV_DEG)
    parser.add_argument("--width", type=int, default=DEFAULT_OUT_W)
    parser.add_argument("--height", type=int, default=DEFAULT_OUT_H)
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--use-gpu", action="store_true", help="Use COLMAP GPU SIFT extraction/matching.")
    parser.add_argument("--skip-colmap", action="store_true", help="Only generate perspective crops and manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory before running.")
    args = parser.parse_args()

    args.frames_dir = repo_path(args.frames_dir)
    args.positions_json = repo_path(args.positions_json)
    args.out_dir = repo_path(args.out_dir)
    if args.selection_list is not None:
        args.selection_list = repo_path(args.selection_list)

    positions = load_positions(args.positions_json)
    selected = select_frames(args.frames_dir, positions, args.selection_list, args.max_frames, args.stride)
    selected_names = {p.name for p in selected}
    heading_names = sorted(
        [name for name in positions if (args.frames_dir / name).exists() or name in selected_names],
        key=lambda n: frame_timestamp(args.frames_dir / n, positions),
    )

    images_dir, sparse_dir, db_path, ply_path = prepare_output(args.out_dir, args.overwrite)
    log_path = args.out_dir / "colmap_run.log"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    manifest: list[dict] = []
    focal_px = args.width / (2.0 * math.tan(math.radians(args.hfov_deg / 2.0)))
    for i, src in enumerate(selected, 1):
        if positions and src.name in heading_names:
            heading = heading_from_positions(positions, heading_names, heading_names.index(src.name))
        else:
            heading = 0.0
        out_name = src.name
        out_path = images_dir / out_name
        focal_px = crop_frame(src, out_path, heading, args.width, args.height, args.hfov_deg)
        pos = positions.get(src.name, {})
        manifest.append(
            {
                "source_frame": src.name,
                "crop_image": out_name,
                "timestamp_sec": frame_timestamp(src, positions),
                "heading_rad": heading,
                "hfov_deg": args.hfov_deg,
                "width": args.width,
                "height": args.height,
                "focal_px": focal_px,
                "x_meters": pos.get("x_meters"),
                "y_meters": pos.get("y_meters"),
                "lat": pos.get("lat"),
                "lon": pos.get("lon"),
            }
        )
        print(f"[{i:>3}/{len(selected)}] cropped {src.name} -> {out_path.relative_to(ROOT)}")

    write_manifest(args.out_dir / "frame_manifest.json", manifest)
    print(f"Wrote {len(manifest)} crops to {images_dir.relative_to(ROOT)}")

    if args.skip_colmap:
        print("Skipping COLMAP run because --skip-colmap was provided.")
        return

    if not shutil.which(args.colmap_bin):
        raise SystemExit(
            f"ERROR: '{args.colmap_bin}' was not found on PATH. "
            f"Crops are ready in {images_dir.relative_to(ROOT)}; install COLMAP and rerun without --skip-colmap."
        )

    camera_params = f"{focal_px},{focal_px},{args.width / 2.0},{args.height / 2.0}"
    run_colmap(
        [
            args.colmap_bin,
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            camera_params,
            "--SiftExtraction.use_gpu",
            "1" if args.use_gpu else "0",
        ],
        log_path,
    )
    matcher_cmd = "sequential_matcher" if args.matcher == "sequential" else "exhaustive_matcher"
    run_colmap(
        [
            args.colmap_bin,
            matcher_cmd,
            "--database_path",
            str(db_path),
            "--SiftMatching.use_gpu",
            "1" if args.use_gpu else "0",
        ],
        log_path,
    )
    run_colmap(
        [
            args.colmap_bin,
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        log_path,
    )

    sparse_models = sorted(
        (p for p in sparse_dir.iterdir() if p.is_dir()),
        key=lambda p: (p / "points3D.bin").stat().st_size if (p / "points3D.bin").exists() else 0,
        reverse=True,
    )
    if not sparse_models:
        raise RuntimeError(f"COLMAP mapper completed but no sparse model was written under {sparse_dir}")
    model_dir = sparse_models[0]
    run_colmap(
        [
            args.colmap_bin,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(ply_path),
            "--output_type",
            "PLY",
        ],
        log_path,
    )
    print(f"Sparse model: {model_dir.relative_to(ROOT)}")
    print(f"PLY point cloud: {ply_path.relative_to(ROOT)}")
    print(f"COLMAP log: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
