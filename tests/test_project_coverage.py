"""Integration test for map-age grid (synthetic frames in a temp dir)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

def _synthetic_depth_png(path: Path, h: int = 320, w: int = 640) -> None:
    rng = np.random.default_rng(42)
    d = rng.uniform(0.2, 0.7, size=(h, w)).astype(np.float32)
    d[int(0.6 * h) :, :] = np.maximum(d[int(0.6 * h) :, :], 0.9)
    img = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def test_run_map_age_updates_last_seen(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "project_coverage", SCRIPTS / "project_coverage.py"
    )
    assert spec and spec.loader
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)

    frames = tmp_path / "frames"
    depths = tmp_path / "depths"
    frames.mkdir()
    depths.mkdir()

    # Two colour JPEGs
    for name in ("frame_00000_0.000s.jpg", "frame_00001_0.501s.jpg"):
        Image.new("RGB", (64, 64), color=(128, 64, 32)).save(frames / name)
        stem = Path(name).stem
        _synthetic_depth_png(depths / f"{stem}_depth.png")

    positions = {
        "frame_00000_0.000s.jpg": {
            "x_meters": 18.0,
            "y_meters": 16.0,
            "timestamp_sec": 0.0,
        },
        "frame_00001_0.501s.jpg": {
            "x_meters": 18.5,
            "y_meters": 16.0,
            "timestamp_sec": 0.501,
        },
    }
    monkeypatch.setattr(pc, "FRAMES_DIR", frames)
    monkeypatch.setattr(pc, "DEPTH_DIR", depths)

    last_seen, ever, t_end, meta = pc.run_map_age(
        positions,
        list(positions.keys()),
        resolution_m=0.5,
        subsample=12,
        margin_m=2.0,
    )

    assert t_end == 0.501
    assert meta["frames_used"] == 2
    assert meta["rays_accumulated"] > 0
    assert ever.any()
    assert np.nanmax(last_seen[ever]) <= t_end + 1e-6
    # After walk, some cell should have low age (recently seen)
    age = np.full_like(last_seen, np.nan)
    m = last_seen > -np.inf
    age[m] = t_end - last_seen[m]
    assert np.nanmin(age[m]) < 1.0


def test_select_frames_greedy_local_respects_budget(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "project_coverage", SCRIPTS / "project_coverage.py"
    )
    assert spec and spec.loader
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)

    frames = tmp_path / "frames"
    depths = tmp_path / "depth_maps"
    frames.mkdir()
    depths.mkdir()

    positions: dict = {}
    for i in range(12):
        name = f"frame_{i:05d}_{i * 0.5:.3f}s.jpg"
        Image.new("RGB", (64, 64), color=(60 + i * 5, 90, 120)).save(frames / name)
        stem = Path(name).stem
        _synthetic_depth_png(depths / f"{stem}_depth.png")
        positions[name] = {
            "x_meters": 10.0 + float(i) * 0.4,
            "y_meters": 12.0 + float(i) * 0.05,
            "timestamp_sec": float(i) * 0.5,
        }

    monkeypatch.setattr(pc, "FRAMES_DIR", frames)
    monkeypatch.setattr(pc, "DEPTH_DIR", depths)

    names = list(positions.keys())
    names.sort(key=lambda n: positions[n]["timestamp_sec"])
    full_rows = pc.eligible_timeline_rows(positions, names)
    heading_map = pc.heading_by_name_from_rows(full_rows)

    sel, info, _ls = pc.select_frames_greedy_local(
        full_rows,
        2.0,
        resolution_m=0.5,
        margin_m=2.0,
        subsample=16,
        hfov_deg=120.0,
        vfov_deg=90.0,
        projection="ground_plane",
        heading_by_name=heading_map,
        greedy_trigger="mean_age",
        greedy_unseen_scope="foot",
        greedy_radius_m=6.0,
        greedy_mean_age_threshold_s=4.0,
        greedy_hysteresis=False,
    )

    assert info["policy"] == "greedy_local"
    assert len(sel) <= info["k_target"]
    assert len(sel) >= 1
    ts = [positions[n]["timestamp_sec"] for n in sel]
    assert ts == sorted(ts)


def test_lerp_staleness_age_grids_for_viz() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "project_coverage", SCRIPTS / "project_coverage.py"
    )
    assert spec and spec.loader
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)

    a0 = np.array([[np.nan, 10.0], [10.0, np.nan]], dtype=np.float64)
    a1 = np.array([[0.0, 20.0], [20.0, 5.0]], dtype=np.float64)
    m = pc.lerp_staleness_age_grids_for_viz(a0, a1, 0.5)
    # was unseen at t0, seen at t1 → ramps in with alpha
    assert abs(float(m[0, 0]) - 0.0) < 1e-9
    assert abs(float(m[0, 1]) - 15.0) < 1e-9
    assert abs(float(m[1, 0]) - 15.0) < 1e-9
    assert abs(float(m[1, 1]) - 2.5) < 1e-9
