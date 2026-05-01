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
