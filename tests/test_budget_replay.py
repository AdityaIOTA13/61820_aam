"""Budget replay frame selection (no depth I/O)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import importlib.util


def _load_pc():
    spec = importlib.util.spec_from_file_location(
        "project_coverage", SCRIPTS / "project_coverage.py"
    )
    assert spec and spec.loader
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    return pc


def _fake_rows(n: int = 20):
    return [
        (f"f{i:04d}.jpg", 1.0, 2.0, float(i) * 0.5) for i in range(n)
    ]


def test_select_budget_all_when_budget_covers_span():
    pc = _load_pc()
    rows = _fake_rows(10)
    span = rows[-1][3] - rows[0][3]
    sel, info = pc.select_frames_for_budget(rows, span + 10.0, "random", seed=0)
    assert len(sel) == len(rows)
    assert "note" in info or info.get("k_target") == 10


def test_prefix_smaller_than_full():
    pc = _load_pc()
    rows = _fake_rows(30)
    sel, info = pc.select_frames_for_budget(rows, 4.0, "prefix", seed=0)
    assert len(sel) < len(rows)
    assert len(sel) == info["k_target"]
    assert sel[0] == rows[0][0]


def test_random_reproducible():
    pc = _load_pc()
    rows = _fake_rows(40)
    a, _ = pc.select_frames_for_budget(rows, 5.0, "random", seed=42)
    b, _ = pc.select_frames_for_budget(rows, 5.0, "random", seed=42)
    assert a == b


def test_random_differs_across_seeds():
    pc = _load_pc()
    rows = _fake_rows(60)
    a, _ = pc.select_frames_for_budget(rows, 8.0, "random", seed=1)
    b, _ = pc.select_frames_for_budget(rows, 8.0, "random", seed=2)
    assert a != b


def test_heading_by_name_matches_trajectory_length():
    pc = _load_pc()
    rows = _fake_rows(5)
    h = pc.heading_by_name_from_rows(rows)
    assert len(h) == 5
    for name in [r[0] for r in rows]:
        assert name in h


def test_neighborhood_mean_age_disk_high_when_unseen():
    pc = _load_pc()
    last = np.full((4, 4), -np.inf, dtype=np.float64)
    m = pc.neighborhood_mean_age_disk(
        last,
        10.0,
        cx=2.0,
        cy=2.0,
        x0=0.0,
        y0=0.0,
        resolution_m=1.0,
        nx=4,
        ny=4,
        radius_m=3.0,
        prior_unseen_age_s=1e6,
    )
    assert m > 1e5


def test_floor_grid_layout_positive_cells():
    pc = _load_pc()
    x0, y0, nx, ny = pc.floor_grid_layout(0.5, 2.0)
    assert nx >= 1 and ny >= 1
    assert x0 == -2.0 and y0 == -2.0


def test_greedy_location_needs_coverage_foot():
    pc = _load_pc()
    last = np.full((4, 4), -np.inf, dtype=np.float64)
    assert pc.greedy_location_needs_coverage(
        last,
        0.5,
        0.5,
        x0=0.0,
        y0=0.0,
        resolution_m=1.0,
        nx=4,
        ny=4,
        unseen_scope="foot",
        radius_m=2.0,
    )
    last[0, 0] = 1.0
    assert not pc.greedy_location_needs_coverage(
        last,
        0.5,
        0.5,
        x0=0.0,
        y0=0.0,
        resolution_m=1.0,
        nx=4,
        ny=4,
        unseen_scope="foot",
        radius_m=2.0,
    )


def test_wrap_angle_rad():
    pc = _load_pc()
    assert abs(pc._wrap_angle_rad(0.0)) < 1e-9
    wpi = pc._wrap_angle_rad(math.pi)
    assert abs(abs(wpi) - math.pi) < 1e-9
    w = pc._wrap_angle_rad(3 * math.pi)
    assert abs(w + math.pi) < 1e-9 or abs(w - math.pi) < 1e-9


def test_greedy_location_stale_foot_only_when_seen():
    pc = _load_pc()
    last = np.full((3, 3), -np.inf, dtype=np.float64)
    assert not pc.greedy_location_stale(
        last,
        100.0,
        0.5,
        0.5,
        x0=0.0,
        y0=0.0,
        resolution_m=1.0,
        nx=3,
        ny=3,
        unseen_scope="foot",
        radius_m=2.0,
        stale_threshold_s=10.0,
    )
    last[0, 0] = 50.0
    assert pc.greedy_location_stale(
        last,
        100.0,
        0.5,
        0.5,
        x0=0.0,
        y0=0.0,
        resolution_m=1.0,
        nx=3,
        ny=3,
        unseen_scope="foot",
        radius_m=2.0,
        stale_threshold_s=10.0,
    )


def test_greedy_location_needs_coverage_disk():
    pc = _load_pc()
    last = np.full((5, 5), -np.inf, dtype=np.float64)
    last[2, 2] = 1.0
    assert pc.greedy_location_needs_coverage(
        last,
        2.5,
        2.5,
        x0=0.0,
        y0=0.0,
        resolution_m=1.0,
        nx=5,
        ny=5,
        unseen_scope="disk",
        radius_m=2.5,
    )
    last[:, :] = 1.0
    assert not pc.greedy_location_needs_coverage(
        last,
        2.5,
        2.5,
        x0=0.0,
        y0=0.0,
        resolution_m=1.0,
        nx=5,
        ny=5,
        unseen_scope="disk",
        radius_m=2.5,
    )
