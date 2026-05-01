"""Unit tests for courtyard_geom (no real dataset required)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import courtyard_geom as cg


def test_camera_heading_from_trajectory_east():
    xy = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    h = cg.camera_heading_from_trajectory(xy)
    assert len(h) == 3
    assert abs(h[0] - 0.0) < 1e-6  # duplicate point fallback
    assert abs(h[1] - 0.0) < 1e-6  # atan2(0,1)=0
    assert abs(h[2] - 0.0) < 1e-6


def test_camera_heading_from_trajectory_north():
    xy = [(0.0, 0.0), (0.0, 1.0)]
    h = cg.camera_heading_from_trajectory(xy)
    assert abs(h[1] - math.pi / 2) < 1e-6


def test_combined_scale_finite():
    rng = np.random.default_rng(0)
    depth_norm = rng.uniform(0.1, 0.9, size=(256, 512)).astype(np.float32)
    # Bottom-heavy floor cue
    depth_norm[-40:, :] = 0.95
    s = cg.combined_scale(depth_norm, 18.0, 16.0, 0.1, 512, 256)
    assert math.isfinite(s)
    assert 0.5 < s < 200.0


def test_parse_svg_waypoints_count():
    svg = ROOT / "data" / "svg_path.svg"
    pts = cg.parse_svg_waypoints(svg)
    assert len(pts) == 14
    assert abs(pts[0][0] - 373.5) < 1e-6


def test_ground_plane_hits_returns_points():
    rng = np.random.default_rng(1)
    H, W = 320, 640
    depth_norm = rng.uniform(0.15, 0.85, size=(H, W)).astype(np.float32)
    depth_norm[int(0.65 * H) :, :] = np.maximum(depth_norm[int(0.65 * H) :, :], 0.92)

    gx, gy = cg.ground_plane_hits(
        depth_norm, cam_x=18.0, cam_y=16.0, cam_heading=0.0, subsample=10
    )
    assert gx.size == gy.size
    assert gx.size > 0
    assert np.isfinite(gx).all() and np.isfinite(gy).all()
