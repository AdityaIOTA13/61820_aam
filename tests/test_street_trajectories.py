"""Unit tests for street routing (no OSM download)."""

import numpy as np
import networkx as nx

from adaptive_scanning.street_trajectories import (
    affine_map_points,
    annotate_playback_events_si_timeline,
    build_single_leg_trajectory,
    build_street_trajectory,
    inverse_affine_world_to_graph,
    resample_polyline_at_speed,
)


def _toy_street_graph() -> nx.MultiDiGraph:
    """Straight line of walkable edges in projected 'metres'."""
    G = nx.MultiDiGraph()
    n = 25
    for i in range(n):
        G.add_node(i, x=float(i * 25.0), y=0.0)
    for i in range(n - 1):
        G.add_edge(i, i + 1, length=25.0)
        G.add_edge(i + 1, i, length=25.0)
    return G


def test_annotate_playback_si_timeline_moving_cumulative():
    ev = [
        {"phase": "morning_home", "t_start_s": 0.0, "t_end_s": 100.0, "day_index": 0},
        {
            "phase": "travel",
            "t_start_s": 100.0,
            "t_end_s": 200.0,
            "day_index": 0,
            "arc_length_m": 135.0,
        },
    ]
    annotate_playback_events_si_timeline(ev, walk_speed_m_s=1.35)
    assert ev[0]["t_moving_cumulative_at_end_s"] == 0.0
    assert ev[1]["t_moving_cumulative_at_start_s"] == 0.0
    assert ev[1]["t_moving_cumulative_at_end_s"] == 100.0
    assert abs(float(ev[1]["effective_speed_m_s"]) - 1.35) < 1e-6


def test_inverse_affine_roundtrips_graph_points():
    xy = np.array([[0.0, 0.0], [100.0, 50.0], [200.0, 0.0]], dtype=np.float64)
    w = 128.0
    h = 128.0
    mapped = affine_map_points(xy, margin=1.0, world_w_m=w, world_h_m=h)
    back = inverse_affine_world_to_graph(
        mapped, xy, world_w_m=w, world_h_m=h, margin=1.0
    )
    assert back.shape == xy.shape
    assert np.allclose(back, xy, atol=1e-5, rtol=0.0)


def test_affine_map_aspect_preserves_shape():
    xy = np.array([[0.0, 0.0], [100.0, 50.0], [200.0, 0.0]], dtype=np.float64)
    out = affine_map_points(xy, margin=1.0, world_w_m=128.0, world_h_m=128.0)
    assert out.shape == xy.shape
    assert np.all(out[:, 0] >= 0) and np.all(out[:, 0] <= 128)
    assert np.all(out[:, 1] >= 0) and np.all(out[:, 1] <= 128)


def test_build_street_trajectory_toy_graph():
    rng = np.random.default_rng(0)
    G = _toy_street_graph()
    x, y, h = build_street_trajectory(
        G,
        rng,
        world_w_m=120.0,
        world_h_m=100.0,
        n_steps=200,
        speed_m_s=1.3,
        dt_s=5.0,
        margin_m=2.0,
        n_anchors=8,
        anchor_reuse_bias=0.9,
    )
    assert len(x) == 201
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(h))


def test_build_single_leg_toy_graph():
    rng = np.random.default_rng(1)
    G = _toy_street_graph()
    x, y, h, poly_m = build_single_leg_trajectory(
        G,
        rng,
        world_w_m=120.0,
        world_h_m=100.0,
        n_steps=100,
        speed_m_s=1.2,
        dt_s=5.0,
        n_anchors=6,
    )
    assert len(x) == 101
    assert poly_m.shape[1] == 2
    assert np.ptp(x) > 1.0 or np.ptp(y) > 1.0


def test_resample_polyline_repeats():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    x, y, h = resample_polyline_at_speed(xy, speed_m_s=1.0, dt_s=1.0, n_out=50)
    assert len(x) == 51
    assert float(np.ptp(x)) > 1.0
    assert np.all(np.isfinite(h))


def test_resample_polyline_repeat_path_backward_jump():
    """Long horizon with repeat joins copies: last vertex → first of next copy (not collinear tail)."""
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    x, y, h = resample_polyline_at_speed(
        xy, speed_m_s=1.0, dt_s=1.0, n_out=120, repeat_path=True
    )
    assert np.any(np.diff(x) < -0.5)


def test_resample_polyline_no_repeat_one_way():
    """No repeat: position stays forward along the segment (no return chord)."""
    xy = np.array([[0.0, 0.0], [50.0, 0.0]], dtype=np.float64)
    x, y, h = resample_polyline_at_speed(
        xy, speed_m_s=1.0, dt_s=1.0, n_out=200, repeat_path=False
    )
    assert np.all(np.diff(x) >= -1e-6)
    assert float(x[-1]) >= 49.0
    assert np.allclose(y, 0.0)
