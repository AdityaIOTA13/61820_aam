import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adaptive_scanning.config import AdaptiveScanningConfig, config_from_saved_dict
from adaptive_scanning.env import CameraBudgetEnv, daily_video_budget_seconds
from adaptive_scanning.policies import (
    AlwaysOffPolicy,
    AlwaysOnPolicy,
    BudgetAwareGreedyUnseenOnlyPolicy,
    RandomPolicy,
)
from adaptive_scanning.rollout import run_episode
from adaptive_scanning.data_export import generate_synthetic_episode_batch


def test_config_from_saved_dict_strips_unknown_and_fills_defaults():
    c = config_from_saved_dict(
        {"nx": 10, "obsolete_field_xyz": 123, "motion_mode": "box", "osm_bbox": [-71.0, 42.0, -70.0, 43.0]}
    )
    assert c.nx == 10
    assert c.motion_mode == "box"
    assert c.osm_bbox == (-71.0, 42.0, -70.0, 43.0)
    assert c.evening_lenient_after_s_since_day_start == AdaptiveScanningConfig.evening_lenient_after_s_since_day_start


def _tiny_cfg():
    return AdaptiveScanningConfig(
        nx=12,
        ny=12,
        resolution_m=2.0,
        max_sim_time_s=600.0,
        day_duration_s=300.0,
        seconds_video_budget_per_day=40.0,
        dt_s=10.0,
        patch_cells=7,
        hfov_deg=100.0,
        scan_radius_m=15.0,
        motion_mode="box",
        osm_daily_home_commute=False,
        osm_bbox=None,
        osm_place="",
        w_unused_budget_end_of_day=0.0,
    )


def test_env_steps_and_budget_reset():
    env = CameraBudgetEnv(_tiny_cfg(), seed=1)
    obs, info = env.reset(seed=42)
    assert obs.shape[0] == env.observation_dim
    assert info["uncovered_fraction"] == 1.0
    # Burn budget then wait for day rollover
    budget_hits = 0
    for _ in range(200):
        st = env.step(1)
        if st.info.get("camera_on_effective"):
            budget_hits += 1
        if st.truncated:
            break
    assert st.truncated
    assert budget_hits > 0


def test_foot_cell_observation_mode_shape_and_signal():
    cfg = replace(_tiny_cfg(), observation_mode="foot_cell")
    env = CameraBudgetEnv(cfg, seed=0)
    obs, _info = env.reset(seed=0)
    assert obs.shape[0] == env.observation_dim == 14
    # First 7 dims are foot-cell features; with never-scanned start, "ever" flags are zero.
    assert float(obs[0]) == pytest.approx(0.0)
    assert float(obs[2]) == pytest.approx(0.0)
    assert float(obs[6]) == pytest.approx(1.0)


def test_greedy_unseen_wedge_revisit_vs_same_pass():
    """Recent wedge on foot cell does not suppress; old wedge (multi-step / revisit) does; foot + grace."""
    cfg = _tiny_cfg()
    cfg.greedy_unseen_coverage_grace_seconds = 2.0
    cfg.greedy_unseen_wedge_suppress_min_age_s = 80.0
    env = CameraBudgetEnv(cfg, seed=0)
    env.reset(seed=0)
    ax0 = float(env._traj_x[env._step_idx])
    ay0 = float(env._traj_y[env._step_idx])
    ix = int(ax0 // cfg.resolution_m)
    iy = int(ay0 // cfg.resolution_m)
    # Same-pass wedge (~dt): effective min is max(80, 2.5*dt)=80 with dt=10
    env._sim_time_s = 100.0
    env.last_seen[iy, ix] = 95.0
    info_young_wedge = env._info_dict()
    assert info_young_wedge["foot_cell_never_scanned_for_policy"] is True
    assert info_young_wedge["foot_cell_greedy_unseen_on"] is True

    env.last_seen[iy, ix] = 0.0
    env._sim_time_s = 500.0
    info_old_wedge = env._info_dict()
    assert info_old_wedge["foot_cell_greedy_unseen_on"] is False

    env.last_seen[iy, ix] = np.nan
    env._last_seen_foot[iy, ix] = 10.0
    env._sim_time_s = 500.0
    assert env._info_dict()["foot_cell_greedy_unseen_on"] is False

    env._last_seen_foot[iy, ix] = 498.0
    info_fresh_foot = env._info_dict()
    assert info_fresh_foot["foot_cell_greedy_unseen_on"] is True

    pol = BudgetAwareGreedyUnseenOnlyPolicy(min_budget_frac_to_turn_on=0.0)
    obs = env._observation()
    assert pol.act(obs, info_young_wedge) == 1
    assert pol.act(obs, info_fresh_foot) == 1


def test_evening_lenient_greedy_unseen_reopens_mildly_stale_wedge():
    """Late in simulated day, wedge suppress threshold rises so moderate staleness still scans."""
    base = _tiny_cfg()
    base.greedy_unseen_coverage_grace_seconds = 2.0
    base.greedy_unseen_wedge_suppress_min_age_s = 80.0

    cfg_no = replace(base, evening_lenient_after_s_since_day_start=-1.0)
    env_no = CameraBudgetEnv(cfg_no, seed=0)
    env_no.reset(seed=0)
    ax0 = float(env_no._traj_x[env_no._step_idx])
    ay0 = float(env_no._traj_y[env_no._step_idx])
    ix = int(ax0 // base.resolution_m)
    iy = int(ay0 // base.resolution_m)
    env_no._day_start_s = 0.0
    env_no._sim_time_s = 290.0
    # Effective min age is max(80, 10*dt)=100; need age clearly above that when leniency is off.
    env_no.last_seen[iy, ix] = 290.0 - 150.0
    assert env_no._info_dict()["foot_cell_greedy_unseen_on"] is False

    cfg_yes = replace(
        base,
        evening_lenient_after_s_since_day_start=200.0,
        evening_lenient_ramp_s=50.0,
        evening_lenient_wedge_suppress_age_mult=2.25,
        evening_lenient_foot_grace_mult=4.0,
        evening_lenient_min_budget_frac=0.05,
    )
    env_yes = CameraBudgetEnv(cfg_yes, seed=0)
    env_yes.reset(seed=0)
    env_yes._day_start_s = 0.0
    env_yes._sim_time_s = 290.0
    env_yes.last_seen[iy, ix] = 140.0
    assert env_yes._info_dict()["greedy_unseen_evening_leniency"] > 0.7
    assert env_yes._info_dict()["foot_cell_greedy_unseen_on"] is True


def test_info_includes_day_timing_and_foot_cell_ages():
    env = CameraBudgetEnv(_tiny_cfg(), seed=0)
    env.reset(seed=0)
    info = env._info_dict()
    assert "seconds_since_day_start" in info
    assert "day_duration_s" in info
    assert float(info["day_duration_s"]) == pytest.approx(300.0)
    assert "foot_cell_wedge_age_s" in info
    assert "foot_cell_foot_age_s" in info


def test_progressive_rescan_policy_runs():
    from adaptive_scanning.policies import BudgetAwareGreedyUnseenProgressiveRescanPolicy

    cfg = _tiny_cfg()
    cfg.observation_mode = "foot_cell"
    st = run_episode(CameraBudgetEnv(cfg, seed=0), BudgetAwareGreedyUnseenProgressiveRescanPolicy(), seed=1)
    assert st.steps > 0


def test_stationary_interval_does_not_scan_or_spend_budget():
    env = CameraBudgetEnv(_tiny_cfg(), seed=1)
    _obs, info = env.reset(seed=42)
    start_budget = float(info["budget_s"])
    env._traj_x = np.array([5.0, 5.0], dtype=np.float64)
    env._traj_y = np.array([7.0, 7.0], dtype=np.float64)
    env._traj_h = np.array([0.0, 0.0], dtype=np.float64)
    env._step_idx = 0

    st = env.step(1)
    assert bool(st.info["step_interval_is_moving"]) is False
    assert bool(st.info["camera_on_effective"]) is False
    assert float(st.info["budget_s"]) == pytest.approx(start_budget)


def test_always_on_beats_always_off_on_coverage():
    cfg = _tiny_cfg()
    on_stats = run_episode(CameraBudgetEnv(cfg, seed=0), AlwaysOnPolicy(), seed=1)
    off_stats = run_episode(CameraBudgetEnv(cfg, seed=0), AlwaysOffPolicy(), seed=1)
    assert on_stats.final_uncovered_fraction < off_stats.final_uncovered_fraction


def test_random_policy_runs():
    cfg = _tiny_cfg()
    rng = np.random.default_rng(0)
    st = run_episode(CameraBudgetEnv(cfg, seed=0), RandomPolicy(rng), seed=2)
    assert st.steps > 0


def test_export_batch_smoke():
    from adaptive_scanning.data_export import save_episode_npz

    cfg = _tiny_cfg()
    batch = generate_synthetic_episode_batch(cfg=cfg, n_episodes=2, seed=0)
    assert len(batch["episodes"]) == 2
    assert batch["episodes"][0]["actions"].shape[0] == batch["episodes"][0]["rewards"].shape[0]
    assert batch["episodes"][0]["trajectory_source"] == "box"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "batch.npz"
        save_episode_npz(p, batch)
        with np.load(p, allow_pickle=True) as z:
            assert "ep0_trajectory_source" in z.files
            assert str(z["ep0_trajectory_source"][0]) == "box"


def test_visualize_episode_writes_png():
    from adaptive_scanning.viz import visualize_episode

    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "preview.png"
        path, _src, _bm, _cov_pack, _pb, _dp = visualize_episode(
            cfg, policy_name="always_on", seed=3, out_path=out
        )
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 500


def test_train_reinforce_writes_log_dir():
    from adaptive_scanning.training import train_reinforce

    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as td:
        logd = Path(td) / "logs"
        pol, res = train_reinforce(
            cfg,
            epochs=2,
            episodes_per_epoch=2,
            lr=1e-2,
            seed=0,
            show_progress=False,
            log_dir=logd,
        )
        assert len(res.history) == 2
        assert (logd / "config.json").is_file()
        assert (logd / "metrics.jsonl").is_file()
        assert (logd / "history.json").is_file()
        assert (logd / "training_curves.png").is_file()
        assert pol.net[0].in_features == CameraBudgetEnv(cfg, seed=0).observation_dim


def test_visualize_episode_skip_png_no_panel_file():
    from adaptive_scanning.viz import visualize_episode

    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "maps_only.png"
        path, _src, _bm, _cov_pack, _pb, _dp = visualize_episode(
            cfg, policy_name="always_on", seed=3, out_path=out, skip_episode_png=True
        )
        assert path is None
        assert not out.exists()


def test_daily_video_budget_scales_inversely_with_walk_speed():
    base = AdaptiveScanningConfig(
        seconds_video_budget_per_day=300.0,
        video_budget_reference_walk_speed_m_s=1.35,
        walk_speed_m_s=2.70,
    )
    assert daily_video_budget_seconds(base) == pytest.approx(150.0)
    off = AdaptiveScanningConfig(
        seconds_video_budget_per_day=300.0,
        video_budget_reference_walk_speed_m_s=0.0,
        walk_speed_m_s=2.70,
    )
    assert daily_video_budget_seconds(off) == pytest.approx(300.0)


def test_env_info_reports_effective_daily_video_budget():
    cfg = AdaptiveScanningConfig(
        nx=8,
        ny=8,
        resolution_m=2.0,
        max_sim_time_s=100.0,
        day_duration_s=100.0,
        seconds_video_budget_per_day=300.0,
        video_budget_reference_walk_speed_m_s=1.35,
        walk_speed_m_s=2.70,
        dt_s=10.0,
        patch_cells=5,
        motion_mode="box",
        osm_daily_home_commute=False,
        osm_bbox=None,
        osm_place="",
    )
    env = CameraBudgetEnv(cfg, seed=0)
    _obs, info = env.reset(seed=0)
    assert float(info["budget_s"]) == pytest.approx(150.0)
    assert float(info["seconds_video_budget_per_day_effective"]) == pytest.approx(150.0)


def test_unused_budget_penalty_at_simulated_day_end():
    cfg = AdaptiveScanningConfig(
        nx=8,
        ny=8,
        resolution_m=2.0,
        max_sim_time_s=500.0,
        day_duration_s=100.0,
        seconds_video_budget_per_day=50.0,
        dt_s=10.0,
        patch_cells=5,
        w_unused_budget_end_of_day=1.0,
        reward_camera_on_bonus=0.0,
        motion_mode="box",
        osm_daily_home_commute=False,
        osm_bbox=None,
        osm_place="",
    )
    env = CameraBudgetEnv(cfg, seed=0)
    env.reset(seed=0)
    last_pen = 0.0
    for _ in range(10):
        st = env.step(0)
        last_pen = float(st.info.get("end_of_day_unused_budget_penalty", 0.0))
    assert last_pen == pytest.approx(-1.0)


def test_segment_day_indices_playback_per_day_si_bounds():
    """Regression: day assignment uses merged SI bounds per day, not fragile per-event containment."""
    from adaptive_scanning.viz import _segment_day_indices_playback

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    nseg = 10
    sim_time_s = np.array([(i + 1) * dt for i in range(nseg)], dtype=np.float64)
    rec = {
        "cfg": cfg,
        "sim_time_s": sim_time_s,
        "playback": {
            "events": [
                {"day_index": 0, "t_start_s": 0.0, "t_end_s": 4.5 * dt, "phase": "morning_home"},
                {"day_index": 1, "t_start_s": 5.0 * dt, "t_end_s": 9.5 * dt, "phase": "travel"},
            ]
        },
    }
    di = _segment_day_indices_playback(rec, nseg)
    assert di is not None
    assert int(di[0]) == 0
    assert int(di[4]) == 0
    assert int(di[5]) == 1
    assert int(di[9]) == 1


def test_moving_segment_day_indices_follow_travel_time():
    """Day labels use SI wall overlap sim[k]-dt..sim[k] with travel [t_start,t_end], not k*dt as moving time."""
    from adaptive_scanning.viz import _moving_segment_day_indices_playback

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    nseg = 4
    sim_time_s = np.array([(i + 1) * dt for i in range(nseg)], dtype=np.float64)
    rec = {
        "cfg": cfg,
        "sim_time_s": sim_time_s,
        "playback": {
            "events": [
                {
                    "day_index": 0,
                    "phase": "travel",
                    "t_start_s": 0.0,
                    "t_end_s": 2.0 * dt,
                    "t_moving_cumulative_at_start_s": 0.0,
                    "t_moving_cumulative_at_end_s": 2.0 * dt,
                },
                {
                    "day_index": 1,
                    "phase": "travel",
                    "t_start_s": 2.0 * dt,
                    "t_end_s": 4.0 * dt,
                    "t_moving_cumulative_at_start_s": 2.0 * dt,
                    "t_moving_cumulative_at_end_s": 4.0 * dt,
                },
            ]
        },
    }
    di = _moving_segment_day_indices_playback(rec, nseg)
    assert di is not None
    assert list(di.astype(int)) == [0, 0, 1, 1]


def test_moving_segment_day_indices_fallback_when_no_travel_overlap():
    from adaptive_scanning.viz import _moving_segment_day_indices_playback

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    nseg = 3
    sim_time_s = np.array([(i + 1) * dt for i in range(nseg)], dtype=np.float64)
    rec = {
        "cfg": cfg,
        "sim_time_s": sim_time_s,
        "playback": {
            "events": [
                {"day_index": 2, "phase": "morning_home", "t_start_s": 0.0, "t_end_s": 2.5 * dt},
                {
                    "day_index": 3,
                    "phase": "travel",
                    "t_start_s": 2.5 * dt,
                    "t_end_s": 5.0 * dt,
                },
            ]
        },
    }
    di = _moving_segment_day_indices_playback(rec, nseg)
    assert di is not None
    assert int(di[0]) == 2 and int(di[1]) == 2
    assert int(di[2]) == 3


def test_moving_segment_count_from_playback_trims_stationary_tail():
    from adaptive_scanning.viz import _moving_segment_count_from_playback

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    rec = {
        "cfg": cfg,
        "playback": {
            "events": [
                {
                    "day_index": 0,
                    "phase": "travel",
                    "t_moving_cumulative_at_end_s": 2.2 * dt,
                }
            ]
        },
    }
    assert _moving_segment_count_from_playback(rec, 100) == 3


def test_canonical_day_segment_indices_from_day_polylines():
    from adaptive_scanning.viz import _canonical_day_segment_indices_from_meta

    cfg = _tiny_cfg()
    rec = {
        "cfg": cfg,
        "osm_home_daily_meta": {
            "day_polylines_graph_m": [
                np.array([[0.0, 0.0], [20.0, 0.0]], dtype=np.float64),
                np.array([[20.0, 0.0], [60.0, 0.0]], dtype=np.float64),
            ]
        },
    }
    di = _canonical_day_segment_indices_from_meta(rec, 6)
    assert di is not None
    assert list(di.astype(int)) == [0, 0, 1, 1, 1, 1]


def test_home_daily_resampled_polylines_graph_returns_one_per_day():
    from adaptive_scanning.viz import _home_daily_resampled_polylines_graph

    cfg = _tiny_cfg()
    rec = {
        "cfg": cfg,
        "osm_home_daily_meta": {
            "day_polylines_graph_m": [
                np.array([[0.0, 0.0], [20.0, 0.0]], dtype=np.float64),
                np.array([[20.0, 0.0], [40.0, 0.0]], dtype=np.float64),
            ]
        },
    }
    out = _home_daily_resampled_polylines_graph(rec)
    assert out is not None
    assert [int(d) for d, *_rest in out] == [0, 1]


def test_policy_on_moving_segments_respects_travel_overlap():
    from adaptive_scanning.viz import _policy_on_moving_segments

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    rec = {
        "cfg": cfg,
        "camera_on_effective": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "sim_time_s": np.array([dt, 2.0 * dt, 3.0 * dt, 4.0 * dt], dtype=np.float64),
        "playback": {
            "events": [
                {
                    "day_index": 0,
                    "phase": "travel",
                    "t_start_s": 0.0,
                    "t_end_s": 2.0 * dt,
                    "t_moving_cumulative_at_start_s": 0.0,
                    "t_moving_cumulative_at_end_s": 2.0 * dt,
                },
                {
                    "day_index": 1,
                    "phase": "travel",
                    "t_start_s": 2.0 * dt,
                    "t_end_s": 4.0 * dt,
                    "t_moving_cumulative_at_start_s": 2.0 * dt,
                    "t_moving_cumulative_at_end_s": 4.0 * dt,
                },
            ]
        },
    }
    on = _policy_on_moving_segments(rec, 4)
    assert on is not None
    assert list(on.astype(bool)) == [False, True, False, True]


def test_policy_on_day_segments_from_playback_respects_day_local_segments():
    from adaptive_scanning.viz import _policy_on_day_segments_from_playback

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    rec = {
        "cfg": cfg,
        "camera_on_effective": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "sim_time_s": np.array([dt, 2.0 * dt, 3.0 * dt, 4.0 * dt], dtype=np.float64),
        "playback": {
            "events": [
                {
                    "day_index": 0,
                    "phase": "travel",
                    "t_start_s": 0.0,
                    "t_end_s": 2.0 * dt,
                    "t_moving_cumulative_at_start_s": 0.0,
                    "t_moving_cumulative_at_end_s": 2.0 * dt,
                },
                {
                    "day_index": 1,
                    "phase": "travel",
                    "t_start_s": 2.0 * dt,
                    "t_end_s": 4.0 * dt,
                    "t_moving_cumulative_at_start_s": 2.0 * dt,
                    "t_moving_cumulative_at_end_s": 4.0 * dt,
                },
            ]
        },
    }
    on0 = _policy_on_day_segments_from_playback(rec, day_index=0, nseg_day=2)
    on1 = _policy_on_day_segments_from_playback(rec, day_index=1, nseg_day=2)
    assert on0 is not None and on1 is not None
    assert list(on0.astype(bool)) == [True, False]
    assert list(on1.astype(bool)) == [True, False]


def test_policy_coverage_layers_follow_actual_env_path_day_labels():
    pytest.importorskip("geopandas")

    from adaptive_scanning.viz import policy_camera_coverage_layers_3857_v4

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    rec = {
        "cfg": cfg,
        "xs": np.array([0.0, 0.0, 10.0], dtype=np.float64),
        "ys": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "traj_heading_rad": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "polyline_graph_m": np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64),
        "camera_on_effective": np.array([1.0, 0.0], dtype=np.float32),
        "sim_time_s": np.array([dt, 2.0 * dt], dtype=np.float64),
        "playback": {
            "events": [
                {"day_index": 0, "t_start_s": 0.0, "t_end_s": dt, "phase": "morning_home"},
                {
                    "day_index": 0,
                    "t_start_s": 0.0,
                    "t_end_s": 2.0 * dt,
                    "phase": "travel",
                    "t_moving_cumulative_at_start_s": 0.0,
                    "t_moving_cumulative_at_end_s": 2.0 * dt,
                },
            ]
        },
    }
    layers = policy_camera_coverage_layers_3857_v4(
        rec,
        x_graph=np.array([0.0, 10.0], dtype=np.float64),
        y_graph=np.array([0.0, 0.0], dtype=np.float64),
        h_graph=np.array([0.0, 0.0], dtype=np.float64),
        graph_crs="EPSG:3857",
        stride=1,
        cfg=cfg,
    )
    assert layers is not None
    assert [name for name, _col, _geom in layers] == ["Coverage — policy camera Day 1"]


def test_policy_coverage_layers_are_subset_of_always_on_layers():
    pytest.importorskip("geopandas")

    from adaptive_scanning.viz import (
        home_daily_per_day_coverage_layers_3857_v2,
        policy_camera_coverage_layers_3857_v4,
    )

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    x = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    y = np.zeros_like(x)
    h = np.zeros_like(x)
    rec = {
        "cfg": cfg,
        "camera_on_effective": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "sim_time_s": np.array([dt, 2.0 * dt, 3.0 * dt, 4.0 * dt], dtype=np.float64),
        "playback": {
            "events": [
                {
                    "day_index": 0,
                    "t_start_s": 0.0,
                    "t_end_s": 2.0 * dt,
                    "phase": "travel",
                    "t_moving_cumulative_at_start_s": 0.0,
                    "t_moving_cumulative_at_end_s": 2.0 * dt,
                },
                {
                    "day_index": 1,
                    "t_start_s": 2.0 * dt,
                    "t_end_s": 4.0 * dt,
                    "phase": "travel",
                    "t_moving_cumulative_at_start_s": 2.0 * dt,
                    "t_moving_cumulative_at_end_s": 4.0 * dt,
                },
            ]
        },
    }

    always_layers = home_daily_per_day_coverage_layers_3857_v2(
        rec,
        graph_crs="EPSG:3857",
        x_graph=x,
        y_graph=y,
        h_graph=h,
        stride=1,
        cfg=cfg,
    )
    policy_layers = policy_camera_coverage_layers_3857_v4(
        rec,
        x_graph=x,
        y_graph=y,
        h_graph=h,
        graph_crs="EPSG:3857",
        stride=1,
        cfg=cfg,
    )

    assert always_layers is not None
    assert policy_layers is not None

    always_by_day = {name: geom for name, _col, geom in always_layers}
    for name, _col, geom in policy_layers:
        always_name = name.replace("policy camera", "always-on")
        assert always_name in always_by_day
        diff = geom.difference(always_by_day[always_name])
        assert diff.is_empty or float(diff.area) < 1e-9


def test_per_day_coverage_uses_episode_timeline_not_one_pass_route():
    pytest.importorskip("geopandas")

    from adaptive_scanning.viz import home_daily_per_day_coverage_layers_3857_v2

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    x = np.array([0.0, 10.0, 10.0, 20.0, 20.0], dtype=np.float64)
    y = np.zeros_like(x)
    h = np.zeros_like(x)
    rec = {
        "cfg": cfg,
        "sim_time_s": np.array([dt, 2.0 * dt, 3.0 * dt, 4.0 * dt], dtype=np.float64),
        "playback": {
            "events": [
                {"day_index": 0, "t_start_s": 0.0, "t_end_s": 2.0 * dt, "phase": "travel"},
                {"day_index": 1, "t_start_s": 2.0 * dt, "t_end_s": 4.0 * dt, "phase": "travel"},
            ]
        },
    }

    layers = home_daily_per_day_coverage_layers_3857_v2(
        rec,
        graph_crs="EPSG:3857",
        x_graph=x,
        y_graph=y,
        h_graph=h,
        stride=1,
        cfg=cfg,
    )

    assert layers is not None
    by_name = {name: geom for name, _col, geom in layers}
    g1 = by_name["Coverage — always-on Day 1"]
    g2 = by_name["Coverage — always-on Day 2"]
    assert float(g1.bounds[2]) < float(g2.bounds[2])


def test_policy_coverage_layers_include_multiple_days_from_actual_path():
    pytest.importorskip("geopandas")

    from adaptive_scanning.viz import policy_camera_coverage_layers_3857_v4

    cfg = _tiny_cfg()
    dt = float(cfg.dt_s)
    rec = {
        "cfg": cfg,
        "xs": np.array([2.0, 6.0, 10.0, 14.0, 18.0], dtype=np.float64),
        "ys": np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64),
        "traj_heading_rad": np.zeros(5, dtype=np.float64),
        "polyline_graph_m": np.array([[0.0, 0.0], [40.0, 20.0]], dtype=np.float64),
        "camera_on_effective": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "sim_time_s": np.array([dt, 2.0 * dt, 3.0 * dt, 4.0 * dt], dtype=np.float64),
        "playback": {
            "events": [
                {
                    "day_index": 0,
                    "t_start_s": 0.0,
                    "t_end_s": dt,
                    "phase": "travel",
                    "t_moving_cumulative_at_start_s": 0.0,
                    "t_moving_cumulative_at_end_s": dt,
                },
                {
                    "day_index": 1,
                    "t_start_s": dt,
                    "t_end_s": 2.0 * dt,
                    "phase": "travel",
                    "t_moving_cumulative_at_start_s": dt,
                    "t_moving_cumulative_at_end_s": 2.0 * dt,
                },
                {
                    "day_index": 2,
                    "t_start_s": 2.0 * dt,
                    "t_end_s": 3.0 * dt,
                    "phase": "travel",
                    "t_moving_cumulative_at_start_s": 2.0 * dt,
                    "t_moving_cumulative_at_end_s": 3.0 * dt,
                },
                {
                    "day_index": 3,
                    "t_start_s": 3.0 * dt,
                    "t_end_s": 4.0 * dt,
                    "phase": "travel",
                    "t_moving_cumulative_at_start_s": 3.0 * dt,
                    "t_moving_cumulative_at_end_s": 4.0 * dt,
                },
            ]
        },
    }

    layers = policy_camera_coverage_layers_3857_v4(
        rec,
        x_graph=np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        y_graph=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        h_graph=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        graph_crs="EPSG:3857",
        stride=1,
        cfg=cfg,
    )

    assert layers is not None
    assert [name for name, _col, _geom in layers] == [
        "Coverage — policy camera Day 1",
        "Coverage — policy camera Day 2",
        "Coverage — policy camera Day 3",
        "Coverage — policy camera Day 4",
    ]
