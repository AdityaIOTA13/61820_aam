import numpy as np
import pytest

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.coverage_eval import (
    PrecomputedActionPolicy,
    run_episode_metrics,
    run_benchmark,
)
from adaptive_scanning.env import CameraBudgetEnv
from adaptive_scanning.oracle import (
    _max_camera_on_steps_per_day,
    greedy_coverage_oracle_actions,
    random_full_daily_budget_actions,
)
from adaptive_scanning.policies import AlwaysOffPolicy, AlwaysOnPolicy, BudgetAwareGreedyUnseenOnlyPolicy, RandomPolicy


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
        observation_mode="foot_cell",
    )


def test_random_policy_respects_stationary_interval():
    pol = RandomPolicy(np.random.default_rng(0))
    assert pol.act(np.zeros(3), {"interval_is_moving": False}) == 0
    assert pol.act(np.zeros(3), {"interval_is_moving": True}) in (0, 1)


def test_precomputed_policy_aligns_with_env_step_idx():
    cfg = _tiny_cfg()
    actions = greedy_coverage_oracle_actions(cfg, seed=0)
    env = CameraBudgetEnv(cfg, seed=0)
    env.reset(seed=0)
    assert env._traj_x is not None
    assert len(actions) == len(env._traj_x) - 1
    m = run_episode_metrics(CameraBudgetEnv(cfg, seed=0), PrecomputedActionPolicy(actions), seed=0)
    assert 0.0 <= m.coverage_vs_theoretic_max <= 1.0 + 1e-6
    assert m.n_cells_policy <= m.n_cells_theoretic_max + 1
    assert m.lcc_vs_theoretic_max <= m.coverage_vs_theoretic_max + 1e-9
    assert m.lcc_cells <= m.n_cells_policy


def test_always_on_covers_at_least_as_much_as_always_off():
    cfg = _tiny_cfg()
    seed = 3
    m_on = run_episode_metrics(CameraBudgetEnv(cfg, seed=seed), AlwaysOnPolicy(), seed=seed)
    m_off = run_episode_metrics(CameraBudgetEnv(cfg, seed=seed), AlwaysOffPolicy(), seed=seed)
    assert m_on.n_cells_policy >= m_off.n_cells_policy
    assert m_on.coverage_vs_theoretic_max > m_off.coverage_vs_theoretic_max
    assert m_off.coverage_vs_theoretic_max == pytest.approx(0.0)


def test_oracle_coverage_vs_always_on_high():
    cfg = _tiny_cfg()
    seed = 7
    actions = greedy_coverage_oracle_actions(cfg, seed=seed)
    m_oracle = run_episode_metrics(CameraBudgetEnv(cfg, seed=seed), PrecomputedActionPolicy(actions), seed=seed)
    m_greedy = run_episode_metrics(CameraBudgetEnv(cfg, seed=seed), BudgetAwareGreedyUnseenOnlyPolicy(), seed=seed)
    assert m_oracle.coverage_vs_theoretic_max >= m_greedy.coverage_vs_theoretic_max - 1e-6


def test_random_full_daily_budget_uses_quota_per_day():
    cfg = _tiny_cfg()
    seed = 42
    actions = random_full_daily_budget_actions(cfg, seed=seed)
    env = CameraBudgetEnv(cfg, seed=seed)
    env.reset(seed=seed)
    assert env._traj_x is not None
    T = len(env._traj_x) - 1
    dt = float(cfg.dt_s)
    day_dur = float(cfg.day_duration_s)
    R = _max_camera_on_steps_per_day(cfg)
    by_day: dict[int, list[int]] = {}
    for k in range(T):
        mv, _ = env.coverage_mask_for_step_index(k)
        if mv:
            d = int((k * dt) // max(day_dur, 1e-9))
            by_day.setdefault(d, []).append(k)
    expected_on = sum(min(R, len(v)) for v in by_day.values())
    assert sum(actions) == expected_on


def test_run_benchmark_smoke_no_checkpoint():
    cfg = _tiny_cfg()
    out = run_benchmark(cfg, checkpoint_path=None, n_scenarios=2, seed0=1, device="cpu")
    assert out["summary"]["random"] is not None
    assert out["summary"]["trained"] is None
    assert float(out["dt_s"]) == float(cfg.dt_s)
    assert "coverage_vs_theoretic_max_mean" in out["summary"]["oracle"]
    assert "lcc_vs_theoretic_max_mean" in out["summary"]["oracle"]
    assert "effective_camera_on_sec_mean" in out["summary"]["oracle"]
    assert out["summary"]["greedy_unseen_progressive"] is not None
    assert "coverage_vs_theoretic_max_mean" in out["summary"]["greedy_unseen_progressive"]
