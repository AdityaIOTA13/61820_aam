import tempfile
from pathlib import Path

import numpy as np

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.env import CameraBudgetEnv
from adaptive_scanning.policies import AlwaysOffPolicy, AlwaysOnPolicy, RandomPolicy
from adaptive_scanning.rollout import run_episode
from adaptive_scanning.data_export import generate_synthetic_episode_batch


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
    cfg = _tiny_cfg()
    batch = generate_synthetic_episode_batch(cfg=cfg, n_episodes=2, seed=0)
    assert len(batch["episodes"]) == 2
    assert batch["episodes"][0]["actions"].shape[0] == batch["episodes"][0]["rewards"].shape[0]


def test_visualize_episode_writes_png():
    from adaptive_scanning.viz import visualize_episode

    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "preview.png"
        path, _src, _bm, _cov_pack = visualize_episode(cfg, policy_name="always_on", seed=3, out_path=out)
        assert path.exists()
        assert path.stat().st_size > 500
