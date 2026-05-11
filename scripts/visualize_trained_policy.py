#!/usr/bin/env python3
"""
Load ``models/policy.pt`` (or any checkpoint from ``train``) and visualize one episode.

Uses the ``AdaptiveScanningConfig`` stored in the checkpoint (MIT campus + home commute
if that was how you trained). Writes the same artifacts as ``run_sim visualize``:
3-panel PNG, basemap, coverage maps when OSM data is available.

By default, episode length follows ``AdaptiveScanningConfig`` (``--days`` defaults to that many
walking days), so maps match the current repo even when the checkpoint still stores an older
shorter ``max_sim_time_s``. Use ``--use-checkpoint-horizon`` to keep the checkpoint horizon exactly.
Override length with ``--days N``.

Example::

    python scripts/visualize_trained_policy.py \\
        --checkpoint models/policy.pt \\
        --out outputs/adaptive_scanning/trained_demo.png \\
        --seed 42

Greedy baseline (same env / checkpoint config except 10 min/day budget and first-hit stamps;
camera only when the agent's current grid cell has never been scanned)::

    python scripts/visualize_trained_policy.py \\
        --checkpoint models/policy-new.pt \\
        --policy greedy_unseen \\
        --out outputs/adaptive_scanning/policy_new_greedy_unseen.png \\
        --seed 44
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_episode_ndays() -> int:
    """Walking-day count implied by current ``AdaptiveScanningConfig`` defaults."""
    from adaptive_scanning.config import AdaptiveScanningConfig

    c = AdaptiveScanningConfig()
    dd = float(c.day_duration_s)
    return max(1, int(round(float(c.max_sim_time_s) / max(dd, 1e-9))))


def main() -> None:
    default_ndays = _default_episode_ndays()
    p = argparse.ArgumentParser(description="Visualize a trained MLPPolicy on one MIT-style episode")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "models" / "policy.pt"),
        help="Path to policy .pt from adaptive_scanning.run_sim train",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "adaptive_scanning" / "trained_policy_demo.png"),
        help="Output PNG stem (sidecars use the same stem)",
    )
    p.add_argument("--seed", type=int, default=42, help="Episode RNG seed")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="cpu | cuda | cuda:0 … (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--skip-episode-png",
        action="store_true",
        help="Only write basemap/coverage sidecars, not the 3-panel figure",
    )
    p.add_argument(
        "--coverage-first-minutes-per-day",
        type=float,
        default=0.0,
        help="If >0, also write day-prefix coverage like run_sim visualize",
    )
    p.add_argument(
        "--policy",
        type=str,
        choices=("trained", "greedy_unseen"),
        default="trained",
        help=(
            "trained: MLPPolicy weights from checkpoint; "
            "greedy_unseen: camera only when the agent's current grid cell has never been scanned (no map age at foot); "
            "10 min/day budget for this run"
        ),
    )
    p.add_argument(
        "--greedy-min-budget-frac",
        type=float,
        default=0.08,
        help="If remaining daily budget / day budget is below this, greedy_unseen stays OFF",
    )
    p.add_argument(
        "--days",
        type=int,
        default=default_ndays,
        metavar="N",
        help=(
            "Episode length in full simulated walking days "
            "(sets max_sim_time_s = N * day_duration_s). "
            f"Default {default_ndays} from AdaptiveScanningConfig."
        ),
    )
    p.add_argument(
        "--use-checkpoint-horizon",
        action="store_true",
        help="Do not override max_sim_time_s; use the value stored in the checkpoint.",
    )
    args = p.parse_args()

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck.resolve()}")

    from dataclasses import replace

    from adaptive_scanning.config import config_from_saved_dict
    from adaptive_scanning.policies import BudgetAwareGreedyUnseenOnlyPolicy
    from adaptive_scanning.training import load_policy
    from adaptive_scanning.viz import visualize_episode

    import torch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.policy == "trained":
        dev = (args.device or "").strip() or None
        policy, cfg = load_policy(str(ck), device=dev)
        label = "trained_mlp"
    else:
        try:
            ckpt = torch.load(str(ck), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ck), map_location="cpu")
        cfg = replace(
            config_from_saved_dict(ckpt["cfg"]),
            update_last_seen_only_on_first_hit=True,
            seconds_video_budget_per_day=600.0,
        )
        policy = BudgetAwareGreedyUnseenOnlyPolicy(
            min_budget_frac_to_turn_on=float(args.greedy_min_budget_frac),
        )
        label = "greedy_unseen"

    if not bool(getattr(args, "use_checkpoint_horizon", False)):
        nd = max(1, int(args.days))
        new_max = float(nd) * float(cfg.day_duration_s)
        old_max = float(cfg.max_sim_time_s)
        cfg = replace(cfg, max_sim_time_s=new_max)
        if abs(new_max - old_max) > 0.5:
            print(
                f"episode_horizon: {nd} walking days → max_sim_time_s={new_max:.0f}s "
                f"(checkpoint had {old_max:.0f}s)",
                flush=True,
            )

    panel, traj_src, basemap, coverage_pack, playback_json, day_prefix = visualize_episode(
        cfg,
        policy=policy,
        policy_name=label,
        seed=int(args.seed),
        out_path=out,
        skip_episode_png=bool(args.skip_episode_png),
        coverage_first_minutes_per_day=float(args.coverage_first_minutes_per_day),
    )
    if panel is not None:
        print(str(panel.resolve()))
    print(f"trajectory_source={traj_src}")
    if basemap is not None:
        print(str(basemap.resolve()))
    if coverage_pack is not None:
        for path in coverage_pack:
            if path is not None:
                print(str(path.resolve()))
    if playback_json is not None:
        print(str(playback_json.resolve()))
    if day_prefix is not None:
        for path in day_prefix:
            if path is not None:
                print(str(path.resolve()))


if __name__ == "__main__":
    main()
