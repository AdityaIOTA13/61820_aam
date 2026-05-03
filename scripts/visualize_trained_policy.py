#!/usr/bin/env python3
"""
Load ``models/policy.pt`` (or any checkpoint from ``train``) and visualize one episode.

Uses the ``AdaptiveScanningConfig`` stored in the checkpoint (MIT campus + home commute
if that was how you trained). Writes the same artifacts as ``run_sim visualize``:
3-panel PNG, basemap, coverage maps when OSM data is available.

Example::

    python scripts/visualize_trained_policy.py \\
        --checkpoint models/policy.pt \\
        --out outputs/adaptive_scanning/trained_demo.png \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
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
    args = p.parse_args()

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"checkpoint not found: {ck.resolve()}")

    from adaptive_scanning.training import load_policy
    from adaptive_scanning.viz import visualize_episode

    dev = (args.device or "").strip() or None
    policy, cfg = load_policy(str(ck), device=dev)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    panel, traj_src, basemap, coverage_pack, playback_json, day_prefix = visualize_episode(
        cfg,
        policy=policy,
        policy_name="trained_mlp",
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
