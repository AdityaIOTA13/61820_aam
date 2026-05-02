"""
CLI for adaptive scanning simulation.

  python -m adaptive_scanning.run_sim eval
  python -m adaptive_scanning.run_sim train --epochs 30
  python -m adaptive_scanning.run_sim export --out outputs/adaptive_scanning/episodes.npz
  python -m adaptive_scanning.run_sim visualize --fast --policy random --out outputs/adaptive_scanning/preview.png
  python -m adaptive_scanning.run_sim visualize --fast --one-path --out outputs/adaptive_scanning/osm_one_leg.png
  python -m adaptive_scanning.run_sim four-paths --place "Cambridge, Massachusetts, USA" --seed 2 --out outputs/adaptive_scanning/four_paths_example
  # Default OSM area when --streets/--one-path with no --place/--bbox: Cambridge, MA (see street_trajectories.DEFAULT_OSM_PLACE)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _add_street_cli_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--streets",
        action="store_true",
        help="Use OSM walk network + shortest-path motion (requires: pip install osmnx geopandas)",
    )
    sub.add_argument(
        "--place",
        type=str,
        default="",
        help="OSMnx place query (default when using streets with no bbox: Cambridge, MA). Example: 'Somerville, Massachusetts, USA'",
    )
    sub.add_argument(
        "--bbox",
        type=str,
        default="",
        help="WGS84 bounding box west,south,east,north (comma-separated lon/lat degrees)",
    )
    sub.add_argument(
        "--osm-cache-dir",
        type=str,
        default="",
        help="Directory for pickled OSM graphs (default: config osm_cache_dir)",
    )
    sub.add_argument(
        "--osm-network-type",
        type=str,
        default="",
        help="OSMnx network_type, default walk",
    )
    sub.add_argument(
        "--one-path",
        action="store_true",
        help="Single OSM shortest path (one start→end); implies --streets",
    )


def _merge_street_cli(cfg: "AdaptiveScanningConfig", args: argparse.Namespace) -> None:
    one_path = bool(getattr(args, "one_path", False))
    if one_path:
        cfg.osm_single_leg = True
        cfg.motion_mode = "streets"
    if not getattr(args, "streets", False) and not one_path:
        return
    if not one_path:
        cfg.motion_mode = "streets"
    bbox_str = (getattr(args, "bbox", "") or "").strip()
    if bbox_str:
        parts = [float(x.strip()) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be four comma-separated numbers: west,south,east,north")
        cfg.osm_bbox = (parts[0], parts[1], parts[2], parts[3])
        cfg.osm_place = ""
    elif (getattr(args, "place", "") or "").strip():
        cfg.osm_place = args.place.strip()
        cfg.osm_bbox = None
    ocd = (getattr(args, "osm_cache_dir", "") or "").strip()
    if ocd:
        cfg.osm_cache_dir = ocd
    ont = (getattr(args, "osm_network_type", "") or "").strip()
    if ont:
        cfg.osm_network_type = ont


def _fast_cfg() -> "AdaptiveScanningConfig":
    from adaptive_scanning.config import AdaptiveScanningConfig

    return AdaptiveScanningConfig(
        nx=24,
        ny=24,
        resolution_m=2.0,
        max_sim_time_s=2 * 3600.0,
        day_duration_s=3600.0,
        seconds_video_budget_per_day=60.0,
        dt_s=10.0,
        patch_cells=15,
    )


def _default_cfg() -> "AdaptiveScanningConfig":
    from adaptive_scanning.config import AdaptiveScanningConfig

    return AdaptiveScanningConfig()


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Adaptive camera budget simulation")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Evaluate baseline policies")
    pe.add_argument("--fast", action="store_true", help="Small grid / short horizon for smoke tests")
    pe.add_argument("--episodes", type=int, default=16)
    pe.add_argument("--seed", type=int, default=0)
    _add_street_cli_args(pe)

    pt = sub.add_parser("train", help="Train REINFORCE MLP policy")
    pt.add_argument("--fast", action="store_true", help="Small grid / short horizon for smoke tests")
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--episodes-per-epoch", type=int, default=8)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--out", type=str, default="", help="Optional path to save policy .pt")
    _add_street_cli_args(pt)

    px = sub.add_parser("export", help="Generate synthetic episode batch to .npz")
    px.add_argument("--fast", action="store_true", help="Small grid / short horizon for smoke tests")
    px.add_argument("--out", type=str, default=str(ROOT / "outputs" / "adaptive_scanning" / "episodes.npz"))
    px.add_argument("--n-episodes", type=int, default=32)
    px.add_argument("--seed", type=int, default=0)
    _add_street_cli_args(px)

    pv = sub.add_parser("visualize", help="Save a 3-panel PNG of one synthetic episode (path, map age, camera on)")
    pv.add_argument("--fast", action="store_true", help="Small grid / short horizon for smoke tests")
    pv.add_argument(
        "--policy",
        type=str,
        default="random",
        help="random | always_on | always_off | greedy_stale | greedy_budget",
    )
    pv.add_argument("--seed", type=int, default=0)
    pv.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "adaptive_scanning" / "episode_preview.png"),
    )
    _add_street_cli_args(pv)

    pch = sub.add_parser(
        "check-osm",
        help="Verify osmnx/geopandas and download default bbox graph (prints JSON)",
    )
    pch.add_argument(
        "--osm-cache-dir",
        type=str,
        default=str(ROOT / "outputs" / "adaptive_scanning" / "osm_cache"),
    )

    pf = sub.add_parser(
        "four-paths",
        help="Four OSM shortest paths with overlapping start/end clusters (PNG + optional HTML)",
    )
    pf.add_argument("--fast", action="store_true", help="Unused for now; keeps CLI consistent")
    pf.add_argument("--seed", type=int, default=0)
    pf.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "adaptive_scanning" / "four_paths_example"),
        help="Base path without extension; writes .png and .html",
    )
    _add_street_cli_args(pf)

    args = p.parse_args(argv)
    from adaptive_scanning.config import AdaptiveScanningConfig
    from adaptive_scanning.env import CameraBudgetEnv

    use_fast = bool(getattr(args, "fast", False))
    cfg: AdaptiveScanningConfig = _fast_cfg() if use_fast else _default_cfg()
    _merge_street_cli(cfg, args)

    if args.cmd == "eval":
        from adaptive_scanning.policies import (
            AlwaysOffPolicy,
            AlwaysOnPolicy,
            BudgetAwareGreedyPolicy,
            GreedyLocalStalenessPolicy,
            RandomPolicy,
        )
        from adaptive_scanning.rollout import eval_policy, run_episode

        rng_seed = int(args.seed)
        env = CameraBudgetEnv(cfg, seed=rng_seed)
        policies = {
            "random": RandomPolicy(np.random.default_rng(rng_seed)),
            "always_on": AlwaysOnPolicy(),
            "always_off": AlwaysOffPolicy(),
            "greedy_stale": GreedyLocalStalenessPolicy(),
            "greedy_budget": BudgetAwareGreedyPolicy(),
        }
        rows = {}
        for name, pol in policies.items():
            rows[name] = eval_policy(env, pol, n_episodes=args.episodes, seed0=rng_seed + 1)
        ref = run_episode(env, AlwaysOnPolicy(), seed=rng_seed + 999)
        rows["always_on_single_ep_final_uncovered"] = ref.final_uncovered_fraction
        print(json.dumps(rows, indent=2))

    elif args.cmd == "train":
        from adaptive_scanning.training import save_policy, train_reinforce

        pol, result = train_reinforce(
            cfg,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            lr=args.lr,
            seed=args.seed,
        )
        print("last_epoch", json.dumps(result.history[-1] if result.history else {}, indent=2))
        if args.out:
            save_policy(args.out, pol, cfg)
            print("saved", args.out)

    elif args.cmd == "export":
        from adaptive_scanning.data_export import generate_synthetic_episode_batch, save_episode_npz

        batch = generate_synthetic_episode_batch(
            cfg=cfg, n_episodes=args.n_episodes, seed=args.seed
        )
        save_episode_npz(args.out, batch)
        print("wrote", args.out)

    elif args.cmd == "visualize":
        from adaptive_scanning.viz import visualize_episode

        path, traj_src, basemap, coverage_pack = visualize_episode(
            cfg,
            policy_name=str(args.policy),
            seed=int(args.seed),
            out_path=str(args.out),
        )
        print(str(path.resolve()))
        print(f"trajectory_source={traj_src}")
        if basemap is not None:
            print(str(basemap.resolve()))
        if coverage_pack is not None:
            for p in coverage_pack:
                if p is not None:
                    print(str(p.resolve()))

    elif args.cmd == "check-osm":
        from adaptive_scanning.street_trajectories import check_osm_setup

        cd = Path(str(args.osm_cache_dir))
        print(json.dumps(check_osm_setup(cache_dir=cd), indent=2))

    elif args.cmd == "four-paths":
        setattr(args, "streets", True)
        _merge_street_cli(cfg, args)
        from adaptive_scanning.viz import export_four_overlapping_paths_example

        png, html = export_four_overlapping_paths_example(
            cfg, seed=int(args.seed), out_base=str(args.out)
        )
        print(str(png.resolve()))
        if html is not None:
            print(str(html.resolve()))


if __name__ == "__main__":
    main()
