"""
CLI for adaptive scanning simulation.

  python -m adaptive_scanning.run_sim eval
  python -m adaptive_scanning.run_sim train --epochs 30 --out models/policy.pt
  # With --out, also writes models/policy_training/{config.json,metrics.jsonl,history.json,training_curves.png}
  # Override: --train-log-dir PATH | disable: --no-train-log
  # Evening greedy-unseen: --evening-leniency-after-s SEC | --evening-leniency-ramp-s SEC | --no-evening-leniency | …
  python -m adaptive_scanning.run_sim export --out outputs/adaptive_scanning/episodes.npz
  python -m adaptive_scanning.run_sim visualize --fast --policy random --out outputs/adaptive_scanning/preview.png
  python -m adaptive_scanning.run_sim visualize --fast --one-path --out outputs/adaptive_scanning/osm_one_leg.png
  python -m adaptive_scanning.run_sim four-paths --place "Cambridge, Massachusetts, USA" --seed 2 --out outputs/adaptive_scanning/four_paths_example
  python -m adaptive_scanning.run_sim visualize --fast --one-path --mit-campus --out outputs/adaptive_scanning/mit.png
  python -m adaptive_scanning.run_sim visualize --streets --mit-campus --home-commute --walks-per-day 3 --days 4 --same-home-p 0.6 --skip-episode-png --out outputs/adaptive_scanning/home_round
  python -m adaptive_scanning.run_sim export --streets --mit-campus --home-commute --walks-per-day 3 --days 4 --n-episodes 8 --out outputs/adaptive_scanning/home_commute_batch.npz
  python -m adaptive_scanning.run_sim visualize --policy greedy_budget --skip-episode-png --video-budget-minutes-per-day 3 --coverage-first-minutes-per-day 3 --out outputs/adaptive_scanning/day_prefix_demo.png
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
    sub.add_argument(
        "--mit-campus",
        action="store_true",
        help="Restrict OSM graph/maps to MIT main campus bbox (WGS84); sets bbox, clears --place",
    )


def _add_home_commute_cli_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--home-commute",
        action="store_true",
        help="Streets only: chained daily walks from/to one fixed home; see --repeat-destination-p",
    )
    sub.add_argument(
        "--walks-per-day",
        type=int,
        default=3,
        help="With --home-commute: shortest-path legs per day (>=2); walk i+1 starts where walk i ended",
    )
    sub.add_argument(
        "--days",
        type=int,
        default=0,
        help="With --home-commute: if >0, set episode length to this many full walking days (day_duration_s each)",
    )
    sub.add_argument(
        "--same-home-p",
        type=float,
        default=0.6,
        help="With --home-commute: unused (one home for the whole episode); kept for CLI compatibility",
    )
    sub.add_argument(
        "--repeat-destination-p",
        type=float,
        default=0.35,
        help="With --home-commute: per intermediate stop, prob. of reusing a prior day's destination (else new node)",
    )
    sub.add_argument(
        "--repeat-prior-recency",
        type=float,
        default=None,
        help="With --home-commute: when reusing a prior stop, weight ~ this^(days since that stop). "
        "1.0 = uniform over all prior days; lower values favor yesterday (less long-gap overlap). "
        "Omit to keep AdaptiveScanningConfig.osm_repeat_prior_stops_recency (default 0.55).",
    )


def _apply_home_commute_cli(cfg: "AdaptiveScanningConfig", args: argparse.Namespace) -> None:
    if not bool(getattr(args, "home_commute", False)):
        return
    if cfg.motion_mode != "streets":
        raise SystemExit("--home-commute requires --streets (OSM motion)")
    cfg.osm_daily_home_commute = True
    cfg.osm_single_leg = False
    cfg.osm_walks_per_day = max(2, int(args.walks_per_day))
    cfg.osm_same_home_next_day_p = float(args.same_home_p)
    cfg.osm_repeat_destination_across_days_p = float(
        getattr(args, "repeat_destination_p", 0.35)
    )
    if getattr(args, "repeat_prior_recency", None) is not None:
        cfg.osm_repeat_prior_stops_recency = float(args.repeat_prior_recency)
    ndays = int(getattr(args, "days", 0))
    if ndays > 0:
        cfg.max_sim_time_s = float(ndays) * float(cfg.day_duration_s)


def _merge_street_cli(cfg: "AdaptiveScanningConfig", args: argparse.Namespace) -> None:
    one_path = bool(getattr(args, "one_path", False))
    if one_path:
        cfg.osm_single_leg = True
        cfg.motion_mode = "streets"
        cfg.osm_daily_home_commute = False  # incompatible with single start→end leg
    if not getattr(args, "streets", False) and not one_path:
        return
    if not one_path:
        cfg.motion_mode = "streets"
    bbox_str = (getattr(args, "bbox", "") or "").strip()
    mit = bool(getattr(args, "mit_campus", False))
    if mit and bbox_str:
        raise SystemExit("Use only one of --mit-campus or --bbox")
    if mit:
        from adaptive_scanning.street_trajectories import MIT_CAMPUS_BBOX_WGS84

        cfg.osm_bbox = MIT_CAMPUS_BBOX_WGS84
        cfg.osm_place = ""
    elif bbox_str:
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
        seconds_video_budget_per_day=150.0,  # reduced budget for fast smoke runs
        dt_s=5.0,
        patch_cells=15,
        motion_mode="box",
        osm_daily_home_commute=False,
        osm_single_leg=False,
        osm_bbox=None,
        osm_place="",
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
    _add_home_commute_cli_args(pe)

    pt = sub.add_parser("train", help="Train REINFORCE MLP policy")
    pt.add_argument("--fast", action="store_true", help="Small grid / short horizon for smoke tests")
    pt.add_argument("--epochs", type=int, default=40)
    pt.add_argument("--episodes-per-epoch", type=int, default=8)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument(
        "--entropy-coef",
        type=float,
        default=0.1,
        help="REINFORCE entropy bonus (maximize H); use 0 to disable. Helps avoid always-off collapse.",
    )
    pt.add_argument(
        "--camera-on-bonus",
        type=float,
        default=0.0,
        help="Extra reward per env step when camera is effectively on (RL bootstrap). Default 0 (off).",
    )
    pt.add_argument(
        "--unused-budget-penalty",
        type=float,
        default=None,
        metavar="W",
        help="Weight on unused fraction of daily camera budget applied at each simulated day boundary "
        "(subtract W × leftover_fraction). Omit to use AdaptiveScanningConfig.w_unused_budget_end_of_day.",
    )
    pt.add_argument(
        "--no-evening-leniency",
        action="store_true",
        help="Disable greedy-unseen evening relax (evening_lenient_after_s_since_day_start=-1).",
    )
    pt.add_argument(
        "--evening-leniency-after-s",
        type=float,
        default=None,
        metavar="SEC",
        help="Override evening_lenient_after_s_since_day_start (seconds since simulated day start; <0 disables).",
    )
    pt.add_argument(
        "--evening-leniency-ramp-s",
        type=float,
        default=None,
        metavar="SEC",
        help="Override evening_lenient_ramp_s (linear 0→1 leniency weight over this span after --after-s).",
    )
    pt.add_argument(
        "--evening-leniency-wedge-mult",
        type=float,
        default=None,
        metavar="M",
        help="Override evening_lenient_wedge_suppress_age_mult (>=1 typical).",
    )
    pt.add_argument(
        "--evening-leniency-foot-mult",
        type=float,
        default=None,
        metavar="M",
        help="Override evening_lenient_foot_grace_mult (>=1 typical).",
    )
    pt.add_argument(
        "--evening-leniency-min-budget-frac",
        type=float,
        default=None,
        metavar="F",
        help="Override evening_lenient_min_budget_frac; leniency only if remaining budget fraction ≥ this.",
    )
    pt.add_argument("--out", type=str, default="", help="Optional path to save policy .pt")
    pt.add_argument(
        "--video-budget-minutes-per-day",
        type=float,
        default=0.0,
        help="If >0, set seconds_video_budget_per_day to this many minutes of camera-on per day_duration_s",
    )
    pt.add_argument(
        "--video-budget-ref-speed",
        type=float,
        default=None,
        metavar="M_S",
        help="If set: video_budget_reference_walk_speed_m_s for scaling daily SI budget with walk_speed "
        "(0 = fixed seconds regardless of speed). Omit for config default.",
    )
    pt.add_argument(
        "--no-train-progress",
        action="store_true",
        help="Disable tqdm epoch progress bar during training",
    )
    pt.add_argument(
        "--train-log-dir",
        type=str,
        default="",
        help="Write config.json, metrics.jsonl (one line/epoch), history.json, training_curves.png here",
    )
    pt.add_argument(
        "--no-train-log",
        action="store_true",
        help="Disable training logs even when --out would imply a default log directory",
    )
    _add_street_cli_args(pt)
    _add_home_commute_cli_args(pt)

    px = sub.add_parser("export", help="Generate synthetic episode batch to .npz")
    px.add_argument("--fast", action="store_true", help="Small grid / short horizon for smoke tests")
    px.add_argument("--out", type=str, default=str(ROOT / "outputs" / "adaptive_scanning" / "episodes.npz"))
    px.add_argument("--n-episodes", type=int, default=32)
    px.add_argument("--seed", type=int, default=0)
    _add_street_cli_args(px)
    _add_home_commute_cli_args(px)

    pv = sub.add_parser(
        "visualize",
        help="Episode preview: 3-panel PNG (optional) plus basemap/coverage sidecars when streets mode applies",
    )
    pv.add_argument("--fast", action="store_true", help="Small grid / short horizon for smoke tests")
    pv.add_argument(
        "--policy",
        type=str,
        default="random",
        help="random | always_on | always_off | greedy_stale | greedy_budget | greedy_unseen",
    )
    pv.add_argument("--seed", type=int, default=0)
    pv.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "adaptive_scanning" / "episode_preview.png"),
    )
    pv.add_argument(
        "--skip-episode-png",
        action="store_true",
        help="Do not write the 3-panel episode PNG; still write basemap/coverage outputs that use the same stem",
    )
    pv.add_argument(
        "--coverage-first-minutes-per-day",
        type=float,
        default=0.0,
        help="If >0, also write day-prefix coverage: with playback, first N·60·walk_speed metres of path *after morning_home* each day; else first N minutes of sim clock per day. Writes *_day_first{Ns}_*.png/.html",
    )
    pv.add_argument(
        "--video-budget-minutes-per-day",
        type=float,
        default=0.0,
        help="If >0, set seconds_video_budget_per_day to this many minutes of camera-on per day_duration_s (default is 10 min unless --fast)",
    )
    pv.add_argument(
        "--video-budget-ref-speed",
        type=float,
        default=None,
        metavar="M_S",
        help="If set: video_budget_reference_walk_speed_m_s (0 = no walk-speed scaling of daily budget). Omit for config default.",
    )
    _add_home_commute_cli_args(pv)
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
        help="Four OSM shortest paths with probabilistic reuse of prior starts/ends (PNG + optional HTML)",
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
    _apply_home_commute_cli(cfg, args)

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

        vbm_tr = float(getattr(args, "video_budget_minutes_per_day", 0.0))
        if vbm_tr > 0.0:
            cfg.seconds_video_budget_per_day = float(vbm_tr) * 60.0
        vbr = getattr(args, "video_budget_ref_speed", None)
        if vbr is not None:
            cfg.video_budget_reference_walk_speed_m_s = float(vbr)

        cfg.reward_camera_on_bonus = float(args.camera_on_bonus)
        ubp = getattr(args, "unused_budget_penalty", None)
        if ubp is not None:
            cfg.w_unused_budget_end_of_day = float(ubp)

        if bool(getattr(args, "no_evening_leniency", False)):
            cfg.evening_lenient_after_s_since_day_start = -1.0
        else:
            ela = getattr(args, "evening_leniency_after_s", None)
            if ela is not None:
                cfg.evening_lenient_after_s_since_day_start = float(ela)
        elr = getattr(args, "evening_leniency_ramp_s", None)
        if elr is not None:
            cfg.evening_lenient_ramp_s = float(elr)
        elwm = getattr(args, "evening_leniency_wedge_mult", None)
        if elwm is not None:
            cfg.evening_lenient_wedge_suppress_age_mult = float(elwm)
        elfm = getattr(args, "evening_leniency_foot_mult", None)
        if elfm is not None:
            cfg.evening_lenient_foot_grace_mult = float(elfm)
        elbf = getattr(args, "evening_leniency_min_budget_frac", None)
        if elbf is not None:
            cfg.evening_lenient_min_budget_frac = float(elbf)

        train_log: str | None = None
        if not bool(getattr(args, "no_train_log", False)):
            tld = (getattr(args, "train_log_dir", "") or "").strip()
            if tld:
                train_log = tld
            elif (getattr(args, "out", "") or "").strip():
                outp = Path(str(args.out).strip())
                train_log = str(outp.parent / f"{outp.stem}_training")

        pol, result = train_reinforce(
            cfg,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            lr=args.lr,
            entropy_coef=float(getattr(args, "entropy_coef", 0.1)),
            seed=args.seed,
            show_progress=not bool(getattr(args, "no_train_progress", False)),
            log_dir=train_log,
        )
        print("last_epoch", json.dumps(result.history[-1] if result.history else {}, indent=2))
        if train_log:
            print("training_log_dir", train_log)
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

        vbm = float(getattr(args, "video_budget_minutes_per_day", 0.0))
        vbm_set = vbm > 0.0
        if vbm_set:
            cfg.seconds_video_budget_per_day = float(vbm) * 60.0
        vbr = getattr(args, "video_budget_ref_speed", None)
        if vbr is not None:
            cfg.video_budget_reference_walk_speed_m_s = float(vbr)

        if str(args.policy).lower().strip() == "greedy_unseen":
            from dataclasses import replace

            g_kw = {"update_last_seen_only_on_first_hit": True}
            if not vbm_set:
                g_kw["seconds_video_budget_per_day"] = 600.0
            cfg = replace(cfg, **g_kw)

        path, traj_src, basemap, coverage_pack, playback_json, day_prefix_pack = visualize_episode(
            cfg,
            policy_name=str(args.policy),
            seed=int(args.seed),
            out_path=str(args.out),
            skip_episode_png=bool(args.skip_episode_png),
            coverage_first_minutes_per_day=float(
                getattr(args, "coverage_first_minutes_per_day", 0.0)
            ),
        )
        if path is not None:
            print(str(path.resolve()))
        print(f"trajectory_source={traj_src}")
        if basemap is not None:
            print(str(basemap.resolve()))
        if coverage_pack is not None:
            for p in coverage_pack:
                if p is not None:
                    print(str(p.resolve()))
        if playback_json is not None:
            print(str(playback_json.resolve()))
        if day_prefix_pack is not None:
            for p in day_prefix_pack:
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
