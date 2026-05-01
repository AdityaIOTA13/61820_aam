#!/usr/bin/env python3
"""
Simulate revisits with a **shared coverage map** (``last_seen``) across calendar days.

**Default: two sessions per calendar day (time-of-day style)**

- **Morning** (first session): walk the eligible timeline **forward**. Greedy default
  ``--morning-greedy-trigger unseen`` so you mostly capture for **new coverage**, not for
  overnight staleness at the start of the walk.
- **Evening** (second session): walk **backward** (end of path first), optional **+pi rad** on rays
  (rear 360-deg hemisphere). Greedy default ``--evening-greedy-trigger unseen_or_stale`` so
  remaining budget can **refresh stale** cells before the day ends.

Sim timestamps advance by one **walk span** (+ optional gap) between morning and evening so
evening runs **later in the same calendar day** than morning (see ``--between-session-gap-sec``).
Across calendar days, ``--seconds-per-day-gap`` still shifts the clock (nights).

With ``--sessions-per-day 1``, behaviour falls back to a **single** walk per calendar day using
``--path-direction`` (forward / reverse / alternate) and ``--greedy-trigger``.

With ``--png-dir``, writes one set of PNGs **per session** (``dayNN_am_*`` / ``dayNN_pm_*`` when
two sessions). ``--reset-coverage`` clears the map at the **start of each calendar day**
(before morning), not between morning and evening.

``--animation-gif`` renders the **full eligible path** each session (pose moves every frame;
``SCAN`` when greedy merges depth). Staleness is **interpolated** between eligibles (see
``--animation-interp-substeps``, default **4**) so the heatmap does not jump in a single GIF
frame when sim time steps are coarse. Default ``--animation-frame-ms`` is **36** ms (fast playback).
JSON reports **coverage vs unconstrained**: fraction of cells ever hit vs merging **all**
eligible frames in time order with the same FOV (always-on, no budget, forward headings).

**FOV (Ray-Ban-style horizontal mask):** By default, rays are masked to ~100° horizontal
(``--hfov-deg`` / ``--vfov-deg``; see ``project_coverage``). Pass ``--fov-full-360`` only to
disable that horizontal masking. On the **evening** (reverse) session, default ``+π`` ray heading
(``--no-rear-evening-session`` to turn off) applies the **same** narrow cone after rotating the
camera azimuth, so the cone samples the **opposite** travel direction—wearable “rear hemisphere”
without widening the FOV.

Run from repo root::

    python scripts/simulate_revisit_days.py --days 4 --budget-seconds 18
    python scripts/simulate_revisit_days.py --sessions-per-day 1 --path-direction forward
    python scripts/simulate_revisit_days.py --greedy-trigger unseen_or_stale --png-dir outputs/revisit_maps
    python scripts/simulate_revisit_days.py --png-dir outputs/maps --animation-gif outputs/revisit_anim.gif
    python scripts/simulate_revisit_days.py --json-out outputs/revisit_sim.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import project_coverage as pc


def _rows_for_day(
    full_rows: list[tuple[str, float, float, float]],
    day_index: int,
    path_direction: str,
) -> tuple[list[tuple[str, float, float, float]], dict[str, float], bool]:
    if path_direction == "reverse":
        rev = True
    elif path_direction == "alternate":
        rev = day_index % 2 == 1
    else:
        rev = False
    rows = list(reversed(full_rows)) if rev else list(full_rows)
    hmap = pc.heading_by_name_from_rows(rows)
    return rows, hmap, rev


def _rows_for_session(
    full_rows: list[tuple[str, float, float, float]],
    *,
    sessions_per_day: int,
    session_index_zero_based: int,
    path_direction: str,
    calendar_day_index_zero_based: int,
) -> tuple[list[tuple[str, float, float, float]], dict[str, float], bool]:
    """
    Session 0 = morning (forward), session 1 = evening (reverse), when ``sessions_per_day >= 2``.
    Single-session mode delegates to ``_rows_for_day``.
    """
    if sessions_per_day <= 1:
        rows, hmap, rev = _rows_for_day(full_rows, calendar_day_index_zero_based, path_direction)
        return rows, hmap, rev
    rev = session_index_zero_based % 2 == 1
    rows = list(reversed(full_rows)) if rev else list(full_rows)
    hmap = pc.heading_by_name_from_rows(rows)
    return rows, hmap, rev


def _session_label(session_index: int, sessions_per_day: int) -> str:
    if sessions_per_day <= 1:
        return "day"
    return "morning" if session_index == 0 else "evening"


def _session_file_tag(session_index: int, sessions_per_day: int) -> str:
    if sessions_per_day <= 1:
        return ""
    return "am_" if session_index == 0 else "pm_"


def _frac_ever(last_seen: np.ndarray) -> float:
    ever = np.isfinite(last_seen) & (last_seen > -np.inf)
    return float(ever.mean()) if ever.size else 0.0


def _k_target_calendar_day(
    budget_seconds: float,
    full_rows: list[tuple[str, float, float, float]],
) -> int:
    """Same K = floor(budget / mean_dt) rule as greedy, for sharing across morning+evening."""
    n = len(full_rows)
    if n == 0:
        return 0
    times = [r[3] for r in full_rows]
    t_lo, t_hi = min(times), max(times)
    if n == 1:
        mean_dt = 0.5
    else:
        mean_dt = (t_hi - t_lo) / float(n - 1)
    if mean_dt <= 0:
        mean_dt = 0.5
    total_span = t_hi - t_lo
    if budget_seconds + 1e-6 >= total_span:
        return n
    return max(1, min(n, int(math.floor(budget_seconds / mean_dt + 1e-9))))


def _baseline_unconstrained_coverage(
    positions: dict,
    full_rows: list[tuple[str, float, float, float]],
    *,
    resolution_m: float,
    margin_m: float,
    subsample: int,
    hfov_deg: float,
    vfov_deg: float,
    projection: str,
) -> tuple[float, dict]:
    """
    Upper bound: every eligible frame in **chronological** order, path-tangent headings,
    **no** budget and **no** +pi offset (always-on forward cone vs trajectory).
    """
    names = [r[0] for r in full_rows]
    hmap = pc.heading_by_name_from_rows(full_rows)
    _ls, _ever, _t_end, meta = pc.run_map_age(
        positions,
        names,
        resolution_m=resolution_m,
        margin_m=margin_m,
        subsample=subsample,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        projection=projection,
        heading_by_name=hmap,
    )
    return float(meta["frac_cells_ever_seen"]), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=3, help="Number of calendar days")
    ap.add_argument(
        "--budget-seconds",
        type=float,
        default=18.0,
        help="Total camera budget per **calendar day** (shared across morning+evening when sessions-per-day is 2)",
    )
    ap.add_argument(
        "--sessions-per-day",
        type=int,
        default=2,
        help="2 = morning forward + evening reverse (default). 1 = one walk/day using --path-direction",
    )
    ap.add_argument(
        "--between-session-gap-sec",
        type=float,
        default=0.0,
        help="Extra simulated seconds between end of morning walk clock and start of evening clock",
    )
    ap.add_argument(
        "--path-direction",
        choices=("forward", "reverse", "alternate"),
        default="alternate",
        help="Only used when --sessions-per-day is 1: timeline order for that single walk",
    )
    ap.add_argument(
        "--morning-greedy-trigger",
        choices=("unseen", "unseen_or_stale", "mean_age"),
        default="unseen",
        help="Greedy trigger for morning when sessions-per-day >= 2 (default: unseen)",
    )
    ap.add_argument(
        "--evening-greedy-trigger",
        choices=("unseen", "unseen_or_stale", "mean_age"),
        default="unseen_or_stale",
        help="Greedy trigger for evening when sessions-per-day >= 2 (default: unseen_or_stale)",
    )
    ap.add_argument(
        "--rear-camera",
        action="store_true",
        help="Add pi rad to ray heading every session",
    )
    ap.add_argument(
        "--no-rear-evening-session",
        action="store_false",
        dest="rear_pi_evening_session",
        default=True,
        help="When two sessions/day: disable +pi on the evening (reverse) walk",
    )
    ap.add_argument(
        "--no-rear-on-reverse-days",
        action="store_false",
        dest="rear_pi_on_reverse_timeline",
        default=True,
        help="When **sessions-per-day 1** and path is reversed: disable +pi on those days",
    )
    ap.add_argument(
        "--rear-camera-on-alternate-days",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--reset-coverage",
        action="store_true",
        help="Clear the coverage map at the start of each **calendar day** (before morning)",
    )
    ap.add_argument(
        "--seconds-per-day-gap",
        type=float,
        default=86400.0,
        help="Simulated seconds between calendar midnights (map-age / day offset)",
    )
    ap.add_argument("--resolution", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--subsample", type=int, default=8)
    ap.add_argument("--projection", choices=("backproject", "ground_plane"), default="backproject")
    ap.add_argument(
        "--fov-full-360",
        action="store_true",
        help="Disable horizontal FOV masking (full sphere); default keeps ~100° Ray-Ban-style cone",
    )
    ap.add_argument("--hfov-deg", type=float, default=pc.RAYBAN_META_DEFAULT_HFOV_DEG)
    ap.add_argument("--vfov-deg", type=float, default=pc.RAYBAN_META_DEFAULT_VFOV_DEG)
    ap.add_argument(
        "--greedy-trigger",
        choices=("unseen", "unseen_or_stale", "mean_age"),
        default="unseen",
        help="Greedy trigger when --sessions-per-day is 1 (ignored when 2+ unless you only use morning/evening defaults)",
    )
    ap.add_argument("--greedy-unseen-scope", choices=("foot", "disk"), default="foot")
    ap.add_argument("--greedy-radius-m", type=float, default=4.0)
    ap.add_argument("--greedy-mean-age-threshold-s", type=float, default=8.0)
    ap.add_argument("--greedy-stale-threshold-s", type=float, default=120.0)
    ap.add_argument(
        "--no-greedy-hysteresis",
        action="store_true",
        help="Disable capture burst: camera arms only on the raw trigger each step (choppier ON/OFF)",
    )
    ap.add_argument(
        "--greedy-stale-release-s",
        type=float,
        default=None,
        help="With hysteresis + unseen_or_stale: stop capturing when disk staleness is below this "
        "(default ~28%% of --greedy-stale-threshold-s, min 6s)",
    )
    ap.add_argument(
        "--greedy-mean-age-release-s",
        type=float,
        default=None,
        help="With hysteresis + mean_age: stop when disk mean age is below this (default ~42%% of mean-age threshold)",
    )
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="If set, write PNGs per session (dayNN_am_* / dayNN_pm_* when two sessions)",
    )
    ap.add_argument(
        "--png-metric-only",
        action="store_true",
        help="With --png-dir: skip floorplan overlay PNGs (only metre-grid heatmaps)",
    )
    ap.add_argument(
        "--animation-gif",
        type=Path,
        default=None,
        help="Write animated GIF: staleness on floorplan + pose dot through greedy captures",
    )
    ap.add_argument(
        "--animation-frame-ms",
        type=int,
        default=36,
        help="GIF frame duration in milliseconds (lower = faster playback)",
    )
    ap.add_argument(
        "--animation-vmax-age-sec",
        type=float,
        default=600.0,
        help="Staleness color scale max (seconds) for GIF frames; older = saturated red",
    )
    ap.add_argument(
        "--animation-dual-panel",
        action="store_true",
        help="GIF: [staleness | cumulative coverage] side by side (wider file)",
    )
    ap.add_argument(
        "--animation-interp-substeps",
        type=int,
        default=4,
        help="Blend staleness + pose between eligible frames (1 = one GIF frame per eligible, choppier)",
    )
    args = ap.parse_args()
    if args.rear_camera_on_alternate_days:
        args.rear_pi_on_reverse_timeline = True

    spd = max(1, int(args.sessions_per_day))
    if spd > 2:
        sys.exit("--sessions-per-day > 2 is not supported yet (use 1 or 2)")

    hfov = 400.0 if args.fov_full_360 else float(args.hfov_deg)
    vfov = 200.0 if args.fov_full_360 else float(args.vfov_deg)

    if not pc.POSITIONS_JSON.exists():
        sys.exit(f"Missing {pc.POSITIONS_JSON}")

    positions = pc.load_positions(pc.POSITIONS_JSON)
    frame_paths = sorted(pc.FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        sys.exit(f"No frames in {pc.FRAMES_DIR}")

    names = [p.name for p in frame_paths if p.name in positions]
    names.sort(key=lambda n: positions[n]["timestamp_sec"])
    if args.max_frames > 0:
        names = names[: args.max_frames]

    full_rows = pc.eligible_timeline_rows(positions, names)
    if not full_rows:
        sys.exit("No eligible frames (JPEG + depth)")

    baseline_frac, baseline_meta = _baseline_unconstrained_coverage(
        positions,
        full_rows,
        resolution_m=args.resolution,
        margin_m=args.margin,
        subsample=args.subsample,
        hfov_deg=hfov,
        vfov_deg=vfov,
        projection=args.projection,
    )

    times = [r[3] for r in full_rows]
    t_lo, t_hi = min(times), max(times)
    walk_span = float(t_hi - t_lo)
    if walk_span <= 0:
        walk_span = 0.5

    day_sec = float(args.seconds_per_day_gap)
    gap_sess = float(args.between_session_gap_sec)

    last_seen: np.ndarray | None = None
    days_report: list[dict] = []

    png_root = args.png_dir.expanduser().resolve() if args.png_dir is not None else None
    if png_root is not None:
        png_root.mkdir(parents=True, exist_ok=True)

    anim_gif_path = args.animation_gif.expanduser().resolve() if args.animation_gif else None
    anim_keyframes: list[tuple[np.ndarray, float, float, float, list[str]]] = []

    for d in range(args.days):
        if args.reset_coverage:
            last_seen = None

        day_sessions: list[dict] = []

        k_day = _k_target_calendar_day(float(args.budget_seconds), full_rows)
        k_left = k_day

        for s in range(spd):
            rows, hmap, rev_timeline = _rows_for_session(
                full_rows,
                sessions_per_day=spd,
                session_index_zero_based=s,
                path_direction=args.path_direction,
                calendar_day_index_zero_based=d,
            )

            h_off = 0.0
            if args.rear_camera:
                h_off += math.pi
            elif spd >= 2 and s % 2 == 1 and args.rear_pi_evening_session:
                h_off += math.pi
            elif spd <= 1 and args.rear_pi_on_reverse_timeline and rev_timeline:
                h_off += math.pi
            h_off = pc._wrap_angle_rad(h_off)

            frac_before = _frac_ever(last_seen) if last_seen is not None else 0.0

            ts_off = float(d) * day_sec + float(s) * (walk_span + gap_sess)

            if spd >= 2:
                greedy_tr = args.morning_greedy_trigger if s == 0 else args.evening_greedy_trigger
                k_at_session_start = k_left
                max_kept_session = k_left
            else:
                greedy_tr = args.greedy_trigger
                k_at_session_start = None
                max_kept_session = None

            budget_s = float(args.budget_seconds)

            sess_label = _session_label(s, spd)
            file_tag = _session_file_tag(s, spd)

            anim_tail = anim_gif_path is not None

            def _per_eligible(
                ls: np.ndarray,
                name: str,
                cx: float,
                cy: float,
                te: float,
                kept: bool,
            ) -> None:
                if anim_gif_path is None:
                    return
                c = _frac_ever(ls)
                pct = (100.0 * min(1.0, c / baseline_frac)) if baseline_frac > 1e-18 else 0.0
                anim_keyframes.append(
                    (
                        ls.copy(),
                        float(te),
                        float(cx),
                        float(cy),
                        [
                            f"Day {d + 1} {sess_label}",
                            f"{'SCAN' if kept else 'no scan'}  t={te:.1f}s",
                            f"coverage {c:.4f}  ({pct:.1f}% of always-on max)",
                        ],
                    )
                )

            sel, info, last_seen = pc.select_frames_greedy_local(
                rows,
                budget_s,
                resolution_m=args.resolution,
                margin_m=args.margin,
                subsample=args.subsample,
                hfov_deg=hfov,
                vfov_deg=vfov,
                projection=args.projection,
                heading_by_name=hmap,
                greedy_trigger=greedy_tr,
                greedy_unseen_scope=args.greedy_unseen_scope,
                greedy_radius_m=args.greedy_radius_m,
                greedy_mean_age_threshold_s=args.greedy_mean_age_threshold_s,
                greedy_stale_threshold_s=args.greedy_stale_threshold_s,
                greedy_hysteresis=not args.no_greedy_hysteresis,
                greedy_stale_release_s=args.greedy_stale_release_s,
                greedy_mean_age_release_s=args.greedy_mean_age_release_s,
                last_seen_initial=last_seen,
                heading_offset_rad=h_off,
                timestamp_day_offset_sec=ts_off,
                per_eligible_frame_cb=_per_eligible if anim_tail else None,
                walk_eligible_tail_after_budget=anim_tail,
                max_kept_frames=max_kept_session,
            )

            if spd >= 2:
                k_left = max(0, k_left - len(sel))

            t_end = max(r[3] for r in rows) + ts_off
            frac_after = _frac_ever(last_seen)
            cov_vs_base = (
                min(1.0, float(frac_after / baseline_frac)) if baseline_frac > 1e-18 else None
            )
            if png_root is not None:
                ever = np.isfinite(last_seen) & (last_seen > -np.inf)
                meta = pc.metadata_from_last_seen(
                    last_seen,
                    resolution_m=args.resolution,
                    margin_m=args.margin,
                    t_end=float(t_end),
                    hfov_deg=hfov,
                    vfov_deg=vfov,
                    projection=args.projection,
                )
                meta["sim_calendar_day"] = d + 1
                meta["session_index"] = s + 1
                meta["session_label"] = sess_label
                meta["greedy_trigger"] = greedy_tr
                meta["greedy_stale_threshold_s"] = float(args.greedy_stale_threshold_s)
                pref = f"day{d + 1:02d}_{file_tag}" if file_tag else f"day{d + 1:02d}_"
                if args.reset_coverage:
                    title_suffix = f" — day {d + 1} {sess_label} (coverage reset at day start)"
                else:
                    title_suffix = f" — cumulative after day {d + 1} {sess_label}"
                pc.save_outputs(
                    last_seen,
                    ever,
                    float(t_end),
                    meta,
                    overlay_alpha=0.52,
                    skip_floorplan_overlay=args.png_metric_only,
                    output_dir=png_root,
                    file_prefix=pref,
                    title_suffix=title_suffix,
                )

            day_sessions.append(
                {
                    "session_index": s + 1,
                    "label": sess_label,
                    "timeline_reversed": rev_timeline,
                    "path_reversed": bool(len(rows) and rows[0][0] != full_rows[0][0]),
                    "heading_offset_deg": round(float(h_off) * 180.0 / math.pi, 2),
                    "shared_daily_budget_seconds": budget_s,
                    "greedy_trigger": greedy_tr,
                    "timestamp_offset_sec": float(ts_off),
                    "max_kept_frames_at_session_start": int(k_at_session_start)
                    if k_at_session_start is not None
                    else None,
                    "captures_remaining_after_session": int(k_left) if spd >= 2 else None,
                    "selected_frame_count": len(sel),
                    "selected_first_last": [sel[0], sel[-1]] if sel else [],
                    "k_target": info.get("k_target"),
                    "rays_this_session": info.get("rays_accumulated", 0),
                    "frac_cells_ever_seen_before_session": float(frac_before),
                    "frac_cells_ever_seen_after_session": float(frac_after),
                    "coverage_fraction_of_baseline": cov_vs_base,
                    "baseline_frac_cells_ever_unconstrained": float(baseline_frac),
                    "had_coverage_before_session": frac_before > 0.0,
                    "t_end_map_age_sec": float(t_end),
                    "replay": {k: v for k, v in info.items() if k != "rays_accumulated"},
                }
            )

        days_report.append(
            {
                "day_index": d + 1,
                "k_target_calendar_day": int(k_day),
                "captures_remaining_end_of_calendar_day": int(k_left) if spd >= 2 else None,
                "sessions": day_sessions,
            }
        )

    final_frac = _frac_ever(last_seen) if last_seen is not None else 0.0
    final_vs_baseline = (
        min(1.0, float(final_frac / baseline_frac)) if baseline_frac > 1e-18 else None
    )

    out_payload = {
        "eligible_frames": len(full_rows),
        "baseline_unconstrained": {
            "frac_cells_ever_seen": float(baseline_frac),
            "frames_used": int(baseline_meta.get("frames_used", 0)),
            "rays_accumulated": int(baseline_meta.get("rays_accumulated", 0)),
            "definition": (
                "All eligible frames in chronological order, forward path, path-tangent headings, "
                "same FOV/projection/subsample as this run; no budget, no +pi offset "
                "(baseline is forward-only; evening +pi is not included)."
            ),
        },
        "coverage_fraction_of_baseline_after_sim": final_vs_baseline,
        "frac_cells_ever_seen_after_sim": float(final_frac),
        "walk_span_sec": walk_span,
        "sessions_per_day": spd,
        "budget_seconds_per_calendar_day": float(args.budget_seconds),
        "between_session_gap_sec": gap_sess if spd >= 2 else None,
        "morning_greedy_trigger": args.morning_greedy_trigger if spd >= 2 else None,
        "evening_greedy_trigger": args.evening_greedy_trigger if spd >= 2 else None,
        "path_direction": args.path_direction if spd <= 1 else None,
        "rear_pi_evening_session": bool(args.rear_pi_evening_session) if spd >= 2 else None,
        "rear_pi_on_reverse_timeline": bool(args.rear_pi_on_reverse_timeline) if spd <= 1 else None,
        "rear_camera_all_days": bool(args.rear_camera),
        "reset_coverage_each_day": bool(args.reset_coverage),
        "greedy_trigger_single_session": args.greedy_trigger if spd <= 1 else None,
        "greedy_stale_threshold_s": float(args.greedy_stale_threshold_s),
        "greedy_hysteresis": not bool(args.no_greedy_hysteresis),
        "greedy_stale_release_s_override": args.greedy_stale_release_s,
        "greedy_mean_age_release_s_override": args.greedy_mean_age_release_s,
        "fov": {
            "hfov_deg": float(hfov),
            "vfov_deg": float(vfov),
            "full_sphere_horizontal": bool(args.fov_full_360),
            "note": (
                "Default uses Ray-Ban-style horizontal masking (~100° unless overridden). "
                "Evening session +pi centers that cone opposite travel along the reversed path."
            ),
        },
        "png_dir": str(png_root) if png_root is not None else None,
        "png_metric_only": bool(args.png_metric_only),
        "animation_gif": str(anim_gif_path) if anim_gif_path is not None else None,
        "animation_frame_ms": int(args.animation_frame_ms),
        "animation_vmax_age_sec": float(args.animation_vmax_age_sec),
        "animation_dual_panel": bool(args.animation_dual_panel),
        "animation_interp_substeps": int(args.animation_interp_substeps),
        "days": days_report,
    }
    txt = json.dumps(out_payload, indent=2)
    print(txt)
    if args.json_out is not None:
        out = args.json_out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt, encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)
    if png_root is not None:
        print(f"Wrote PNGs under {png_root}", file=sys.stderr)
    if anim_gif_path is not None:
        anim_frames = pc.expand_revisit_staleness_keyframes_to_frames(
            anim_keyframes,
            int(args.animation_interp_substeps),
            resolution_m=args.resolution,
            margin_m=args.margin,
            overlay_alpha=0.52,
            vmax_age_s=float(args.animation_vmax_age_sec),
            include_coverage_panel=bool(args.animation_dual_panel),
            root=ROOT,
        )
        pc.save_animation_gif(anim_frames, anim_gif_path, frame_ms=int(args.animation_frame_ms))
        print(f"Wrote {anim_gif_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
