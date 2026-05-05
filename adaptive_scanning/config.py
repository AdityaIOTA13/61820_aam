from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

from adaptive_scanning.street_trajectories import MIT_CAMPUS_BBOX_WGS84


def config_from_saved_dict(saved: dict[str, Any]) -> AdaptiveScanningConfig:
    """
    Rebuild ``AdaptiveScanningConfig`` from a checkpoint ``cfg`` mapping.

    Drops keys that no longer exist on the dataclass (forward compat for old .pt files) and
    applies defaults for any field missing from the save (backward compat after new fields).
    """
    valid = {f.name for f in fields(AdaptiveScanningConfig)}
    kw: dict[str, Any] = {}
    for k, v in saved.items():
        if k not in valid:
            continue
        if k == "osm_bbox" and v is not None and isinstance(v, list):
            v = tuple(float(x) for x in v)
        kw[k] = v
    return AdaptiveScanningConfig(**kw)


@dataclass
class AdaptiveScanningConfig:
    """Defaults: MIT-campus OSM walks + home-commute style paths; FOV ~100° (Ray-Ban ultra-wide class)."""

    # Grid (metres per cell = resolution_m). 64×64 @ 2 m → 128 m env window, 4096 cells (paths affine-mapped into this rectangle).
    nx: int = 64
    ny: int = 64
    resolution_m: float = 2.0

    # Sector sensor (horizontal wedge in the xy plane). Radius is in ground / graph metres
    # (same as Folium wedge geometry); street episodes letterbox graph into the env grid, and
    # the env applies the matching scale so sector coverage matches that radius on the map.
    hfov_deg: float = 100.0
    scan_radius_m: float = 30.0

    # Sim decision interval: one action (camera on/off) and one trajectory sample per dt_s.
    # 30 s → fewer env steps per max_sim_time_s (faster rollouts; coarser control).
    dt_s: float = 30.0
    day_duration_s: float = 8 * 3600.0  # one "walking day" in sim time
    # SI seconds of effective camera-on budget each time the daily budget resets.
    # Interpreted at ``video_budget_reference_walk_speed_m_s`` (default equals ``walk_speed_m_s`` default):
    # effective budget is ``seconds_video_budget_per_day * (reference / walk_speed_m_s)`` so that
    # ``walk_speed * budget_seconds`` (order-of-magnitude path metres while recording) stays stable
    # when you change walking speed; set reference to 0 to keep a fixed SI second cap regardless of speed.
    seconds_video_budget_per_day: float = 600.0  # 10 min/day camera-on budget (SI seconds)
    video_budget_reference_walk_speed_m_s: float = 1.35

    # Episode length in simulated time (budget resets each day_duration_s within this span)
    max_sim_time_s: float = 6 * 8 * 3600.0  # 6×8 h walking days (MIT home-commute style demos)

    # Reward weights (negative reward = cost minimization in RL)
    w_uncovered: float = 1.0
    w_stale_scanned: float = 0.5
    stale_ref_s: float = 3600.0  # normalize ages for observation / local stale mean cap
    # Added to reward each step the camera is effectively on (RL bootstrap; 0 = pure coverage objective).
    reward_camera_on_bonus: float = 0.0
    # At each simulated day boundary: subtract ``w_unused_budget_end_of_day * (unused_budget / daily_cap_s)``;
    # ``daily_cap_s`` matches the env (includes walk-speed scaling when reference speed > 0).
    # Unused fraction is leftover budget before the daily reset (0 = disable).
    w_unused_budget_end_of_day: float = 0.1

    # If True, sector scans only write ``last_seen`` on cells that were never finite before (first hit).
    # Revisits do not refresh timestamps (reduces meaningless "rescan" map churn). Default False for RL.
    update_last_seen_only_on_first_hit: bool = False

    # RL observation mode:
    # - "patch": 2-channel local patch (ever-scanned + age) + global features
    # - "foot_cell": current-cell scan state/age features (greedy-unseen-like) + global features
    observation_mode: Literal["patch", "foot_cell"] = "foot_cell"

    # Local egocentric patch (cells); must be odd (used when observation_mode="patch")
    patch_cells: int = 31

    # Motion: ``streets`` = OSM shortest-path style walks; ``box`` = random rectangle (use e.g. --fast CLI).
    motion_mode: Literal["box", "streets"] = "streets"
    # OSM (when motion_mode == "streets"). Default: MIT main campus bbox (no geocoder place query).
    osm_place: str = ""
    osm_bbox: tuple[float, float, float, float] | None = MIT_CAMPUS_BBOX_WGS84
    osm_cache_dir: str = "outputs/adaptive_scanning/osm_cache"
    osm_network_type: str = "walk"
    osm_anchor_nodes: int = 28
    osm_anchor_reuse_bias: float = 0.72  # prob. OD from anchor set → overlapping corridors
    # If True (streets only): one shortest-path trip start→end, then resample in time (no chaining)
    osm_single_leg: bool = False
    # If True (streets only): each calendar day is ``osm_walks_per_day`` chained shortest paths:
    # start at **home**, each leg starts where the previous ended, last leg returns to **home**.
    # One home for the whole episode (unused but kept for CLI / pickle compatibility).
    osm_daily_home_commute: bool = True
    osm_walks_per_day: int = 3
    osm_same_home_next_day_p: float = 0.6
    # When building day 2+, each intermediate stop is chosen from prior days' stops with this probability
    # (otherwise a new random node); home stays fixed for the episode.
    osm_repeat_destination_across_days_p: float = 0.35
    # When a prior stop is reused, sample stops with weight ~ recency^(days since that stop's day).
    # 1.0 = uniform over all prior stops (strong day-1 vs day-3 corridor overlap). Values below 1
    # favor yesterday's destinations and reduce long-gap geographic reuse in multi-day episodes.
    osm_repeat_prior_stops_recency: float = 0.55

    # Late-day leniency for ``_foot_cell_greedy_unseen_on`` (greedy-unseen heuristic): after this
    # many SI seconds since the **start of the current simulated day** (``env._day_start_s``), ramp up
    # foot grace and wedge suppress age so mildly stale cells are likelier to get camera-on. This uses
    # only elapsed time since day start — no dependence on ``day_duration_s`` or “how long the day is”.
    # Set to a negative value to disable.
    evening_lenient_after_s_since_day_start: float = 5.5 * 3600.0
    # Linear ramp: leniency weight goes from 0→1 over this many seconds after ``evening_lenient_after_*``.
    evening_lenient_ramp_s: float = 3600.0
    evening_lenient_wedge_suppress_age_mult: float = 2.25
    evening_lenient_foot_grace_mult: float = 4.0
    # Only blend in leniency when this much daily video budget remains (fraction in [0, 1]).
    evening_lenient_min_budget_frac: float = 0.12

    # For ``BudgetAwareGreedyUnseenOnlyPolicy``: foot cell counts as already covered only if
    # ``_last_seen_foot`` is older than this many SI seconds before ``sim_time_s``. Fresher foot
    # stamps are ignored for the OFF decision.
    greedy_unseen_coverage_grace_seconds: float = 2.0
    # Wedge ``last_seen`` on the foot cell suppresses the camera only if this age (sim minus stamp)
    # exceeds this minimum (seconds). Too small a value causes gaps: with ``update_last_seen_only_on_first_hit``
    # or short camera-off runs, ``last_seen`` stops refreshing while ``sim_time`` advances, so cells read
    # as ``old`` and greedy stays OFF. Use ~1 h so same-day motion is not mistaken for a multi-day revisit.
    greedy_unseen_wedge_suppress_min_age_s: float = 3600.0

    # Random motion (box mode)
    walk_speed_m_s: float = 1.35  # ~4.9 km/h typical adult walking
    walk_heading_noise_rad: float = 0.15
