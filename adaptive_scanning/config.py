from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from adaptive_scanning.street_trajectories import MIT_CAMPUS_BBOX_WGS84


@dataclass
class AdaptiveScanningConfig:
    """Defaults: MIT-campus OSM walks + home-commute style paths; FOV ~100° (Ray-Ban ultra-wide class)."""

    # Grid (metres per cell = resolution_m). 64×64 @ 2 m → 128 m env window, 4096 cells (paths affine-mapped into this rectangle).
    nx: int = 64
    ny: int = 64
    resolution_m: float = 2.0

    # Sector sensor (horizontal wedge in the xy plane)
    hfov_deg: float = 100.0
    scan_radius_m: float = 30.0

    # Sim decision interval: one action (camera on/off) and one trajectory sample per dt_s.
    # 30 s → fewer env steps per max_sim_time_s (faster rollouts; coarser control).
    dt_s: float = 30.0
    day_duration_s: float = 8 * 3600.0  # one "walking day" in sim time
    # ~5 min/day scanning budget (SI seconds of effective camera-on per day_duration_s window).
    seconds_video_budget_per_day: float = 300.0

    # Episode length in simulated time (budget resets each day_duration_s within this span)
    max_sim_time_s: float = 4 * 8 * 3600.0  # 4×8 h walking days (matches MIT home-commute demos)

    # Reward weights (negative reward = cost minimization in RL)
    w_uncovered: float = 1.0
    w_stale_scanned: float = 0.5
    stale_ref_s: float = 3600.0  # normalize ages for observation / local stale mean cap
    # Added to reward each step the camera is effectively on (RL bootstrap; 0 = pure coverage objective).
    reward_camera_on_bonus: float = 0.0

    # Local egocentric patch (cells); must be odd
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

    # Random motion (box mode)
    walk_speed_m_s: float = 1.35  # ~4.9 km/h typical adult walking
    walk_heading_noise_rad: float = 0.15
