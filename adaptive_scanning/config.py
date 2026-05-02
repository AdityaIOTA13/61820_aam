from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class AdaptiveScanningConfig:
    """Defaults aligned with Meta Ray-Ban ultra-wide horizontal FOV (~100°)."""

    # Grid (metres per cell = resolution_m)
    nx: int = 64
    ny: int = 64
    resolution_m: float = 2.0

    # Sector sensor (horizontal wedge in the xy plane)
    hfov_deg: float = 100.0
    scan_radius_m: float = 30.0

    # Time
    dt_s: float = 5.0
    day_duration_s: float = 8 * 3600.0  # one "walking day" in sim time
    seconds_video_budget_per_day: float = 120.0

    # Episode: multi-day = long sim horizon with budget resets
    max_sim_time_s: float = 3 * 8 * 3600.0  # 3 walking days

    # Reward weights (negative reward = cost minimization in RL)
    w_uncovered: float = 1.0
    w_stale_scanned: float = 0.5
    stale_ref_s: float = 3600.0  # normalize ages for observation / local stale mean cap

    # Local egocentric patch (cells); must be odd
    patch_cells: int = 31

    # Motion: open box random walk vs OSM street routing
    motion_mode: Literal["box", "streets"] = "box"
    # OSM (used when motion_mode == "streets"); set place OR bbox (west, south, east, north) WGS84.
    # If both empty, the env uses ``DEFAULT_OSM_PLACE`` (Cambridge, MA) from street_trajectories.
    osm_place: str = ""
    osm_bbox: tuple[float, float, float, float] | None = None
    osm_cache_dir: str = "outputs/adaptive_scanning/osm_cache"
    osm_network_type: str = "walk"
    osm_anchor_nodes: int = 28
    osm_anchor_reuse_bias: float = 0.72  # prob. OD from anchor set → overlapping corridors
    # If True (streets only): one shortest-path trip start→end, then resample in time (no chaining)
    osm_single_leg: bool = False

    # Random motion (box mode)
    walk_speed_m_s: float = 1.2
    walk_heading_noise_rad: float = 0.15
