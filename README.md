# 61820_aam — 360° Courtyard Scan for Incremental Coverage Mapping

A research dataset and processing pipeline that maps a GPS-traced 360° walk through a courtyard to a georeferenced floorplan, enabling depth-map-based coverage reconstruction.

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract frames from the 360° video at 2 fps → outputs/frames/
python scripts/extract_frames.py

# 3. Match each frame to its real-world XY position on the floorplan → outputs/frame_positions.json
python scripts/sync_gps_video.py

# 4. Run depth estimation on every frame → outputs/depth_maps/
python scripts/estimate_depth.py

# 5. (Optional) Quick test without full pipeline — fetch a few assets from R2, then map-age heatmaps
python scripts/fetch_sample_frames.py --count 5
python scripts/project_coverage.py --max-frames 5 --resolution 0.25
```

All scripts are run from the repo root and use paths relative to it automatically.

**Tests:** `pytest` from repo root (uses synthetic frames; no GPU or R2 required).

---

## Data

| File / Folder | Description |
|---|---|
| `data/base_floorplan.jpg` | Architectural floorplan, 1400×819 px |
| `data/svg_path.svg` | Calibrated GPS walk path, 14 waypoints, same pixel coordinate space as floorplan |
| `data/floorplan_grid.png` | Floorplan with red dotted metric grid overlay (10 m major, 5 m minor) |
| `data/floorplan_walkpath.png` | Floorplan + grid + GPS walk path with numbered waypoints |
| `scripts/extract_frames.py` | Downloads video from R2 CDN (or uses local copy), extracts frames at 2 fps |
| `scripts/sync_gps_video.py` | Interpolates GPS waypoints across video timeline → `frame_positions.json` |
| `scripts/estimate_depth.py` | Runs Depth Anything V2 Small on all frames → grayscale depth PNGs |
| `scripts/generate_images.py` | Regenerates the floorplan grid/walkpath PNGs |
| `scripts/generate_pointcloud.py` | Builds a metric RGB point cloud from depth maps + frame positions |
| `scripts/courtyard_geom.py` | Shared equirectangular depth scaling + ground-plane rays (used by point cloud + coverage) |
| `scripts/project_coverage.py` | Projects depth onto z=0, accumulates **last observation time** per grid cell → map-age PNGs |
| `scripts/fetch_sample_frames.py` | Downloads first N frames + depth from R2 for quick local tests |
| `scripts/add_geo.py` | Adds lat/lon to all waypoints and frame positions using two GPS anchors |
| `data/geo_reference.json` | GPS anchors, floorplan bearing (61.56° CW from north), all 14 waypoint lat/lons |

**Floorplan scale:** `1400 px = 67 meters` → `1 px = 0.04786 m` / `20.896 px/m`

**Georeferencing:** The floorplan is anchored at `42.378878°N, 71.123678°W` (SVG path start). The +x axis points **61.56° clockwise from true north** (NE direction — the courtyard is not cardinal-aligned). All frame positions and waypoints in `frame_positions.json` and `geo_reference.json` include calibrated `lat`/`lon` fields. See `data/geo_reference.json` for full details.

**Raw 360° video** is hosted on Cloudflare R2 — do not commit it to git:
```
https://assets02.aitkena.com/courtyard_360/VID_20260429_143550_00_014.mp4
```
`scripts/extract_frames.py` downloads it automatically if no local copy is found.

> **Credentials:** Copy `.env.example` to `.env` and fill in your R2 keys if you need to re-upload assets. Never commit `.env`.

---

## Pre-computed Outputs

All pipeline outputs have been pre-computed and uploaded to Cloudflare R2. Collaborators can download them directly without re-running the pipeline.

| Asset | Public R2 URL |
|---|---|
| **Frames** (183 JPEGs, 2 fps) | `https://assets02.aitkena.com/courtyard_360/frames/frame_NNNNN_T.TTTs.jpg` |
| **Depth maps** (183 PNGs) | `https://assets02.aitkena.com/courtyard_360/depth_maps/frame_NNNNN_T.TTTs_depth.png` |
| **Frame positions JSON** | `https://assets02.aitkena.com/courtyard_360/frame_positions.json` |
| **Depth timelapse video** | `https://assets02.aitkena.com/courtyard_360/depth_timelapse.mp4` |
| **Point cloud (PLY)** | `https://assets02.aitkena.com/courtyard_360/pointcloud.ply` |
| **360° video** | `https://assets02.aitkena.com/courtyard_360/VID_20260429_143550_00_014.mp4` |

**Base URL:** `https://assets02.aitkena.com/courtyard_360/`

Example — fetch a specific frame and its matching depth map:
```
https://assets02.aitkena.com/courtyard_360/frames/frame_00000_0.000s.jpg
https://assets02.aitkena.com/courtyard_360/depth_maps/frame_00000_0.000s_depth.png
```

Download `frame_positions.json` to get the full list of filenames and their real-world coordinates:
```bash
curl -O https://assets02.aitkena.com/courtyard_360/frame_positions.json
```

---

## Key Output: `outputs/frame_positions.json`

`frame_positions.json` is the bridge between the video and the floorplan. It maps every extracted frame to a real-world position and timestamp:

```json
{
  "frame_00000_0.000s.jpg": {
    "x_px": 373.5,
    "y_px": 475.5,
    "x_meters": 17.873,
    "y_meters": 16.455,
    "timestamp_sec": 0.0
  },
  ...
}
```

With this file you can:
- **Project depth maps onto the floorplan** — for each frame, use its `(x_meters, y_meters)` as the camera origin and cast the depth map into top-down floorplan space
- **Assign map age** — `timestamp_sec` lets you colour-code each coverage patch by when it was observed
- **Build an incremental coverage heatmap** — accumulate projected patches across all frames to show what fraction of the courtyard was visible at each depth threshold

---

## Pipeline Overview

```
360° video (R2)
      │
      ▼
scripts/extract_frames.py  →  outputs/frames/  (183 frames at 2 fps)
      │
      ├──► scripts/sync_gps_video.py  →  outputs/frame_positions.json  (XY + timestamp per frame)
      │
      └──► scripts/estimate_depth.py  →  outputs/depth_maps/  (183 grayscale depth PNGs)
                                                  │
                                                  ▼
                        scripts/project_coverage.py  →  outputs/coverage/
                              map_age_end.png, coverage_ever.png,
                              map_age_on_floorplan.png, coverage_on_floorplan.png,
                              last_seen_sec.npy, coverage_meta.json
```

`project_coverage.py` uses the same depth metric scaling and equirectangular rays as `generate_pointcloud.py`. **Default `--projection backproject`:** each ray’s depth hits a 3D point `P = C + d u`; **(P_x, P_y)** marks the map cell (includes façades, poles, ground — loose z/depth bounds only drop junk). **`--projection ground_plane`:** intersection with **z = 0** plus an occlusion test (stricter floor-only). Writes **seconds since last observation** per cell at end of walk (`RdYlGn_r`: green = fresh, red = stale).

**FOV masking (wearable-style forward cone):** Rays are filtered in the **camera frame** before ground projection: horizontal angle `atan2(dy, dx)` from forward **+x** (equirect `theta=0` on the horizon) and vertical angle `asin(dz)` from the horizontal plane. **Horizontal default 100°** follows **Ray-Ban Meta (Gen 2)** ultra-wide capture (Meta shooting guides). **Vertical default 179°** leaves elevation almost unrestricted: a tight symmetric vertical FOV (e.g. 100°) removes nearly all equirect rays that intersect the ground, because ground appears at steep elevations in this projection. Use `--vfov-deg` to tighten if you switch to a forward pinhole crop. `--fov-full-360` disables horizontal masking too (full sphere).

**Floorplan overlays:** If `data/base_floorplan.jpg`, `data/floorplan_walkpath.png`, or `data/floorplan_grid.png` exists (1400×819 or resized to match), heatmaps are blended on top. Otherwise the script draws the SVG walk path on a neutral background. Tune blend with `--overlay-alpha` (default 0.52).

**Full walk:** `python scripts/fetch_sample_frames.py --all` then `python scripts/project_coverage.py`.

**Budget replay (limited “camera on” time):** Pick a subset of frames whose count matches `floor(budget_seconds / mean_dt)` where `mean_dt` is the mean spacing between consecutive **eligible** frames (JPEG + depth on disk). Headings are always taken from the **full** eligible timeline so FOV rays stay aligned with path tangent.

- **Default policy `greedy_local`:** walk the timeline in order; **capture hysteresis** (default on) keeps the camera **on in bursts**: after the trigger fires, frames stay selected until the **local disk** (`--greedy-radius-m`) is fully covered and staleness there drops below an auto **release** threshold (or mean-age release for `mean_age`). That avoids one-frame ON/OFF flicker when a merge instantly clears the trigger. Use **`--no-greedy-hysteresis`** for legacy stepping. Triggers: default **`--greedy-trigger unseen --greedy-unseen-scope foot`** (foot cell never hit by a kept frame); **`--greedy-unseen-scope disk`** = any unseen cell within radius. **`--greedy-trigger unseen_or_stale`** also fires when seen cells in scope exceed **`--greedy-stale-threshold-s`**. Tune release with **`--greedy-stale-release-s`** / **`--greedy-mean-age-release-s`** if needed.
- **Baselines:** `random`, `uniform` (evenly spaced indices), `prefix` (first K frames). Random uses `--replay-seed`.

**Greedy unseen (default out dir):**

```bash
python scripts/project_coverage.py --fov-full-360 --budget-seconds 18
```

Writes e.g. `outputs/coverage_replays/budget_18s_greedy_local_unseen_foot/`.

**Random baseline (seed in folder name):**

```bash
python scripts/project_coverage.py --fov-full-360 --budget-seconds 18 --replay-policy random --replay-seed 0
```

Writes under `outputs/coverage_replays/budget_18s_greedy_local_unseen_foot/` (default greedy unseen) or `budget_18s_random_seed0/`, or `--out-dir PATH`. Omit `--budget-seconds` (or `0`) for all eligible frames → `outputs/coverage/`.

**Multi-day revisit simulation** (same physical path; **coverage grid** carries unless **`--reset-coverage`** clears it at each **calendar midnight** before morning):

```bash
python scripts/simulate_revisit_days.py --days 4 --budget-seconds 18
python scripts/simulate_revisit_days.py --sessions-per-day 1 --path-direction alternate
python scripts/simulate_revisit_days.py --reset-coverage --json-out outputs/revisit_sim.json
```

- **Default `--sessions-per-day 2` (morning / evening):** each **calendar day** has a **forward** walk (morning) and a **backward** walk (evening, rear 360° half via **+pi** on rays unless **`--no-rear-evening-session`**). Sim clock advances by one **walk span** (+ optional **`--between-session-gap-sec`**) between morning and evening so evening runs **later the same day** than morning (avoids treating morning hits as “instantly stale” at the start of the backward pass).
- **Budget:** **`--budget-seconds`** is the **daily** total shared by morning and evening: one **`k_target`** for the day (`≈ floor(budget / mean_dt)`); morning uses captures first, evening gets what is left. **Morning** default **`--morning-greedy-trigger unseen`**; **evening** default **`--evening-greedy-trigger unseen_or_stale`**. Use **`--morning-greedy-trigger unseen_or_stale --evening-greedy-trigger unseen_or_stale`** for the legacy “same trigger all day” behaviour.
- **`--sessions-per-day 1`:** one walk per calendar day; use **`--path-direction`** (`forward` / `reverse` / `alternate`) and **`--greedy-trigger`**. Rear on reversed days: **`--no-rear-on-reverse-days`** to disable +pi when the single session is backward.
- **`--rear-camera`:** +pi on **every** session.
- **`--seconds-per-day-gap`:** sim seconds between calendar midnights (default 86400).
- **`--png-dir`:** one PNG set **per session** when using two sessions: **`dayNN_am_*`** and **`dayNN_pm_*`** (plus `last_seen` / meta per write).
- **`--animation-gif`:** GIF with pose along the **full** eligible path each session (**SCAN** vs **no scan**); default **`--animation-frame-ms`** **36** (faster playback). Staleness is blended between eligibles via **`--animation-interp-substeps`**. **`--animation-vmax-age-sec`**, **`--animation-dual-panel`** as before.
- **Baseline:** JSON **`baseline_unconstrained`** and **`coverage_fraction_of_baseline_after_sim`** vs always-on all frames (chronological, same FOV).
- JSON: nested **`days[].sessions[]`** with per-session replay stats.

---

## Next Steps

- [x] `project_coverage.py` — floorplane projection + map-age / ever-seen heatmaps (`outputs/coverage/`)
- [x] Per-frame FOV masking — `--hfov-deg` / `--vfov-deg` (default ~100° Ray-Ban Meta style), or `--fov-full-360`
- [ ] Coverage metric: fraction of courtyard area observed at each depth threshold
- [ ] **Temporal scan-order visuals** — The current heatmap encodes **staleness at the end of the walk** (time since *last* hit per cell), not *first* hit time or a time-lapse. A separate “temporal coverage” item would add e.g. first-seen timestamp colouring or an animated GIF over `timestamp_sec`.

---

## Requirements

```
numpy
opencv-python
Pillow
torch
transformers
huggingface_hub
requests
```

Install: `pip install -r requirements.txt`
