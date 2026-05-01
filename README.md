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

`project_coverage.py` uses the same depth metric scaling and equirectangular rays as `generate_pointcloud.py`, intersects rays with the ground plane (occlusion: depth shorter than ground range discards a pixel), and writes **seconds since last observation** per cell at end of walk (`RdYlGn_r`: green = fresh, red = stale).

**Floorplan overlays:** If `data/base_floorplan.jpg`, `data/floorplan_walkpath.png`, or `data/floorplan_grid.png` exists (1400×819 or resized to match), heatmaps are blended on top. Otherwise the script draws the SVG walk path on a neutral background. Tune blend with `--overlay-alpha` (default 0.52).

**Full walk:** `python scripts/fetch_sample_frames.py --all` then `python scripts/project_coverage.py`.

---

## Next Steps

- [x] `project_coverage.py` — floorplane projection + map-age / ever-seen heatmaps (`outputs/coverage/`)
- [ ] Per-frame visible angle masking (equirectangular → top-down projection accounting for FOV)
- [ ] Coverage metric: fraction of courtyard area observed at each depth threshold
- [ ] Temporal coverage: colour patches by `timestamp_sec` to show scan order

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
