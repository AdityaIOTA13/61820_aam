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
```

All scripts are run from the repo root and use paths relative to it automatically.

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

**Floorplan scale:** `1400 px = 67 meters` → `1 px = 0.04786 m` / `20.896 px/m`

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
                                   [next] project_coverage.py  →  coverage heatmap on floorplan
```

---

## Next Steps

- [ ] `project_coverage.py` — project each depth map onto the floorplan using `frame_positions.json`; accumulate into a coverage heatmap
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
