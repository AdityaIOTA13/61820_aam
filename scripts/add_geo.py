"""
add_geo.py

Adds GPS lat/lon coordinates to all SVG waypoints and frame positions
using two real-world anchor points to calibrate both scale and bearing.

Anchor 1 — SVG path start pixel (373.5, 475.5):
    42°22'43.96"N  71°07'25.24"W

Anchor 2 — on first straight horizontal segment (y=475.5, early x):
    42°22'44.14"N  71°07'24.79"W

The vector between them gives the real-world bearing of the SVG +x axis
(the courtyard is not aligned to cardinal directions). All pixel positions
are rotated by this bearing before converting to lat/lon.

Outputs
-------
data/geo_reference.json       — anchors, bearing, all 14 waypoint lat/lons
outputs/frame_positions.json  — updated in-place with lat/lon per frame

Run from repo root:  python scripts/add_geo.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent

POSITIONS_JSON  = ROOT / "outputs" / "frame_positions.json"
GEO_REF_JSON    = ROOT / "data" / "geo_reference.json"

# ── Floorplan scale ────────────────────────────────────────────────────────────
IMG_W_PX = 1400
REAL_W_M = 67.0
PX_PER_M = IMG_W_PX / REAL_W_M   # 20.896 px/m

# ── Anchor 1: SVG path start pixel ────────────────────────────────────────────
ANCHOR_PX   = (373.5, 475.5)
ANCHOR_LAT  = 42 + 22/60 + 43.96/3600   # 42.378878°N
ANCHOR_LON  = -(71 + 7/60 + 25.24/3600) # -71.123678°W

# ── Anchor 2: on first straight segment ───────────────────────────────────────
ANCHOR2_LAT = 42 + 22/60 + 44.14/3600   # 42.378928°N
ANCHOR2_LON = -(71 + 7/60 + 24.79/3600) # -71.123553°W

# ── Earth constants at this latitude ──────────────────────────────────────────
M_PER_DEG_LAT = 111_319.0
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(ANCHOR_LAT))  # ≈ 82,234 m/°

# ── Compute bearing of SVG +x axis from real-world displacement ───────────────
delta_north = (ANCHOR2_LAT - ANCHOR_LAT) * M_PER_DEG_LAT   # metres north
delta_east  = (ANCHOR2_LON - ANCHOR_LON) * M_PER_DEG_LON   # metres east

# Bearing of SVG +x direction: clockwise from north
SVG_X_BEARING_RAD = math.atan2(delta_east, delta_north)     # radians from north
SVG_X_BEARING_DEG = math.degrees(SVG_X_BEARING_RAD)         # ≈ 61.6°

print(f"Anchor 1:  {ANCHOR_LAT:.6f}°N  {ANCHOR_LON:.6f}°W  (SVG pixel {ANCHOR_PX})")
print(f"Anchor 2:  {ANCHOR2_LAT:.6f}°N  {ANCHOR2_LON:.6f}°W")
print(f"Δnorth={delta_north:.2f}m  Δeast={delta_east:.2f}m")
print(f"SVG +x bearing: {SVG_X_BEARING_DEG:.2f}° clockwise from north\n")


# ── Core conversion ───────────────────────────────────────────────────────────

def pixel_to_latlon(x_px: float, y_px: float) -> tuple[float, float]:
    """
    Convert SVG pixel (x_px, y_px) to (lat, lon) using the two-anchor calibration.
    SVG +x is east-ish (bearing SVG_X_BEARING_RAD from north, CW).
    SVG +y is down in pixel space → -y in world space (inverted).
    """
    dx_svg = (x_px - ANCHOR_PX[0]) / PX_PER_M   # metres along SVG +x
    dy_svg = -(y_px - ANCHOR_PX[1]) / PX_PER_M  # metres along SVG +y (flip)

    # SVG +x direction in (east, north): (sin(bearing), cos(bearing))
    sin_b, cos_b = math.sin(SVG_X_BEARING_RAD), math.cos(SVG_X_BEARING_RAD)
    # SVG +y direction (perpendicular, CCW from +x): (-cos(bearing), sin(bearing))

    d_east  = dx_svg * sin_b + dy_svg * (-cos_b)
    d_north = dx_svg * cos_b + dy_svg * sin_b

    lat = ANCHOR_LAT + d_north / M_PER_DEG_LAT
    lon = ANCHOR_LON + d_east  / M_PER_DEG_LON
    return round(lat, 8), round(lon, 8)


# ── SVG waypoints ─────────────────────────────────────────────────────────────
SVG_WAYPOINTS_PX = [
    (373.5, 475.5), (1126.5, 475.5), (1180.0, 503.5), (1193.0, 554.5),
    (1180.0, 604.0), (1133.0, 635.0), (1015.0, 635.0), (901.5,  635.0),
    (796.5,  635.0), (697.5,  626.0), (601.0,  602.5), (498.0,  563.5),
    (415.0,  518.5), (368.5,  483.5),
]

waypoints_geo = []
for i, (x, y) in enumerate(SVG_WAYPOINTS_PX, 1):
    lat, lon = pixel_to_latlon(x, y)
    waypoints_geo.append({"id": i, "x_px": x, "y_px": y, "lat": lat, "lon": lon})
    print(f"  WP{i:>2}: ({x:.1f},{y:.1f}) → {lat:.6f}°N  {lon:.6f}°W")

# ── geo_reference.json ────────────────────────────────────────────────────────
geo_ref = {
    "description": "Georeferencing data for the 61820_aam courtyard scan floorplan",
    "floorplan_scale": {
        "width_px": IMG_W_PX,
        "width_meters": REAL_W_M,
        "px_per_meter": round(PX_PER_M, 4),
    },
    "anchors": [
        {
            "label": "SVG path start (waypoint 1)",
            "svg_pixel": list(ANCHOR_PX),
            "lat": ANCHOR_LAT,
            "lon": ANCHOR_LON,
        },
        {
            "label": "GPS point on first straight segment",
            "lat": ANCHOR2_LAT,
            "lon": ANCHOR2_LON,
        },
    ],
    "svg_x_bearing_deg": round(SVG_X_BEARING_DEG, 2),
    "svg_x_bearing_note": "Degrees clockwise from true north. The floorplan +x axis points NE, not due east.",
    "coordinate_system": {
        "origin": "SVG anchor pixel (373.5, 475.5) = WGS84 anchor lat/lon",
        "x_axis": f"SVG +x (pixel right) → bearing {SVG_X_BEARING_DEG:.1f}° CW from north",
        "y_axis": "SVG +y (pixel down) → opposite direction (SW-ish)",
        "z_axis": "Up from floor, metres",
    },
    "waypoints": waypoints_geo,
}

with open(GEO_REF_JSON, "w") as f:
    json.dump(geo_ref, f, indent=2)
print(f"\nSaved: {GEO_REF_JSON.relative_to(ROOT)}")

# ── Update frame_positions.json ───────────────────────────────────────────────
with open(POSITIONS_JSON) as f:
    positions = json.load(f)

for fname, entry in positions.items():
    lat, lon = pixel_to_latlon(entry["x_px"], entry["y_px"])
    entry["lat"] = lat
    entry["lon"] = lon

with open(POSITIONS_JSON, "w") as f:
    json.dump(positions, f, indent=2)

print(f"Updated: {POSITIONS_JSON.relative_to(ROOT)}  ({len(positions)} frames with lat/lon)")
lats = [v["lat"] for v in positions.values()]
lons = [v["lon"] for v in positions.values()]
print(f"  Lat range: {min(lats):.6f}° – {max(lats):.6f}°N")
print(f"  Lon range: {min(lons):.6f}° – {max(lons):.6f}°W")
