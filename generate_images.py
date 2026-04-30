"""
Generate floorplan grid image and GPS walk path overlay.
Scale: 1400px = 67 meters (0.04786 m/px)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from PIL import Image

# --- Scale constants ---
IMG_W_PX = 1400
IMG_H_PX = 819
REAL_W_M = 67.0
PX_PER_M = IMG_W_PX / REAL_W_M          # 20.896 px/m
REAL_H_M = IMG_H_PX / PX_PER_M          # ≈ 39.2 m

# --- SVG path points (parsed from svg_path.svg) ---
# Path: M373.5 475.5 H1126.5 L1180 503.5 L1193 554.5 L1180 604 L1133 635
#       H1015 H901.5 H796.5 L697.5 626 L601 602.5 L498 563.5 L415 518.5 L368.5 483.5
SVG_PTS_PX = [
    (373.5,  475.5),
    (1126.5, 475.5),
    (1180.0, 503.5),
    (1193.0, 554.5),
    (1180.0, 604.0),
    (1133.0, 635.0),
    (1015.0, 635.0),
    (901.5,  635.0),
    (796.5,  635.0),
    (697.5,  626.0),
    (601.0,  602.5),
    (498.0,  563.5),
    (415.0,  518.5),
    (368.5,  483.5),
]

def px_to_m(pts_px):
    """Convert pixel coords (SVG origin top-left) to meters (origin bottom-left)."""
    return [(x / PX_PER_M, (IMG_H_PX - y) / PX_PER_M) for x, y in pts_px]

SVG_PTS_M = px_to_m(SVG_PTS_PX)

# Load floorplan
floorplan = Image.open("data/base_floorplan.jpg").convert("RGB")
fp_arr = np.array(floorplan)

# Grid tick spacing
MAJOR_TICK_M = 10
MINOR_TICK_M = 5

DPI = 150
fig_w = IMG_W_PX / DPI
fig_h = IMG_H_PX / DPI

# ── IMAGE 1: Floorplan + grid ─────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax1.imshow(fp_arr, extent=[0, REAL_W_M, 0, REAL_H_M], origin="upper", aspect="equal")

ax1.set_xlim(0, REAL_W_M)
ax1.set_ylim(0, REAL_H_M)
ax1.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_M))
ax1.yaxis.set_major_locator(MultipleLocator(MAJOR_TICK_M))
ax1.xaxis.set_minor_locator(MultipleLocator(MINOR_TICK_M))
ax1.yaxis.set_minor_locator(MultipleLocator(MINOR_TICK_M))

ax1.grid(which="major", color="white", linewidth=1.8, alpha=0.95, linestyle="-")
ax1.grid(which="minor", color="white", linewidth=0.7, alpha=0.55, linestyle="--")

ax1.set_xlabel("X (meters)", fontsize=10, color="white", labelpad=4)
ax1.set_ylabel("Y (meters)", fontsize=10, color="white", labelpad=4)
ax1.tick_params(colors="white", labelsize=8)
for spine in ax1.spines.values():
    spine.set_edgecolor("white")

ax1.set_facecolor("black")
fig1.patch.set_facecolor("black")

fig1.savefig("data/floorplan_grid.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
plt.close(fig1)
print("Saved: data/floorplan_grid.png")

# ── IMAGE 2: Floorplan + grid + GPS walk path ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax2.imshow(fp_arr, extent=[0, REAL_W_M, 0, REAL_H_M], origin="upper", aspect="equal")

ax2.set_xlim(0, REAL_W_M)
ax2.set_ylim(0, REAL_H_M)
ax2.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_M))
ax2.yaxis.set_major_locator(MultipleLocator(MAJOR_TICK_M))
ax2.xaxis.set_minor_locator(MultipleLocator(MINOR_TICK_M))
ax2.yaxis.set_minor_locator(MultipleLocator(MINOR_TICK_M))

ax2.grid(which="major", color="white", linewidth=0.9, alpha=0.65, linestyle="--")
ax2.grid(which="minor", color="white", linewidth=0.4, alpha=0.35, linestyle=":")

# Draw GPS walk path
xs_m = [p[0] for p in SVG_PTS_M]
ys_m = [p[1] for p in SVG_PTS_M]
ax2.plot(xs_m, ys_m, color="#0C8CE9", linewidth=2.5, zorder=5, label="GPS walk path")

# Mark start and end
ax2.scatter([xs_m[0]], [ys_m[0]],  color="#00FF88", s=80, zorder=6, label="Start")
ax2.scatter([xs_m[-1]], [ys_m[-1]], color="#FF4444", s=80, zorder=6, label="End")

# Annotate waypoints
for i, (x, y) in enumerate(SVG_PTS_M):
    ax2.annotate(str(i+1), (x, y), fontsize=6.5, color="yellow",
                 xytext=(4, 4), textcoords="offset points", zorder=7)

ax2.set_xlabel("X (meters)", fontsize=10, color="white", labelpad=4)
ax2.set_ylabel("Y (meters)", fontsize=10, color="white", labelpad=4)
ax2.tick_params(colors="white", labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor("white")

legend = ax2.legend(loc="upper left", fontsize=8, facecolor="#111111",
                    edgecolor="white", labelcolor="white")

ax2.set_facecolor("black")
fig2.patch.set_facecolor("black")

fig2.savefig("data/floorplan_walkpath.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
plt.close(fig2)
print("Saved: data/floorplan_walkpath.png")
print(f"\nScale info: {IMG_W_PX}px = {REAL_W_M}m  →  {PX_PER_M:.3f} px/m")
print(f"Canvas: {REAL_W_M:.1f}m × {REAL_H_M:.2f}m")
print(f"Walk path has {len(SVG_PTS_PX)} waypoints")
