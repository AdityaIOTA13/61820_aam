"""
generate_images.py

Generates two annotated PNGs from the floorplan and GPS walk path SVG:
  data/floorplan_grid.png      — floorplan + metric grid overlay
  data/floorplan_walkpath.png  — floorplan + grid + GPS walk path

Run from repo root:  python scripts/generate_images.py
"""

import re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent

IMG_W_PX = 1400
IMG_H_PX = 819
REAL_W_M = 67.0
PX_PER_M = IMG_W_PX / REAL_W_M
REAL_H_M = IMG_H_PX / PX_PER_M

SVG_PTS_PX = [
    (373.5, 475.5), (1126.5, 475.5), (1180.0, 503.5), (1193.0, 554.5),
    (1180.0, 604.0), (1133.0, 635.0), (1015.0, 635.0), (901.5, 635.0),
    (796.5, 635.0), (697.5, 626.0), (601.0, 602.5), (498.0, 563.5),
    (415.0, 518.5), (368.5, 483.5),
]
SVG_PTS_M = [(x / PX_PER_M, (IMG_H_PX - y) / PX_PER_M) for x, y in SVG_PTS_PX]

floorplan = Image.open(ROOT / "data" / "base_floorplan.jpg").convert("RGB")
fp_arr = np.array(floorplan)

DPI = 150
fig_w, fig_h = IMG_W_PX / DPI, IMG_H_PX / DPI


def make_axes(fig_w, fig_h):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(fp_arr, extent=[0, REAL_W_M, 0, REAL_H_M], origin="upper", aspect="equal")
    ax.set_xlim(0, REAL_W_M)
    ax.set_ylim(0, REAL_H_M)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which="major", color="red", linewidth=1.5, alpha=0.4, linestyle=":")
    ax.grid(which="minor", color="red", linewidth=0.7, alpha=0.25, linestyle=":")
    ax.set_xlabel("X (meters)", fontsize=10, color="white", labelpad=4)
    ax.set_ylabel("Y (meters)", fontsize=10, color="white", labelpad=4)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    return fig, ax


# Image 1: grid only
fig1, ax1 = make_axes(fig_w, fig_h)
out1 = ROOT / "data" / "floorplan_grid.png"
fig1.savefig(out1, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
plt.close(fig1)
print(f"Saved: {out1.relative_to(ROOT)}")

# Image 2: grid + walk path
fig2, ax2 = make_axes(fig_w, fig_h)
xs_m = [p[0] for p in SVG_PTS_M]
ys_m = [p[1] for p in SVG_PTS_M]
ax2.plot(xs_m, ys_m, color="#0C8CE9", linewidth=2.5, zorder=5, label="GPS walk path")
ax2.scatter([xs_m[0]], [ys_m[0]], color="#00FF88", s=80, zorder=6, label="Start")
ax2.scatter([xs_m[-1]], [ys_m[-1]], color="#FF4444", s=80, zorder=6, label="End")
for i, (x, y) in enumerate(SVG_PTS_M):
    ax2.annotate(str(i + 1), (x, y), fontsize=6.5, color="yellow",
                 xytext=(4, 4), textcoords="offset points", zorder=7)
ax2.legend(loc="upper left", fontsize=8, facecolor="#111111",
           edgecolor="white", labelcolor="white")
out2 = ROOT / "data" / "floorplan_walkpath.png"
fig2.savefig(out2, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
plt.close(fig2)
print(f"Saved: {out2.relative_to(ROOT)}")
