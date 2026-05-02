from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def generate_trajectory(
    *,
    length: int,
    x0: float,
    y0: float,
    heading0: float,
    world_w_m: float,
    world_h_m: float,
    speed_m_s: float,
    dt_s: float,
    heading_noise_std: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random walk with smooth heading noise inside [margin, world-margin].

    Uses **specular reflection** at walls (not hard clipping). Clipping alone
    tends to park the agent on the perimeter because outward motion is absorbed
    while tangential motion still advances, so trajectories hug the boundary.
    """
    margin = 1.0
    xmin, xmax = margin, world_w_m - margin
    ymin, ymax = margin, world_h_m - margin

    x = np.empty(length + 1, dtype=np.float64)
    y = np.empty(length + 1, dtype=np.float64)
    h = np.empty(length + 1, dtype=np.float64)
    x[0], y[0], h[0] = x0, y0, heading0

    step = speed_m_s * dt_s
    for t in range(length):
        hh = _wrap_pi(h[t] + float(rng.normal(0.0, heading_noise_std)))
        px = x[t] + step * math.cos(hh)
        py = y[t] + step * math.sin(hh)
        # Resolve up to two wall hits per step (corner case)
        for _ in range(3):
            hit = False
            if px > xmax:
                px = xmax - (px - xmax)
                hh = _wrap_pi(math.pi - hh)
                hit = True
            elif px < xmin:
                px = xmin + (xmin - px)
                hh = _wrap_pi(math.pi - hh)
                hit = True
            if py > ymax:
                py = ymax - (py - ymax)
                hh = _wrap_pi(-hh)
                hit = True
            elif py < ymin:
                py = ymin + (ymin - py)
                hh = _wrap_pi(-hh)
                hit = True
            if not hit:
                break
        px = float(np.clip(px, xmin, xmax))
        py = float(np.clip(py, ymin, ymax))
        x[t + 1], y[t + 1], h[t + 1] = px, py, hh

    return x, y, h
