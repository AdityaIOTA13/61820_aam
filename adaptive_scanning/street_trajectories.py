"""
Street-following trajectories from OpenStreetMap (OSMnx) + NetworkX shortest paths.

Street episodes can use random OD pairs; the **four-paths** demo samples four routes where
each new path may **probabilistically** reuse a start or end seen on an earlier path.
Coordinates are affinely mapped into the env world rectangle [0, world_w] x [0, world_h].
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any, TypedDict

import numpy as np


class HomeDailyDayMeta(TypedDict):
    """Graph CRS metre coordinates for one day: intermediate stops only (home is episode-level)."""

    day_index: int
    stop_xy_m: list[Any]  # each (2,) float64 — destinations between walks, before return home

def _half_mile_square_bbox_wgs84(
    *,
    center_lon: float = -71.08975,
    center_lat: float = 42.363,
    half_side_m: float = 0.25 * 1609.344,
) -> tuple[float, float, float, float]:
    """~0.5 mi × 0.5 mi square: half-edge from center = 0.25 mi (west, south, east, north) WGS84."""
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(lat_rad)
    d_lon = half_side_m / m_per_deg_lon
    d_lat = half_side_m / m_per_deg_lat
    return (
        center_lon - d_lon,
        center_lat - d_lat,
        center_lon + d_lon,
        center_lat + d_lat,
    )


# Optional explicit bbox (~0.5 mi square near MIT) if you prefer bbox over a place name
DEFAULT_OSM_BBOX_WGS84: tuple[float, float, float, float] = _half_mile_square_bbox_wgs84()

# Tighter WGS84 box around MIT main campus (Cambridge, MA) for ``--mit-campus`` / small-graph runs
# west, south, east, north — walk network from ``graph_from_bbox`` only inside this extent
MIT_CAMPUS_BBOX_WGS84: tuple[float, float, float, float] = (
    -71.1005,
    42.3533,
    -71.0785,
    42.3648,
)

# Default walk network when ``streets`` mode has no ``osm_place`` / ``osm_bbox`` in config
DEFAULT_OSM_PLACE: str = "Cambridge, Massachusetts, USA"


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def affine_map_points(
    xy: np.ndarray,
    *,
    margin: float,
    world_w_m: float,
    world_h_m: float,
) -> np.ndarray:
    """Map Nx2 points in arbitrary metres to env coordinates preserving aspect ratio, letterboxed."""
    if xy.size == 0:
        return xy
    gx = xy[:, 0].astype(np.float64)
    gy = xy[:, 1].astype(np.float64)
    gw = float(np.ptp(gx)) + 1e-6
    gh = float(np.ptp(gy)) + 1e-6
    inner_w = world_w_m - 2 * margin
    inner_h = world_h_m - 2 * margin
    scale = min(inner_w / gw, inner_h / gh)
    cx = 0.5 * (float(np.min(gx)) + float(np.max(gx)))
    cy = 0.5 * (float(np.min(gy)) + float(np.max(gy)))
    wx0 = 0.5 * world_w_m
    wy0 = 0.5 * world_h_m
    ex = wx0 + (gx - cx) * scale
    ey = wy0 + (gy - cy) * scale
    return np.column_stack([ex, ey])


def inverse_affine_world_to_graph(
    xy_world: np.ndarray,
    reference_xy_graph: np.ndarray,
    *,
    world_w_m: float,
    world_h_m: float,
    margin: float = 1.0,
) -> np.ndarray:
    """Inverse of ``affine_map_points`` using the same scale/center as ``reference_xy_graph``."""
    wxy = np.asarray(xy_world, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(reference_xy_graph, dtype=np.float64)
    if xy.size == 0:
        return wxy.copy()
    gx = xy[:, 0]
    gy = xy[:, 1]
    gw = float(np.ptp(gx)) + 1e-6
    gh = float(np.ptp(gy)) + 1e-6
    inner_w = float(world_w_m) - 2.0 * float(margin)
    inner_h = float(world_h_m) - 2.0 * float(margin)
    scale = min(inner_w / gw, inner_h / gh)
    cx = 0.5 * (float(np.min(gx)) + float(np.max(gx)))
    cy = 0.5 * (float(np.min(gy)) + float(np.max(gy)))
    wx0 = 0.5 * float(world_w_m)
    wy0 = 0.5 * float(world_h_m)
    ex = wxy[:, 0]
    ey = wxy[:, 1]
    gx_out = (ex - wx0) / scale + cx
    gy_out = (ey - wy0) / scale + cy
    return np.column_stack([gx_out, gy_out])


def resample_polyline_at_speed(
    xy_m: np.ndarray,
    *,
    speed_m_s: float,
    dt_s: float,
    n_out: int,
    repeat_path: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk along polyline at constant speed; return arrays length n_out+1 of x,y,heading.

    If ``repeat_path`` is True (default, used by the env), concatenates another copy of
    the polyline when the walk distance exceeds one traverse — so the route never ends.
    That join is from the last vertex back to the first; **do not** use this for one-way
    geographic plots.

    If ``repeat_path`` is False, walk along the polyline once; after the end, the agent
    stays at the terminal point (same heading as the last edge).

    xy_m: (N, 2) consecutive vertices in metres (env coordinates).
    """
    if xy_m.shape[0] < 2:
        x = np.full(n_out + 1, float(xy_m[0, 0]) if xy_m.size else 0.0)
        y = np.full(n_out + 1, float(xy_m[0, 1]) if xy_m.size else 0.0)
        h = np.zeros(n_out + 1, dtype=np.float64)
        return x, y, h

    one = np.asarray(xy_m, dtype=np.float64)
    seg = np.diff(one, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    trip = float(np.sum(seg_len))
    if trip < 1e-6:
        x = np.full(n_out + 1, float(one[0, 0]))
        y = np.full(n_out + 1, float(one[0, 1]))
        h = np.zeros(n_out + 1, dtype=np.float64)
        return x, y, h

    step = max(speed_m_s * dt_s, 1e-4)
    if not repeat_path:
        long_xy = one
    else:
        need = float(n_out) * step
        reps: list[np.ndarray] = [one]
        acc = trip
        while acc < need + step:
            reps.append(one[1:])
            acc += trip
        long_xy = np.vstack(reps)

    seg = np.diff(long_xy, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    dists = np.minimum(np.arange(n_out + 1, dtype=np.float64) * step, cum[-1] - 1e-9)

    x = np.empty(n_out + 1, dtype=np.float64)
    y = np.empty(n_out + 1, dtype=np.float64)
    h = np.empty(n_out + 1, dtype=np.float64)
    for i, d in enumerate(dists):
        j = int(np.searchsorted(cum, float(d), side="right") - 1)
        j = max(0, min(j, len(seg_len) - 1))
        t = (float(d) - cum[j]) / (seg_len[j] + 1e-9)
        t = float(np.clip(t, 0.0, 1.0))
        x[i] = long_xy[j, 0] + t * seg[j, 0]
        y[i] = long_xy[j, 1] + t * seg[j, 1]
        h[i] = math.atan2(seg[j, 1], seg[j, 0])

    return x, y, h


def _largest_cc_nodes(G: Any) -> list[Any]:
    import networkx as nx

    if G.number_of_nodes() == 0:
        return []
    UG = nx.Graph(G.to_undirected(reciprocal=False))
    comps = sorted(nx.connected_components(UG), key=len, reverse=True)
    return list(comps[0]) if comps else []


def _node_xy(G: Any, n: Any) -> tuple[float, float]:
    d = G.nodes[n]
    return float(d["x"]), float(d["y"])


def _path_to_polyline(G: Any, route: list[Any]) -> np.ndarray:
    pts = [_node_xy(G, n) for n in route]
    return np.array(pts, dtype=np.float64)


def shortest_path_polyline(G: Any, o: Any, d: Any) -> np.ndarray | None:
    import networkx as nx

    if o == d:
        x, y = _node_xy(G, o)
        return np.array([[x, y], [x + 1e-3, y]], dtype=np.float64)
    try:
        route = nx.shortest_path(G, o, d, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return _path_to_polyline(G, route)


def _concat_graph_polylines(segments: list[np.ndarray]) -> np.ndarray:
    """Join street polylines in graph metres, dropping duplicate joints when endpoints match."""
    pieces: list[np.ndarray] = []
    for seg in segments:
        s = np.asarray(seg, dtype=np.float64)
        if s.size == 0:
            continue
        if not pieces:
            pieces.append(s)
            continue
        prev = pieces[-1]
        if prev.shape[0] >= 1 and s.shape[0] >= 1 and np.allclose(prev[-1], s[0], atol=1e-3, rtol=0.0):
            pieces.append(s[1:])
        else:
            pieces.append(s)
    if not pieces:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(pieces)


# Home-commute schedule: leave home in morning, dwell >=1h at each intermediate, return evening.
_HOME_COMMUTE_MIN_DWELL_STOP_S = 3600.0
_HOME_COMMUTE_MAX_DWELL_EXTRA_S = 2 * 3600.0
_HOME_COMMUTE_MORNING_WINDOW_MAX_S = 2 * 3600.0
_HOME_COMMUTE_EVENING_BUFFER_MIN_S = 1800.0
_HOME_COMMUTE_EVENING_BUFFER_MAX_S = 2 * 3600.0
_HOME_COMMUTE_SCHEDULE_TRIES = 160


def annotate_playback_events_si_timeline(
    events: list[dict[str, Any]], *, walk_speed_m_s: float
) -> None:
    """
    In-place: add SI-second timeline fields so consumers can align with **real time**
    (1 simulation second ≡ 1 SI second). Travel phases already use ``arc_length / speed``.

    Adds ``duration_si_s``, ``t_moving_cumulative_at_start_s`` / ``_end_s`` (seconds of
    **travel** elapsed since episode t=0, excluding home/dwell clock time).
    """
    spd = max(float(walk_speed_m_s), 0.05)
    events.sort(key=lambda e: (float(e.get("t_start_s", 0.0)), int(e.get("day_index", 0))))
    moving_cum = 0.0
    for ev in events:
        t0 = float(ev.get("t_start_s", 0.0))
        t1 = float(ev.get("t_end_s", 0.0))
        dur = max(0.0, t1 - t0)
        ev["duration_si_s"] = float(dur)
        ph = str(ev.get("phase", ""))
        ev["t_moving_cumulative_at_start_s"] = float(moving_cum)
        if ph == "travel":
            alm = ev.get("arc_length_m")
            if isinstance(alm, (int, float)) and float(alm) > 0.0 and dur > 1e-9:
                ev["effective_speed_m_s"] = float(alm) / float(dur)
            else:
                ev["effective_speed_m_s"] = float(spd)
            moving_cum += dur
        ev["t_moving_cumulative_at_end_s"] = float(moving_cum)


def _polyline_cumdist(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _xyh_at_cumdist(
    xy: np.ndarray, cum: np.ndarray, dist: float
) -> tuple[float, float, float]:
    d = float(np.clip(dist, 0.0, float(cum[-1])))
    j = int(np.searchsorted(cum, d, side="right") - 1)
    j = max(0, min(j, xy.shape[0] - 2))
    span = max(float(cum[j + 1] - cum[j]), 1e-9)
    t = (d - float(cum[j])) / span
    p0 = xy[j]
    p1 = xy[j + 1]
    x = float(p0[0] + t * (p1[0] - p0[0]))
    y = float(p0[1] + t * (p1[1] - p0[1]))
    h = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
    return x, y, h


def _schedule_home_day_positions(
    route_xy: np.ndarray,
    leg_end_vertex_idx: list[int],
    *,
    n_transitions: int,
    dt_s: float,
    speed_m_s: float,
    rng: np.random.Generator,
    day_index: int,
    abs_day_start_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
    """
    Morning at home, travel legs, >=1h dwell at each intermediate, return home, evening at home;
    sample positions every ``dt_s`` for ``n_transitions``+1 points spanning ``n_transitions*dt_s``.

    ``abs_day_start_s`` is simulation time at the first sample of this day (seconds from episode start).
    """
    xy = np.asarray(route_xy, dtype=np.float64)
    if xy.shape[0] < 2 or n_transitions < 1:
        return None
    cum = _polyline_cumdist(xy)
    T_sec = float(n_transitions) * float(dt_s)
    if T_sec < 30.0:
        return None
    spd = max(float(speed_m_s), 0.05)

    k = len(leg_end_vertex_idx)
    break_d = [float(cum[int(i)]) for i in leg_end_vertex_idx]
    travel_lens: list[float] = []
    prev = 0.0
    for bd in break_d:
        travel_lens.append(max(bd - prev, 1e-6))
        prev = bd
    travel_lens.append(max(float(cum[-1]) - prev, 1e-6))
    trav_t = [L / spd for L in travel_lens]

    morning_max = min(_HOME_COMMUTE_MORNING_WINDOW_MAX_S, 0.28 * T_sec)
    eve_lo = min(_HOME_COMMUTE_EVENING_BUFFER_MIN_S, 0.12 * T_sec)
    eve_hi = min(_HOME_COMMUTE_EVENING_BUFFER_MAX_S, 0.28 * T_sec)
    if eve_lo > eve_hi:
        eve_lo, eve_hi = eve_hi, eve_lo

    phases: list[dict[str, Any]] = []
    day_events: list[dict[str, Any]] = []

    for _try in range(_HOME_COMMUTE_SCHEDULE_TRIES):
        phases = []
        day_events = []
        t_leave = float(rng.uniform(0.0, max(morning_max, 120.0)))
        evening_buf = float(rng.uniform(eve_lo, max(eve_hi, eve_lo + 60.0)))
        dwells = [
            float(
                _HOME_COMMUTE_MIN_DWELL_STOP_S
                + rng.uniform(0.0, _HOME_COMMUTE_MAX_DWELL_EXTRA_S)
            )
            for _ in range(k)
        ]
        sum_trav = float(sum(trav_t))
        need = t_leave + sum_trav + float(sum(dwells)) + evening_buf
        if need > T_sec + 1.0:
            for j in range(max(1, k)):
                idx = j % max(k, 1)
                if k > 0:
                    extra = max(0.0, dwells[idx] - _HOME_COMMUTE_MIN_DWELL_STOP_S)
                    dwells[idx] = max(
                        _HOME_COMMUTE_MIN_DWELL_STOP_S,
                        dwells[idx] - min(extra, (need - T_sec) * 0.35 + 1.0),
                    )
            need = t_leave + sum_trav + float(sum(dwells)) + evening_buf
            if need > T_sec + 1.0:
                evening_buf = max(300.0, evening_buf - (need - T_sec))
                need = t_leave + sum_trav + float(sum(dwells)) + evening_buf
            if need > T_sec + 1.0:
                continue

        slack = T_sec - need
        if slack > 1.0 and k > 0:
            u = float(rng.uniform(0.0, slack * 0.65))
            dwells[-1] += u
            evening_buf += slack - u
        elif slack > 1.0:
            evening_buf += slack

        need = t_leave + sum_trav + float(sum(dwells)) + evening_buf
        if need > T_sec + 1.0 and k > 0:
            dwells[-1] = max(
                _HOME_COMMUTE_MIN_DWELL_STOP_S,
                dwells[-1] - max(0.0, need - T_sec),
            )
        if t_leave + sum_trav + float(sum(dwells)) + max(evening_buf, 0.0) > T_sec + 1.0:
            continue

        s_breaks = [0.0] + break_d + [float(cum[-1])]

        def add_ev(
            t0: float,
            t1: float,
            phase: str,
            *,
            leg_i: int | None = None,
            stop_i: int | None = None,
            s0: float | None = None,
            s1: float | None = None,
        ) -> None:
            phases.append(
                {
                    "t0": t0,
                    "t1": t1,
                    "phase": phase,
                    "leg_i": leg_i,
                    "stop_i": stop_i,
                    "s0": s0,
                    "s1": s1,
                }
            )
            sd0 = float(s0) if s0 is not None else 0.0
            xa, ya, ha = _xyh_at_cumdist(xy, cum, sd0)
            ev_out: dict[str, Any] = {
                "t_start_s": abs_day_start_s + t0,
                "t_end_s": abs_day_start_s + t1,
                "day_index": int(day_index),
                "phase": phase,
                "leg_index": leg_i,
                "stop_index": stop_i,
                "x_m": xa,
                "y_m": ya,
                "heading_rad": ha,
            }
            if phase == "travel" and s0 is not None and s1 is not None:
                arc_m = abs(float(s1) - float(s0))
                ev_out["arc_length_m"] = float(arc_m)
                ev_out["travel_speed_m_s"] = float(spd)
            day_events.append(ev_out)

        if t_leave > 1e-6:
            add_ev(0.0, t_leave, "morning_home", s0=0.0, s1=0.0)
        t_cur = t_leave
        for leg_i in range(k + 1):
            s0b = float(s_breaks[leg_i])
            s1b = float(s_breaks[leg_i + 1])
            dur = trav_t[leg_i]
            add_ev(t_cur, t_cur + dur, "travel", leg_i=leg_i, s0=s0b, s1=s1b)
            t_cur += dur
            if leg_i < k:
                sd = float(break_d[leg_i])
                add_ev(t_cur, t_cur + dwells[leg_i], "dwell_stop", stop_i=leg_i, s0=sd, s1=sd)
                t_cur += dwells[leg_i]
        if t_cur < T_sec - 1e-6:
            add_ev(t_cur, T_sec, "evening_home", s0=0.0, s1=0.0)
        break
    else:
        return None

    n_out = int(n_transitions)
    xs = np.empty(n_out + 1, dtype=np.float64)
    ys = np.empty(n_out + 1, dtype=np.float64)
    hs = np.empty(n_out + 1, dtype=np.float64)

    def pos_at(t_rel: float) -> tuple[float, float, float]:
        t_rel = float(np.clip(t_rel, 0.0, T_sec))
        for pi, ph in enumerate(phases):
            t0 = float(ph["t0"])
            t1 = float(ph["t1"])
            last = pi == len(phases) - 1
            inside = (t0 <= t_rel <= t1) if last else (t0 <= t_rel < t1 - 1e-12)
            if not inside:
                continue
            p = str(ph["phase"])
            if p in ("morning_home", "evening_home", "dwell_stop"):
                sd = float(ph.get("s0") or 0.0)
                return _xyh_at_cumdist(xy, cum, sd)
            frac = (t_rel - t0) / max(t1 - t0, 1e-9)
            s0b = float(ph["s0"])
            s1b = float(ph["s1"])
            return _xyh_at_cumdist(xy, cum, s0b + frac * (s1b - s0b))
        return _xyh_at_cumdist(xy, cum, 0.0)

    for j in range(n_out + 1):
        t_rel = min(float(j) * float(dt_s), T_sec)
        xs[j], ys[j], hs[j] = pos_at(t_rel)

    return xs, ys, hs, day_events


def _partition_transition_steps(n_steps: int, transitions_per_full_day: int) -> list[int]:
    """Split ``n_steps`` env transitions into days; last day may be shorter than ``transitions_per_full_day``."""
    spd = max(1, int(transitions_per_full_day))
    out: list[int] = []
    r = int(n_steps)
    while r > 0:
        t = min(spd, r)
        out.append(t)
        r -= t
    return out


def _sample_one_day_home_chain_graph(
    G: Any,
    rng: np.random.Generator,
    home: Any,
    walks_per_day: int,
    nd: np.ndarray,
    *,
    prior_destination_nodes: list[Any] | None = None,
    repeat_destination_p: float = 0.0,
    max_attempts: int = 200,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """
    ``walks_per_day`` legs: home → … → home using ``walks_per_day - 1`` intermediate nodes.

    With probability ``repeat_destination_p`` (and non-empty ``prior_destination_nodes``), each new
    stop prefers a **distinct** node drawn from prior days' destinations; otherwise it is random on
    the graph (excluding home and stops already placed today).

    Returns polyline (N, 2) in graph projected metres plus ``stop_xy_m`` and ``stop_node_ids``.
    """
    w = max(2, int(walks_per_day))
    k = w - 1
    pool = [x for x in nd.tolist() if x != home]
    if len(pool) < k:
        return None
    p_rep = float(np.clip(float(repeat_destination_p), 0.0, 1.0))
    prior_unique = list(dict.fromkeys(prior_destination_nodes or []))
    prior_unique = [n for n in prior_unique if n != home]
    for _ in range(max_attempts):
        if k <= 0:
            return None
        picks: list[Any] = []
        bad_pick = False
        for _slot in range(k):
            placed = False
            for _t in range(150):
                blocked = {home, *picks}
                use_prior = bool(prior_unique) and rng.random() < p_rep
                cand: Any | None = None
                if use_prior:
                    cands = [n for n in prior_unique if n not in blocked]
                    if cands:
                        cand = rng.choice(np.array(cands, dtype=object))
                if cand is None:
                    cands2 = [x for x in pool if x not in blocked]
                    if not cands2:
                        bad_pick = True
                        break
                    cand = rng.choice(np.array(cands2, dtype=object))
                picks.append(cand)
                placed = True
                break
            if bad_pick or not placed:
                bad_pick = True
                break
        if bad_pick or len(picks) != k:
            continue
        segs: list[np.ndarray] = []
        cur = home
        bad = False
        for target in picks:
            p = shortest_path_polyline(G, cur, target)
            if p is None or p.shape[0] < 2:
                bad = True
                break
            segs.append(p)
            cur = target
        if bad:
            continue
        p_home = shortest_path_polyline(G, cur, home)
        if p_home is None or p_home.shape[0] < 2:
            continue
        segs.append(p_home)
        poly = _concat_graph_polylines(segs)
        lens = [int(np.asarray(s, dtype=np.float64).shape[0]) for s in segs]
        pos = lens[0] - 1
        leg_end_vidx: list[int] = [pos]
        for li in lens[1:-1]:
            pos += li - 1
            leg_end_vidx.append(pos)
        meta: dict[str, Any] = {
            "day_index": -1,
            "stop_xy_m": [
                np.array(_node_xy(G, node), dtype=np.float64) for node in picks
            ],
            "stop_node_ids": list(picks),
            "leg_end_vertex_idx": leg_end_vidx,
        }
        return poly, meta
    return None


def build_home_daily_episode_trajectory(
    G: Any,
    rng: np.random.Generator,
    *,
    world_w_m: float,
    world_h_m: float,
    max_sim_time_s: float,
    day_duration_s: float,
    dt_s: float,
    speed_m_s: float,
    walks_per_day: int,
    same_home_next_day_p: float,
    repeat_destination_across_days_p: float = 0.35,
    margin_m: float = 1.0,
    outer_retry: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    """
    Episode motion: each **day** is ``walks_per_day`` chained shortest paths that start and end at **home**.
    Walk *i+1* begins at the graph node where walk *i* ended. One **home** graph node is chosen for the
    whole episode; every day returns there. From day 2 onward, each intermediate stop reuses a node
    visited as a destination on an earlier day with probability ``repeat_destination_across_days_p``.
    (``same_home_next_day_p`` is kept on the API for compatibility but is not used.)

    Returns ``(x, y, h, polyline_graph_m, plot_meta)`` where ``plot_meta`` has ``home_xy_m`` (graph metres),
    ``days`` (``HomeDailyDayMeta`` per day: stops only), and ``day_polylines_graph_m`` (one polyline per day).
    """
    _ = float(np.clip(float(same_home_next_day_p), 0.0, 1.0))  # unused; retained for call compatibility
    p_repeat = float(np.clip(float(repeat_destination_across_days_p), 0.0, 1.0))
    wpd = max(2, int(walks_per_day))
    n_steps = int(max_sim_time_s / float(dt_s))
    if n_steps < 1:
        return None
    spd = max(1, int(math.ceil(float(day_duration_s) / float(dt_s) - 1e-12)))
    day_trans = _partition_transition_steps(n_steps, spd)
    nodes = _largest_cc_nodes(G)
    if len(nodes) < max(4, wpd + 1):
        return None
    nd = np.array(nodes, dtype=object)

    daily_graph: list[np.ndarray] | None = None
    days_plot_meta: list[dict[str, Any]] = []
    for _ in range(outer_retry):
        daily_graph = []
        days_plot_meta = []
        home = rng.choice(nd)
        failed = False
        prior_stop_nodes: list[Any] = []
        for d in range(len(day_trans)):
            got = _sample_one_day_home_chain_graph(
                G,
                rng,
                home,
                wpd,
                nd,
                prior_destination_nodes=prior_stop_nodes if prior_stop_nodes else None,
                repeat_destination_p=p_repeat if d > 0 else 0.0,
            )
            if got is None:
                failed = True
                break
            route, meta_day = got
            if route.shape[0] < 2:
                failed = True
                break
            meta_day["day_index"] = int(d)
            days_plot_meta.append(meta_day)
            daily_graph.append(route)
            for sid in meta_day.get("stop_node_ids", ()):
                prior_stop_nodes.append(sid)
        if not failed and daily_graph is not None:
            break
        daily_graph = None
    if not daily_graph:
        return None

    mega_g = np.vstack(daily_graph)
    xy_all = affine_map_points(mega_g, margin=margin_m, world_w_m=world_w_m, world_h_m=world_h_m)
    off = 0
    daily_world: list[np.ndarray] = []
    for p in daily_graph:
        n = int(p.shape[0])
        daily_world.append(xy_all[off : off + n])
        off += n

    xs_chunks: list[np.ndarray] = []
    ys_chunks: list[np.ndarray] = []
    hs_chunks: list[np.ndarray] = []
    playback_events_all: list[dict[str, Any]] = []
    abs_step_off = 0
    for i, T in enumerate(day_trans):
        leg_idx = days_plot_meta[i].get("leg_end_vertex_idx")
        if not isinstance(leg_idx, list):
            return None
        abs_day_start = float(abs_step_off) * float(dt_s)
        T_sec = float(T) * float(dt_s)
        sch = _schedule_home_day_positions(
            daily_world[i],
            leg_idx,
            n_transitions=int(T),
            dt_s=float(dt_s),
            speed_m_s=float(speed_m_s),
            rng=rng,
            day_index=i,
            abs_day_start_s=abs_day_start,
        )
        if sch is None:
            x_d, y_d, h_d = resample_polyline_at_speed(
                daily_world[i],
                speed_m_s=float(speed_m_s),
                dt_s=float(dt_s),
                n_out=int(T),
                repeat_path=False,
            )
            playback_events_all.append(
                {
                    "t_start_s": abs_day_start,
                    "t_end_s": abs_day_start + T_sec,
                    "day_index": int(i),
                    "phase": "speed_resample_fallback",
                    "leg_index": None,
                    "stop_index": None,
                    "x_m": float(x_d[0]),
                    "y_m": float(y_d[0]),
                    "heading_rad": float(h_d[0]),
                }
            )
        else:
            x_d, y_d, h_d, day_ev = sch
            playback_events_all.extend(day_ev)
        abs_step_off += int(T)
        if i < len(day_trans) - 1:
            xs_chunks.append(x_d[:-1])
            ys_chunks.append(y_d[:-1])
            hs_chunks.append(h_d[:-1])
        else:
            xs_chunks.append(x_d)
            ys_chunks.append(y_d)
            hs_chunks.append(h_d)

    full_x = np.concatenate(xs_chunks)
    full_y = np.concatenate(ys_chunks)
    full_h = np.concatenate(hs_chunks)
    if full_x.shape[0] != n_steps + 1:
        return None
    hx, hy = _node_xy(G, home)
    annotate_playback_events_si_timeline(playback_events_all, walk_speed_m_s=float(speed_m_s))
    plot_meta: dict[str, Any] = {
        "home_xy_m": np.array([hx, hy], dtype=np.float64),
        "days": days_plot_meta,
        "day_polylines_graph_m": [np.asarray(p, dtype=np.float64).copy() for p in daily_graph],
        "playback": {
            "dt_s": float(dt_s),
            "walk_speed_m_s": float(speed_m_s),
            "episode_duration_s": float(full_x.shape[0] - 1) * float(dt_s),
            "time_model": {
                "si_second": (
                    "t_start_s and t_end_s are SI seconds from episode start "
                    "(1 simulation second equals 1 real second)."
                ),
                "travel_phases": (
                    "For phase=travel, duration_si_s equals arc_length_m / walk_speed_m_s "
                    "(graph-route metres at the configured average walking speed)."
                ),
                "t_moving_cumulative_*": (
                    "Cumulative SI seconds spent in travel phases only since episode t=0 "
                    "(stationary home/dwell time does not advance this clock)."
                ),
            },
            "events": playback_events_all,
        },
    }
    return full_x, full_y, full_h, mega_g.copy(), plot_meta


def try_build_home_daily_episode_trajectory(
    *,
    cache_dir: Path,
    place: str | None,
    bbox: tuple[float, float, float, float] | None,
    rng: np.random.Generator,
    world_w_m: float,
    world_h_m: float,
    max_sim_time_s: float,
    day_duration_s: float,
    dt_s: float,
    speed_m_s: float,
    walks_per_day: int,
    same_home_next_day_p: float,
    repeat_destination_across_days_p: float,
    network_type: str = "walk",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]] | None:
    try:
        G = load_or_download_osm_graph(
            cache_dir=cache_dir,
            place=place,
            bbox=bbox,
            network_type=network_type,
        )
        crs_s = _graph_crs_string(G)
        out = build_home_daily_episode_trajectory(
            G,
            rng,
            world_w_m=world_w_m,
            world_h_m=world_h_m,
            max_sim_time_s=max_sim_time_s,
            day_duration_s=day_duration_s,
            dt_s=dt_s,
            speed_m_s=speed_m_s,
            walks_per_day=walks_per_day,
            same_home_next_day_p=same_home_next_day_p,
            repeat_destination_across_days_p=repeat_destination_across_days_p,
        )
        if out is None:
            return None
        x, y, h, poly, plot_meta = out
        return x, y, h, poly, crs_s, plot_meta
    except Exception:
        return None


def nodes_within_planar_radius_m(G: Any, cx: float, cy: float, r_m: float) -> list[Any]:
    """Graph nodes whose projected ``x,y`` lie within ``r_m`` of ``(cx, cy)`` (planar metres)."""
    r2 = float(r_m) * float(r_m)
    out: list[Any] = []
    for n, d in G.nodes(data=True):
        dx = float(d["x"]) - cx
        dy = float(d["y"]) - cy
        if dx * dx + dy * dy <= r2:
            out.append(n)
    return out


def sample_four_od_paths_with_endpoint_reuse_prob(
    G: Any,
    rng: np.random.Generator,
    *,
    endpoint_reuse_prob: float = 0.35,
    min_od_separation_m: float = 120.0,
    inner_max_tries_per_path: int = 250,
    outer_max_batches: int = 500,
) -> list[np.ndarray] | None:
    """
    Four shortest-path polylines in graph projected CRS.

    For path index ``i > 0``, origin and destination are chosen independently: with probability
    ``endpoint_reuse_prob`` reuse a node already used as a start (for ``o``) or as an end (for
    ``d``) on a previous path; otherwise pick a fresh random node from the largest connected
    component. This yields occasional shared endpoints without forcing a fixed 2×2 start×end grid.
    """
    p_reuse = float(np.clip(endpoint_reuse_prob, 0.0, 1.0))
    nodes = _largest_cc_nodes(G)
    if len(nodes) < 4:
        return None
    nd = np.array(nodes, dtype=object)

    for _ in range(outer_max_batches):
        polys: list[np.ndarray] = []
        starts: list[Any] = []
        ends: list[Any] = []
        od_pairs: set[tuple[Any, Any]] = set()
        ok = True
        for _path_idx in range(4):
            placed = False
            for _ in range(inner_max_tries_per_path):
                if starts and rng.random() < p_reuse:
                    o = rng.choice(starts)
                else:
                    o = rng.choice(nd)
                if ends and rng.random() < p_reuse:
                    d = rng.choice(ends)
                else:
                    d = rng.choice(nd)
                if o == d:
                    continue
                ox, oy = _node_xy(G, o)
                dx, dy = _node_xy(G, d)
                if math.hypot(dx - ox, dy - oy) < min_od_separation_m:
                    continue
                key = (o, d)
                if key in od_pairs:
                    continue
                poly = shortest_path_polyline(G, o, d)
                if poly is None or poly.shape[0] < 2:
                    continue
                polys.append(np.asarray(poly, dtype=np.float64))
                starts.append(o)
                ends.append(d)
                od_pairs.add(key)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(polys) == 4:
            return polys
    return None


def load_or_download_osm_graph(
    *,
    cache_dir: Path,
    place: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    network_type: str = "walk",
) -> Any:
    """Load from pickle cache or download via OSMnx. bbox = (west, south, east, north) WGS84."""
    try:
        import osmnx as ox
    except ImportError as e:
        raise ImportError(
            "Street trajectories require osmnx (and geopandas). "
            "Install with: pip install osmnx geopandas"
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (place or "") + "|" + (str(bbox) if bbox else "")
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    pkl = cache_dir / f"osm_graph_{h}.pkl"

    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)

    if place:
        G = ox.graph_from_place(place, network_type=network_type, simplify=True)
    elif bbox is not None:
        w, s, e, n = bbox
        # OSMnx 2: bbox = (left, bottom, right, top) = (west, south, east, north) in WGS84
        try:
            G = ox.graph_from_bbox((w, s, e, n), network_type=network_type, simplify=True)
        except TypeError:
            G = ox.graph_from_bbox(n, s, e, w, network_type=network_type, simplify=True)
    else:
        raise ValueError("Either place= or bbox= must be set for OSM loading")

    G = ox.project_graph(G)
    with open(pkl, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return G


def build_street_trajectory(
    G: Any,
    rng: np.random.Generator,
    *,
    world_w_m: float,
    world_h_m: float,
    n_steps: int,
    speed_m_s: float,
    dt_s: float,
    margin_m: float = 1.0,
    n_anchors: int = 24,
    anchor_reuse_bias: float = 0.72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chain shortest-path trips between anchors (biased) or random largest-component nodes.
    Returns x,y,h each length n_steps+1 in **env world** coordinates.
    """
    nodes = _largest_cc_nodes(G)
    if len(nodes) < 2:
        raise RuntimeError("OSM graph has too few nodes in largest connected component")

    n_anchors = max(2, min(n_anchors, len(nodes)))
    anchor_idx = rng.choice(len(nodes), size=n_anchors, replace=False)
    anchors = [nodes[i] for i in anchor_idx]

    raw_chunks: list[np.ndarray] = []
    target_raw_len_m = max(float(n_steps) * speed_m_s * dt_s * 1.5, 400.0)
    acc_len = 0.0
    safety = 0
    while acc_len < target_raw_len_m and safety < 8000:
        safety += 1
        if rng.random() < anchor_reuse_bias:
            o = rng.choice(anchors)
            d = rng.choice(anchors)
            if o == d and len(anchors) > 1:
                d = rng.choice([a for a in anchors if a != o])
        else:
            o, d = rng.choice(nodes), rng.choice(nodes)
            if o == d:
                continue
        poly = shortest_path_polyline(G, o, d)
        if poly is None or poly.shape[0] < 2:
            continue
        seg = np.diff(poly, axis=0)
        acc_len += float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
        raw_chunks.append(poly)

    if not raw_chunks:
        raise RuntimeError("Could not build any street paths")

    raw = np.vstack([raw_chunks[0]] + [c[1:] for c in raw_chunks[1:]])
    xy_env = affine_map_points(raw, margin=margin_m, world_w_m=world_w_m, world_h_m=world_h_m)

    return resample_polyline_at_speed(xy_env, speed_m_s=speed_m_s, dt_s=dt_s, n_out=n_steps)


def build_single_leg_trajectory(
    G: Any,
    rng: np.random.Generator,
    *,
    world_w_m: float,
    world_h_m: float,
    n_steps: int,
    speed_m_s: float,
    dt_s: float,
    margin_m: float = 1.0,
    n_anchors: int = 28,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One shortest-path walk from a start node to a distinct end node (same anchor pool
    as multi-leg mode for comparable geography), affined into the env rectangle.

    Returns ``(x, y, h, polyline_graph_m)`` where ``polyline_graph_m`` is (N, 2) in the
    OSM projected graph CRS (metres), for basemap overlay.
    """
    nodes = _largest_cc_nodes(G)
    if len(nodes) < 2:
        raise RuntimeError("OSM graph has too few nodes in largest connected component")

    n_anchors = max(2, min(n_anchors, len(nodes)))
    anchor_idx = rng.choice(len(nodes), size=n_anchors, replace=False)
    anchors = [nodes[i] for i in anchor_idx]

    poly: np.ndarray | None = None
    for _ in range(200):
        if len(anchors) >= 2:
            o, d = rng.choice(anchors, size=2, replace=False)
        else:
            o, d = rng.choice(nodes, size=2, replace=False)
        if o == d:
            continue
        poly = shortest_path_polyline(G, o, d)
        if poly is not None and poly.shape[0] >= 2:
            break
    if poly is None or poly.shape[0] < 2:
        raise RuntimeError("Could not find a valid single OD shortest path")

    xy_env = affine_map_points(poly, margin=margin_m, world_w_m=world_w_m, world_h_m=world_h_m)
    x, y, h = resample_polyline_at_speed(
        xy_env, speed_m_s=speed_m_s, dt_s=dt_s, n_out=n_steps
    )
    return x, y, h, np.asarray(poly, dtype=np.float64).copy()


def _graph_crs_string(G: Any) -> str:
    crs = G.graph.get("crs")
    if crs is None:
        return "EPSG:3857"
    if hasattr(crs, "to_string"):
        return str(crs.to_string())
    return str(crs)


def try_build_single_leg_trajectory(
    *,
    cache_dir: Path,
    place: str | None,
    bbox: tuple[float, float, float, float] | None,
    rng: np.random.Generator,
    world_w_m: float,
    world_h_m: float,
    n_steps: int,
    speed_m_s: float,
    dt_s: float,
    n_anchors: int,
    network_type: str = "walk",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
    try:
        G = load_or_download_osm_graph(
            cache_dir=cache_dir,
            place=place,
            bbox=bbox,
            network_type=network_type,
        )
        crs_s = _graph_crs_string(G)
        x, y, h, poly_m = build_single_leg_trajectory(
            G,
            rng,
            world_w_m=world_w_m,
            world_h_m=world_h_m,
            n_steps=n_steps,
            speed_m_s=speed_m_s,
            dt_s=dt_s,
            n_anchors=n_anchors,
        )
        return x, y, h, poly_m, crs_s
    except Exception:
        return None


def check_osm_setup(*, cache_dir: Path | None = None) -> dict[str, Any]:
    """
    Diagnostic: can we import osmnx and download/load the default walk graph?
    """
    cache_dir = cache_dir or Path("outputs/adaptive_scanning/osm_cache")
    try:
        import osmnx as ox  # noqa: F401
    except ImportError as e:
        return {"ok": False, "step": "import", "error": str(e)}
    ver = getattr(ox, "__version__", "unknown")
    try:
        G = load_or_download_osm_graph(
            cache_dir=Path(cache_dir),
            place=DEFAULT_OSM_PLACE,
            bbox=None,
            network_type="walk",
        )
        return {
            "ok": True,
            "osmnx_version": ver,
            "nodes": int(G.number_of_nodes()),
            "edges": int(G.number_of_edges()),
            "cache_dir": str(Path(cache_dir).resolve()),
        }
    except Exception as e:
        return {"ok": False, "osmnx_version": ver, "step": "download_or_parse", "error": repr(e)}


def try_build_street_trajectory(
    *,
    cache_dir: Path,
    place: str | None,
    bbox: tuple[float, float, float, float] | None,
    rng: np.random.Generator,
    world_w_m: float,
    world_h_m: float,
    n_steps: int,
    speed_m_s: float,
    dt_s: float,
    n_anchors: int,
    anchor_reuse_bias: float,
    network_type: str = "walk",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        G = load_or_download_osm_graph(
            cache_dir=cache_dir,
            place=place,
            bbox=bbox,
            network_type=network_type,
        )
        return build_street_trajectory(
            G,
            rng,
            world_w_m=world_w_m,
            world_h_m=world_h_m,
            n_steps=n_steps,
            speed_m_s=speed_m_s,
            dt_s=dt_s,
            n_anchors=n_anchors,
            anchor_reuse_bias=anchor_reuse_bias,
        )
    except Exception:
        return None
