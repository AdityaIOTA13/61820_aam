from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_scanning.config import AdaptiveScanningConfig
from adaptive_scanning.env import CameraBudgetEnv
from adaptive_scanning.policies import Policy


def record_episode(
    env: CameraBudgetEnv,
    policy: Policy,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one episode and return arrays + final last_seen for plotting."""
    obs, info0 = env.reset(seed=seed)
    info = info0
    traj_src = str(info0.get("trajectory_source", "?"))
    act_req: list[int] = []
    act_eff: list[float] = []
    move_eff: list[float] = []
    sim_t: list[float] = []
    budget: list[float] = []

    while True:
        if bool(info.get("interval_is_moving", True)):
            a = policy.act(obs, info)
        else:
            a = 0
        st = env.step(a)
        inf = st.info
        act_req.append(int(inf["action_requested"]))
        act_eff.append(1.0 if inf["camera_on_effective"] else 0.0)
        move_eff.append(1.0 if inf.get("step_interval_is_moving", True) else 0.0)
        sim_t.append(float(inf["sim_time_s"]))
        budget.append(float(inf["budget_s"]))
        obs = st.observation
        info = inf
        if st.terminated or st.truncated:
            break

    assert env.last_seen is not None
    assert env._traj_x is not None
    idx = env._step_idx
    xs = np.array(env._traj_x[: idx + 1], dtype=np.float64)
    ys = np.array(env._traj_y[: idx + 1], dtype=np.float64)
    hs = None
    if env._traj_h is not None:
        hs = np.array(env._traj_h[: idx + 1], dtype=np.float64)
    rec: dict[str, Any] = {
        "xs": xs,
        "ys": ys,
        "action_requested": np.array(act_req, dtype=np.int8),
        "camera_on_effective": np.array(act_eff, dtype=np.float32),
        "step_interval_is_moving": np.array(move_eff, dtype=np.float32),
        "sim_time_s": np.array(sim_t, dtype=np.float64),
        "budget_s": np.array(budget, dtype=np.float64),
        "last_seen": env.last_seen.copy(),
        "final_sim_time_s": float(env._sim_time_s),
        "cfg": env.cfg,
        "trajectory_source": traj_src,
    }
    pg = getattr(env, "_polyline_graph_m", None)
    gc = getattr(env, "_graph_crs", None)
    if pg is not None:
        rec["polyline_graph_m"] = np.asarray(pg, dtype=np.float64).copy()
    if gc is not None:
        rec["graph_crs"] = str(gc)
    hm = getattr(env, "_osm_home_daily_meta", None)
    if hm is not None:
        rec["osm_home_daily_meta"] = copy.deepcopy(hm)
        pb = hm.get("playback")
        if isinstance(pb, dict):
            rec["playback"] = copy.deepcopy(pb)
    if hs is not None:
        rec["traj_heading_rad"] = hs
    return rec


def save_episode_figure(
    rec: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "",
) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg: AdaptiveScanningConfig = rec["cfg"]
    last_seen = rec["last_seen"]
    xs, ys = rec["xs"], rec["ys"]
    eff = rec["camera_on_effective"]
    final_t = rec["final_sim_time_s"]

    age = np.full_like(last_seen, np.nan, dtype=np.float64)
    m = np.isfinite(last_seen)
    age[m] = np.clip((final_t - last_seen[m]) / max(cfg.stale_ref_s, 1.0), 0.0, 3.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    ax0 = axes[0]
    sc = ax0.scatter(xs, ys, c=np.arange(len(xs)), cmap="viridis", s=8, alpha=0.85)
    ax0.plot(xs, ys, "k-", alpha=0.25, linewidth=0.8)
    ax0.scatter([xs[0]], [ys[0]], c="green", s=60, marker="o", zorder=5, label="start")
    ax0.scatter([xs[-1]], [ys[-1]], c="red", s=60, marker="s", zorder=5, label="end")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_title("Agent path (color = time)")
    ax0.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax0, shrink=0.6, label="step index")

    ax1 = axes[1]
    # Equal aspect so one metre on x equals one metre on y; ``aspect="auto"`` stretches the
    # subplot and makes sector coverage look artificially wide compared to the path and to OSM wedges.
    im = ax1.imshow(
        age,
        origin="lower",
        extent=[0, cfg.nx * cfg.resolution_m, 0, cfg.ny * cfg.resolution_m],
        aspect="equal",
        cmap="magma_r",
        interpolation="nearest",
    )
    ax1.set_aspect("equal", adjustable="box")
    ax1.plot(xs, ys, "c-", linewidth=1.2, alpha=0.7)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Map age at end (NaN = never scanned)\nclipped to 3× stale_ref")
    fig.colorbar(im, ax=ax1, shrink=0.6, label="norm. age")

    ax2 = axes[2]
    steps = np.arange(len(eff))
    ax2.fill_between(steps, 0, eff, step="mid", alpha=0.5, color="tab:orange", label="camera ON (effective)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("on")
    ax2.set_title("Sensing (effective; respects budget)")
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(loc="upper right", fontsize=8)

    ts = rec.get("trajectory_source", "?")
    base = title or f"nx={cfg.nx} ny={cfg.ny} res={cfg.resolution_m}m HFOV={cfg.hfov_deg}° R={cfg.scan_radius_m}m"
    supt = f"{base}  |  trajectory={ts}"
    fig.suptitle(supt, fontsize=10)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2 * math.pi) - math.pi


_HOME_DAILY_DAY_COLORS = (
    "#e74c3c",
    "#3498db",
    "#9b59b6",
    "#1abc9c",
    "#f39c12",
    "#34495e",
    "#e91e63",
    "#00bcd4",
)


def plot_home_daily_paths_mercator_ax(
    ax: Any,
    day_polylines_graph_m: list[Any],
    graph_crs: str,
    *,
    linewidth: float = 3.4,
) -> None:
    """Plot each day's route polyline in graph CRS with a distinct color (axis EPSG:3857)."""
    import geopandas as gpd
    from shapely.geometry import LineString

    for d, apoly in enumerate(day_polylines_graph_m):
        arr = np.asarray(apoly, dtype=np.float64)
        if arr.shape[0] < 2:
            continue
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        line = LineString(arr)
        seg = gpd.GeoDataFrame(geometry=[line], crs=graph_crs).to_crs(3857)
        seg.plot(ax=ax, color=col, linewidth=linewidth, alpha=0.95, zorder=6)


def plot_home_daily_markers_mercator_ax(
    ax: Any,
    meta: dict[str, Any],
    graph_crs: str,
    *,
    legend: bool = True,
) -> None:
    """One home (★) plus per-day intermediate stops (●) on an axis in EPSG:3857."""
    import geopandas as gpd
    from matplotlib.lines import Line2D
    from shapely.geometry import Point

    home = meta.get("home_xy_m")
    if home is not None:
        hxy = np.asarray(home, dtype=np.float64).ravel()[:2]
        g_home = gpd.GeoDataFrame(
            geometry=[Point(float(hxy[0]), float(hxy[1]))], crs=graph_crs
        ).to_crs(3857)
        gx, gy = float(g_home.geometry.iloc[0].x), float(g_home.geometry.iloc[0].y)
        ax.scatter(
            [gx],
            [gy],
            c="k",
            marker="*",
            s=240,
            zorder=12,
            edgecolors="w",
            linewidths=0.75,
        )
    days_meta = meta.get("days") or []
    for row in days_meta:
        d = int(row["day_index"])
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        for stop in row.get("stop_xy_m", []):
            sxy = np.asarray(stop, dtype=np.float64).ravel()[:2]
            g_s = gpd.GeoDataFrame(
                geometry=[Point(float(sxy[0]), float(sxy[1]))], crs=graph_crs
            ).to_crs(3857)
            sx, sy = float(g_s.geometry.iloc[0].x), float(g_s.geometry.iloc[0].y)
            ax.scatter(
                [sx],
                [sy],
                c=col,
                marker="o",
                s=100,
                zorder=11,
                edgecolors="k",
                linewidths=0.45,
            )
    if legend and (home is not None or days_meta):
        handles = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="k",
                markersize=13,
                linestyle="None",
                label="Home",
            )
        ]
        seen: set[int] = set()
        for row in days_meta:
            d = int(row["day_index"])
            if d in seen:
                continue
            seen.add(d)
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=col,
                    markersize=9,
                    linestyle="None",
                    label=f"Day {d + 1} destinations",
                )
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=col,
                    linewidth=2.5,
                    label=f"Day {d + 1} path",
                )
            )
        ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.92)


def home_daily_colored_paths_3857(
    rec: dict[str, Any],
) -> list[tuple[str, str, Any]] | None:
    """``[(layer_name, color_hex, shapely LineString EPSG:3857), ...]`` for Folium."""
    meta = rec.get("osm_home_daily_meta")
    crs = rec.get("graph_crs")
    if not meta or not crs:
        return None
    polys = meta.get("day_polylines_graph_m")
    if not polys:
        return None
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        return None
    out: list[tuple[str, str, Any]] = []
    for d, apoly in enumerate(polys):
        arr = np.asarray(apoly, dtype=np.float64)
        if arr.shape[0] < 2:
            continue
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        line = LineString(arr)
        ls3857 = gpd.GeoDataFrame(geometry=[line], crs=crs).to_crs(3857).geometry.iloc[0]
        out.append((f"Day {d + 1} path", col, ls3857))
    return out if out else None


def folium_feature_groups_home_daily(rec: dict[str, Any]) -> list[Any] | None:
    """Feature groups: single home + per-day destination markers (WGS84 for Folium)."""
    meta = rec.get("osm_home_daily_meta")
    crs = rec.get("graph_crs")
    if not meta or not crs:
        return None
    if not meta.get("days") and meta.get("home_xy_m") is None:
        return None
    try:
        import folium
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        return None
    groups: list[Any] = []
    home = meta.get("home_xy_m")
    if home is not None:
        fg_home = folium.FeatureGroup(name="Home", show=True)
        hxy = np.asarray(home, dtype=np.float64).ravel()[:2]
        g = gpd.GeoDataFrame(
            geometry=[Point(float(hxy[0]), float(hxy[1]))], crs=crs
        ).to_crs(4326)
        lat, lon = float(g.geometry.iloc[0].y), float(g.geometry.iloc[0].x)
        folium.CircleMarker(
            [lat, lon],
            radius=10,
            popup="Home",
            color="black",
            weight=2,
            fill=True,
            fillColor="#fff566",
            fillOpacity=1.0,
        ).add_to(fg_home)
        groups.append(fg_home)
    for row in meta["days"]:
        d = int(row["day_index"])
        col_hex = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        fg = folium.FeatureGroup(name=f"Day {d + 1} destinations", show=True)
        for j, stop in enumerate(row.get("stop_xy_m", [])):
            sxy = np.asarray(stop, dtype=np.float64).ravel()[:2]
            g = gpd.GeoDataFrame(
                geometry=[Point(float(sxy[0]), float(sxy[1]))], crs=crs
            ).to_crs(4326)
            lat, lon = float(g.geometry.iloc[0].y), float(g.geometry.iloc[0].x)
            folium.CircleMarker(
                [lat, lon],
                radius=7,
                popup=f"Day {d + 1} stop {j + 1}",
                color=col_hex,
                fill=True,
                fillColor=col_hex,
                fillOpacity=0.9,
                weight=1,
            ).add_to(fg)
        groups.append(fg)
    return groups if groups else None


def _segment_day_indices_playback(rec: dict[str, Any], nseg: int) -> np.ndarray | None:
    """
    Map each motion segment ``k`` to ``playback`` ``day_index`` using segment-start SI time
    ``sim_time_s[k] - dt_s`` (same clock as map-age stamping).

    Uses **per-day SI bounds** ``[min(t_start_s), max(t_end_s)]`` over all events with that
    ``day_index`` (days partition wall-clock time). Resolves boundary overlaps by choosing the
    day whose window **starts** latest among those containing ``t``.
    """
    sim_arr = np.asarray(rec.get("sim_time_s"), dtype=np.float64).ravel()
    if sim_arr.size < nseg or nseg < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)
    stamps = sim_arr[:nseg] - dt_s

    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return None

    bounds_m: dict[int, tuple[float, float]] = {}
    for ev in evs:
        if ev.get("day_index") is None:
            continue
        di = int(ev["day_index"])
        if di < 0:
            continue
        t0 = float(ev.get("t_start_s", 0.0))
        t1 = float(ev.get("t_end_s", t0))
        if t1 < t0:
            t0, t1 = t1, t0
        if di not in bounds_m:
            bounds_m[di] = (t0, t1)
        else:
            lo, hi = bounds_m[di]
            bounds_m[di] = (min(lo, t0), max(hi, t1))
    if not bounds_m:
        return None

    days_sorted = sorted(bounds_m.keys())
    out = np.empty(nseg, dtype=np.int32)
    for k in range(nseg):
        t = float(stamps[k])
        cands = [d for d in days_sorted if bounds_m[d][0] - 1e-6 <= t <= bounds_m[d][1] + 1e-6]
        if len(cands) == 1:
            out[k] = int(cands[0])
        elif len(cands) > 1:
            out[k] = int(max(cands, key=lambda d: bounds_m[d][0]))
        else:
            best_d = int(days_sorted[0])
            best_dist = float("inf")
            for d in days_sorted:
                lo, hi = bounds_m[d]
                mid = 0.5 * (lo + hi)
                dist = abs(t - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_d = int(d)
            out[k] = best_d
    return out


def _moving_segment_day_indices_playback(rec: dict[str, Any], nseg: int) -> np.ndarray | None:
    """
    Map each **episode** motion segment ``k`` (same ``k`` as ``sim_time_s[k]`` / env step ``k``)
    to playback ``day_index`` by overlapping the segment's SI wall window
    ``[sim_time_s[k]-dt_s, sim_time_s[k]]`` with each **travel** event's ``[t_start_s, t_end_s]``.

    Dwell/home steps have no travel overlap; we fall back to ``_segment_day_indices_playback``
    (all playback phases). Using ``k * dt_s`` as a moving-time index was incorrect when ``nseg``
    is the full episode length while travel only advances during legs.
    """
    if nseg < 1:
        return None
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)
    sim_arr = np.asarray(rec.get("sim_time_s"), dtype=np.float64).ravel()
    if sim_arr.size < nseg:
        return None

    travel: list[tuple[float, float, int]] = []
    for ev in evs:
        if str(ev.get("phase")) != "travel":
            continue
        if ev.get("day_index") is None:
            continue
        di = int(ev["day_index"])
        t0 = float(ev.get("t_start_s", 0.0))
        t1 = float(ev.get("t_end_s", t0))
        if t1 < t0:
            t0, t1 = t1, t0
        travel.append((t0, t1, di))
    if not travel:
        return _segment_day_indices_playback(rec, nseg)

    fb = _segment_day_indices_playback(rec, nseg)

    out = np.empty(nseg, dtype=np.int32)
    for k in range(nseg):
        w1 = float(sim_arr[k])
        w0 = w1 - dt_s
        best: list[tuple[float, int]] = []
        for t0, t1, di in travel:
            ov0 = max(w0, t0)
            ov1 = min(w1, t1)
            if ov1 <= ov0 + 1e-9:
                continue
            best.append((t0, di))
        if not best:
            out[k] = int(fb[k]) if fb is not None else 0
        else:
            _, di_pick = max(best, key=lambda row: (row[0], row[1]))
            out[k] = int(di_pick)
    return out


def _moving_segment_count_from_playback(rec: dict[str, Any], max_nseg: int) -> int | None:
    """Number of canonical green-route segments that correspond to actual travel time."""
    if max_nseg < 1:
        return None
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)
    moving_end = 0.0
    seen = False
    for ev in evs:
        if str(ev.get("phase")) != "travel":
            continue
        m1 = float(ev.get("t_moving_cumulative_at_end_s", 0.0))
        moving_end = max(moving_end, m1)
        seen = True
    if not seen:
        return None
    n_move = int(math.ceil(max(moving_end, 0.0) / max(dt_s, 1e-9)))
    return max(1, min(max_nseg, n_move))


def _canonical_day_segment_indices_from_meta(rec: dict[str, Any], nseg: int) -> np.ndarray | None:
    """
    Map canonical green-route segments to day indices using the stored per-day graph polylines.

    This matches the full-walk geometry more directly than wall-clock inference because the green
    layer is built by resampling the concatenated day polylines once at constant walking speed.
    """
    if nseg < 1:
        return None
    meta = rec.get("osm_home_daily_meta")
    if not isinstance(meta, dict):
        return None
    polys = meta.get("day_polylines_graph_m")
    if not isinstance(polys, list) or len(polys) < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    step_m = max(float(cfg.walk_speed_m_s) * float(cfg.dt_s), 1e-9)

    lens: list[float] = []
    for apoly in polys:
        arr = np.asarray(apoly, dtype=np.float64)
        if arr.shape[0] < 2:
            lens.append(0.0)
            continue
        seg = np.diff(arr, axis=0)
        lens.append(float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))))
    if not lens:
        return None

    cum = np.cumsum(np.asarray(lens, dtype=np.float64))
    out = np.empty(nseg, dtype=np.int32)
    for k in range(nseg):
        dist = float(k) * step_m
        d = int(np.searchsorted(cum, dist + 1e-9, side="right"))
        out[k] = min(d, len(lens) - 1)
    return out


def _policy_on_moving_segments(rec: dict[str, Any], nseg: int) -> np.ndarray | None:
    """
    Boolean mask over the canonical moving-route segment stream used by the green always-on layer.

    A route segment is ON iff its moving-time interval falls inside a travel portion of an episode
    step where ``camera_on_effective`` was true.
    """
    if nseg < 1:
        return None
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return None
    sim_arr = np.asarray(rec.get("sim_time_s"), dtype=np.float64).ravel()
    on_ep = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()
    if sim_arr.size < 1 or on_ep.size < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)
    n_ep = min(sim_arr.size, on_ep.size)
    out = np.zeros(nseg, dtype=bool)

    travel_evs: list[dict[str, float | int]] = []
    for ev in evs:
        if str(ev.get("phase")) != "travel":
            continue
        t0 = float(ev.get("t_start_s", 0.0))
        t1 = float(ev.get("t_end_s", t0))
        if t1 < t0:
            t0, t1 = t1, t0
        m0 = float(ev.get("t_moving_cumulative_at_start_s", 0.0))
        m1 = float(ev.get("t_moving_cumulative_at_end_s", m0))
        if m1 < m0:
            m0, m1 = m1, m0
        travel_evs.append({"t0": t0, "t1": t1, "m0": m0, "m1": m1})
    if not travel_evs:
        return None

    for k in range(n_ep):
        if not bool(on_ep[k] > 0.5):
            continue
        w1 = float(sim_arr[k])
        w0 = w1 - dt_s
        for ev in travel_evs:
            t0 = float(ev["t0"])
            t1 = float(ev["t1"])
            ov0 = max(w0, t0)
            ov1 = min(w1, t1)
            if ov1 <= ov0 + 1e-9:
                continue
            dur = max(t1 - t0, 1e-9)
            frac0 = (ov0 - t0) / dur
            frac1 = (ov1 - t0) / dur
            g0 = float(ev["m0"]) + frac0 * (float(ev["m1"]) - float(ev["m0"]))
            g1 = float(ev["m0"]) + frac1 * (float(ev["m1"]) - float(ev["m0"]))
            if g1 < g0:
                g0, g1 = g1, g0
            g1 = max(g1, g0 + 1e-9)
            i0 = max(0, int(math.floor(g0 / dt_s)))
            i1 = min(nseg - 1, int(math.floor((g1 - 1e-9) / dt_s)))
            if i1 >= i0:
                out[i0 : i1 + 1] = True
    return out


def _moving_segment_stamp_times_playback(rec: dict[str, Any], nseg: int) -> np.ndarray | None:
    """Absolute episode-time stamps for canonical moving-route segment starts."""
    if nseg < 1:
        return None
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)

    travel: list[tuple[float, float, float, float]] = []
    for ev in evs:
        if str(ev.get("phase")) != "travel":
            continue
        t0 = float(ev.get("t_start_s", 0.0))
        t1 = float(ev.get("t_end_s", t0))
        if t1 < t0:
            t0, t1 = t1, t0
        m0 = float(ev.get("t_moving_cumulative_at_start_s", 0.0))
        m1 = float(ev.get("t_moving_cumulative_at_end_s", m0))
        if m1 < m0:
            m0, m1 = m1, m0
        travel.append((m0, m1, t0, t1))
    if not travel:
        return None

    out = np.empty(nseg, dtype=np.float64)
    for k in range(nseg):
        m = float(k) * dt_s
        chosen = None
        for m0, m1, t0, t1 in travel:
            if m0 - 1e-6 <= m <= m1 + 1e-6:
                chosen = (m0, m1, t0, t1)
        if chosen is None:
            best = min(travel, key=lambda row: abs(m - 0.5 * (row[0] + row[1])))
            m0, m1, t0, t1 = best
            frac = 0.0 if m1 <= m0 + 1e-9 else float(np.clip((m - m0) / (m1 - m0), 0.0, 1.0))
            out[k] = t0 + frac * (t1 - t0)
            continue
        m0, m1, t0, t1 = chosen
        frac = 0.0 if m1 <= m0 + 1e-9 else float(np.clip((m - m0) / (m1 - m0), 0.0, 1.0))
        out[k] = t0 + frac * (t1 - t0)
    return out


def _home_daily_resampled_polylines_graph(
    rec: dict[str, Any],
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] | None:
    """
    Per-day graph polylines resampled at the same walk speed / dt as the full-walk green layer.
    """
    meta = rec.get("osm_home_daily_meta")
    if not isinstance(meta, dict):
        return None
    polys = meta.get("day_polylines_graph_m")
    if not isinstance(polys, list) or len(polys) < 1:
        return None

    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph, resample_polyline_at_speed

    cfg: AdaptiveScanningConfig = rec["cfg"]
    out: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for d, apoly in enumerate(polys):
        arr = np.asarray(apoly, dtype=np.float64)
        if arr.shape[0] < 2:
            continue
        seg = np.diff(arr, axis=0)
        plen = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
        step_m = max(float(cfg.walk_speed_m_s) * float(cfg.dt_s), 1e-9)
        n_out = max(1, int(math.ceil(plen / step_m)))
        x_d, y_d, h_d = resample_polyline_at_speed(
            arr,
            speed_m_s=float(cfg.walk_speed_m_s),
            dt_s=float(cfg.dt_s),
            n_out=n_out,
            repeat_path=False,
        )
        out.append((d, x_d, y_d, h_d))
    return out if out else None


def _policy_on_day_segments_from_playback(
    rec: dict[str, Any],
    *,
    day_index: int,
    nseg_day: int,
) -> np.ndarray | None:
    """
    Boolean mask over a single day's resampled route segments that were scanned by the policy.

    This uses the same episode-step wall-clock overlap with travel phases as the map-age raster,
    but converts overlap into local moving time within the day route.
    """
    if nseg_day < 1:
        return None
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return None
    sim_arr = np.asarray(rec.get("sim_time_s"), dtype=np.float64).ravel()
    on_ep = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()
    if sim_arr.size < 1 or on_ep.size < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)
    n_ep = min(sim_arr.size, on_ep.size)

    day_travel: list[dict[str, float]] = []
    day_m0 = float("inf")
    for ev in evs:
        if str(ev.get("phase")) != "travel":
            continue
        if int(ev.get("day_index", -1)) != int(day_index):
            continue
        t0 = float(ev.get("t_start_s", 0.0))
        t1 = float(ev.get("t_end_s", t0))
        if t1 < t0:
            t0, t1 = t1, t0
        m0 = float(ev.get("t_moving_cumulative_at_start_s", 0.0))
        m1 = float(ev.get("t_moving_cumulative_at_end_s", m0))
        if m1 < m0:
            m0, m1 = m1, m0
        day_m0 = min(day_m0, m0)
        day_travel.append({"t0": t0, "t1": t1, "m0": m0, "m1": m1})
    if not day_travel:
        return None

    out = np.zeros(nseg_day, dtype=bool)
    for k in range(n_ep):
        if not bool(on_ep[k] > 0.5):
            continue
        w1 = float(sim_arr[k])
        w0 = w1 - dt_s
        for ev in day_travel:
            t0 = float(ev["t0"])
            t1 = float(ev["t1"])
            ov0 = max(w0, t0)
            ov1 = min(w1, t1)
            if ov1 <= ov0 + 1e-9:
                continue
            dur = max(t1 - t0, 1e-9)
            frac0 = (ov0 - t0) / dur
            frac1 = (ov1 - t0) / dur
            g0 = float(ev["m0"]) + frac0 * (float(ev["m1"]) - float(ev["m0"])) - day_m0
            g1 = float(ev["m0"]) + frac1 * (float(ev["m1"]) - float(ev["m0"])) - day_m0
            if g1 < g0:
                g0, g1 = g1, g0
            # Resampled segment i covers local moving time [i*dt_s, (i+1)*dt_s) along the day's route.
            g1 = max(g1, g0 + 1e-9)
            ia = max(0, int(math.floor(g0 / dt_s)))
            ib = min(nseg_day - 1, int(math.floor((g1 - 1e-9) / dt_s)))
            if ib >= ia:
                out[ia : ib + 1] = True
    return out


def _segment_stamp_times_for_day_from_playback(
    rec: dict[str, Any],
    *,
    day_index: int,
    nseg_day: int,
) -> np.ndarray | None:
    """Absolute episode-time stamps for one day's local moving-route segment starts."""
    if nseg_day < 1:
        return None
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)

    day_travel: list[tuple[float, float, float, float]] = []
    day_m0 = float("inf")
    for ev in evs:
        if str(ev.get("phase")) != "travel":
            continue
        if int(ev.get("day_index", -1)) != int(day_index):
            continue
        t0 = float(ev.get("t_start_s", 0.0))
        t1 = float(ev.get("t_end_s", t0))
        if t1 < t0:
            t0, t1 = t1, t0
        m0 = float(ev.get("t_moving_cumulative_at_start_s", 0.0))
        m1 = float(ev.get("t_moving_cumulative_at_end_s", m0))
        if m1 < m0:
            m0, m1 = m1, m0
        day_m0 = min(day_m0, m0)
        day_travel.append((m0, m1, t0, t1))
    if not day_travel:
        return None

    out = np.empty(nseg_day, dtype=np.float64)
    for k in range(nseg_day):
        m_local = float(k) * dt_s
        m_abs = day_m0 + m_local
        chosen = None
        for m0, m1, t0, t1 in day_travel:
            if m0 - 1e-6 <= m_abs <= m1 + 1e-6:
                chosen = (m0, m1, t0, t1)
        if chosen is None:
            best = min(day_travel, key=lambda row: abs(m_abs - 0.5 * (row[0] + row[1])))
            m0, m1, t0, t1 = best
        else:
            m0, m1, t0, t1 = chosen
        frac = 0.0 if m1 <= m0 + 1e-9 else float(np.clip((m_abs - m0) / (m1 - m0), 0.0, 1.0))
        out[k] = t0 + frac * (t1 - t0)
    return out


def _stationary_policy_scan_points_from_playback(
    rec: dict[str, Any],
) -> list[tuple[int, float, float, float, float]]:
    """
    Policy scan wedges that occurred during non-travel playback phases.

    Returns rows ``(day_index, x_graph_m, y_graph_m, heading_rad, stamp_start_s)``.
    These are the scans that appear in the env-step CSV but are invisible if we only
    project policy scans onto moving route segments.
    """
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return []
    evs = pb.get("events")
    if not isinstance(evs, list) or len(evs) < 1:
        return []

    sim_arr = np.asarray(rec.get("sim_time_s"), dtype=np.float64).ravel()
    on_ep = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()
    if sim_arr.size < 1 or on_ep.size < 1:
        return []

    cfg: AdaptiveScanningConfig = rec["cfg"]
    dt_s = float(cfg.dt_s)
    n_ep = min(sim_arr.size, on_ep.size)

    stationary_events: list[dict[str, float]] = []
    for ev in evs:
        phase = str(ev.get("phase", ""))
        if phase == "travel":
            continue
        try:
            stationary_events.append(
                {
                    "t0": float(ev.get("t_start_s", 0.0)),
                    "t1": float(ev.get("t_end_s", 0.0)),
                    "day": float(ev.get("day_index", -1)),
                    "x": float(ev.get("x_m", 0.0)),
                    "y": float(ev.get("y_m", 0.0)),
                    "h": float(ev.get("heading_rad", 0.0)),
                }
            )
        except Exception:
            continue
    if not stationary_events:
        return []

    out: list[tuple[int, float, float, float, float]] = []
    for k in range(n_ep):
        if not bool(on_ep[k] > 0.5):
            continue
        w1 = float(sim_arr[k])
        w0 = w1 - dt_s
        best: dict[str, float] | None = None
        best_ov = 0.0
        for ev in stationary_events:
            t0 = float(ev["t0"])
            t1 = float(ev["t1"])
            if t1 < t0:
                t0, t1 = t1, t0
            ov = min(w1, t1) - max(w0, t0)
            if ov > best_ov + 1e-9:
                best_ov = ov
                best = ev
        if best is None or best_ov <= 1e-9:
            continue
        day = int(best["day"])
        if day < 0:
            continue
        out.append((day, float(best["x"]), float(best["y"]), float(best["h"]), w0))
    return out


def _append_stationary_segments_to_simulation(
    *,
    x_path: np.ndarray,
    y_path: np.ndarray,
    h_path: np.ndarray,
    day_seg: np.ndarray,
    on_mask: np.ndarray,
    stamps: np.ndarray,
    stationary_scans: list[tuple[int, float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Append zero-length scanned wedges to an existing stepped simulation path.

    Each appended stationary scan becomes one OFF connector segment followed by one ON
    zero-length segment at the stationary location. This lets the unified simulator stamp
    the wedge without fabricating a moving connector on the map.
    """
    xp = np.asarray(x_path, dtype=np.float64).ravel().tolist()
    yp = np.asarray(y_path, dtype=np.float64).ravel().tolist()
    hp = np.asarray(h_path, dtype=np.float64).ravel().tolist()
    day = np.asarray(day_seg, dtype=np.int32).ravel().tolist()
    on = np.asarray(on_mask, dtype=bool).ravel().tolist()
    st = np.asarray(stamps, dtype=np.float64).ravel().tolist()

    if not stationary_scans:
        return (
            np.asarray(xp, dtype=np.float64),
            np.asarray(yp, dtype=np.float64),
            np.asarray(hp, dtype=np.float64),
            np.asarray(day, dtype=np.int32),
            np.asarray(on, dtype=bool),
            np.asarray(st, dtype=np.float64),
        )

    if not xp:
        d0, x0, y0, h0, t0 = stationary_scans[0]
        xp = [x0]
        yp = [y0]
        hp = [h0]
        day = []
        on = []
        st = []

    for d, xs, ys, hs, ts in stationary_scans:
        xp.append(float(xs))
        yp.append(float(ys))
        hp.append(float(hs))
        day.append(int(d))
        on.append(False)
        st.append(float(ts))

        xp.append(float(xs))
        yp.append(float(ys))
        hp.append(float(hs))
        day.append(int(d))
        on.append(True)
        st.append(float(ts))

    return (
        np.asarray(xp, dtype=np.float64),
        np.asarray(yp, dtype=np.float64),
        np.asarray(hp, dtype=np.float64),
        np.asarray(day, dtype=np.int32),
        np.asarray(on, dtype=bool),
        np.asarray(st, dtype=np.float64),
    )


def home_daily_per_day_coverage_layers_3857(
    rec: dict[str, Any],
    *,
    graph_crs: str,
    x_graph: np.ndarray,
    y_graph: np.ndarray,
    h_graph: np.ndarray,
    stride: int,
    cfg: AdaptiveScanningConfig,
) -> list[tuple[str, str, Any]] | None:
    """
    Always-on wedge unions in EPSG:3857, one merged polygon per calendar day.

    These layers follow the actual episode timeline, so callers should pass the graph-space
    trajectory derived from ``rec["xs"]/rec["ys"]`` rather than the one-pass route resample
    used for the combined always-on green layer.
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    day_resampled = _home_daily_resampled_polylines_graph(rec)
    if day_resampled is not None:
        r_m = float(cfg.scan_radius_m)
        st = max(1, int(stride))
        layers: list[tuple[str, str, Any]] = []
        for d, xa, ya, ha in day_resampled:
            n_scan = int(xa.size) - 1
            wedges: list[Any] = []
            for i in range(0, n_scan, st):
                wedges.extend(
                    _wedge_polygons_motion_segment_utm(
                        float(xa[i]),
                        float(ya[i]),
                        float(ha[i]),
                        float(xa[i + 1]),
                        float(ya[i + 1]),
                        float(ha[i + 1]),
                        radius_m=r_m,
                        hfov_deg=float(cfg.hfov_deg),
                        resolution_m=float(cfg.resolution_m),
                    )
                )
            if not wedges:
                continue
            uu = unary_union(wedges)
            if uu.is_empty:
                continue
            g3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            layers.append((f"Coverage — always-on Day {d + 1}", col, g3857))
        return layers if layers else None
    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    xs_a = np.asarray(rec.get("xs"), dtype=np.float64).ravel()
    ys_a = np.asarray(rec.get("ys"), dtype=np.float64).ravel()
    if xs_a.size >= 2 and ys_a.size == xs_a.size:
        n_scan = int(xs_a.size) - 1
        hs = rec.get("traj_heading_rad")
        if hs is not None and int(np.asarray(hs).size) == int(xs_a.size):
            h_pol = np.asarray(hs, dtype=np.float64).ravel()[: int(xs_a.size)]
        else:
            h_pol = np.empty(int(xs_a.size), dtype=np.float64)
            for ii in range(int(xs_a.size) - 1):
                h_pol[ii] = math.atan2(
                    float(ys_a[ii + 1] - ys_a[ii]),
                    float(xs_a[ii + 1] - xs_a[ii]),
                )
            h_pol[-1] = h_pol[-2]

        on = np.asarray(rec["camera_on_effective"], dtype=np.float64).ravel()[:n_scan] > 0.5
        if np.any(on):
            day_seg = _segment_day_indices_playback(rec, n_scan)
            if day_seg is None:
                day_seg = np.zeros(n_scan, dtype=np.int32)
            poly = rec.get("polyline_graph_m")
            if poly is not None:
                poly_a = np.asarray(poly, dtype=np.float64)
                world_w_m = float(cfg.nx) * float(cfg.resolution_m)
                world_h_m = float(cfg.ny) * float(cfg.resolution_m)
                r_m = float(cfg.scan_radius_m)
                by_day_actual: dict[int, list[Any]] = {}
                for i in range(n_scan):
                    if not bool(on[i]):
                        continue
                    d = int(day_seg[i])
                    wseg = np.array(
                        [[xs_a[i], ys_a[i]], [xs_a[i + 1], ys_a[i + 1]]],
                        dtype=np.float64,
                    )
                    gseg = inverse_affine_world_to_graph(
                        wseg,
                        poly_a,
                        world_w_m=world_w_m,
                        world_h_m=world_h_m,
                        margin=1.0,
                    )
                    wedges_g = _wedge_polygons_motion_segment_utm(
                        float(gseg[0, 0]),
                        float(gseg[0, 1]),
                        float(h_pol[i]),
                        float(gseg[1, 0]),
                        float(gseg[1, 1]),
                        float(h_pol[i + 1]),
                        radius_m=r_m,
                        hfov_deg=float(cfg.hfov_deg),
                        resolution_m=float(cfg.resolution_m),
                    )
                    if wedges_g:
                        by_day_actual.setdefault(d, []).extend(wedges_g)
                if by_day_actual:
                    layers_actual: list[tuple[str, str, Any]] = []
                    for d in sorted(by_day_actual.keys()):
                        parts = by_day_actual[d]
                        if not parts:
                            continue
                        uu = unary_union(parts)
                        if uu.is_empty:
                            continue
                        g3857 = gpd.GeoDataFrame(
                            geometry=[uu], crs=str(graph_crs)
                        ).to_crs(3857).geometry.iloc[0]
                        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
                        layers_actual.append((f"Coverage — policy camera Day {d + 1}", col, g3857))
                    if layers_actual:
                        return layers_actual

    day_resampled = _home_daily_resampled_polylines_graph(rec)
    if day_resampled is not None:
        r_m = float(cfg.scan_radius_m)
        st = max(1, int(stride))
        layers: list[tuple[str, str, Any]] = []
        for d, xa, ya, ha in day_resampled:
            n_scan = int(xa.size) - 1
            wedges: list[Any] = []
            for i in range(0, n_scan, st):
                wedges.extend(
                    _wedge_polygons_motion_segment_utm(
                        float(xa[i]),
                        float(ya[i]),
                        float(ha[i]),
                        float(xa[i + 1]),
                        float(ya[i + 1]),
                        float(ha[i + 1]),
                        radius_m=r_m,
                        hfov_deg=float(cfg.hfov_deg),
                        resolution_m=float(cfg.resolution_m),
                    )
                )
            if not wedges:
                continue
            uu = unary_union(wedges)
            if uu.is_empty:
                continue
            g3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            layers.append((f"Coverage — always-on Day {d + 1}", col, g3857))
        return layers if layers else None

    xa = np.asarray(x_graph, dtype=np.float64).ravel()
    ya = np.asarray(y_graph, dtype=np.float64).ravel()
    ha = np.asarray(h_graph, dtype=np.float64).ravel()
    if xa.size < 2 or xa.size != ya.size or ha.size != xa.size:
        return None
    n_scan = int(xa.size) - 1
    n_eff = _moving_segment_count_from_playback(rec, n_scan)
    if n_eff is not None:
        n_scan = int(n_eff)
        xa = xa[: n_scan + 1]
        ya = ya[: n_scan + 1]
        ha = ha[: n_scan + 1]
    day_seg = _canonical_day_segment_indices_from_meta(rec, n_scan)
    if day_seg is None:
        day_seg = _moving_segment_day_indices_playback(rec, n_scan)
    if day_seg is None:
        return None
    if int(day_seg.max()) == int(day_seg.min()):
        return None

    st = max(1, int(stride))
    r_m = float(cfg.scan_radius_m)
    by_day: dict[int, list[Any]] = {}
    for i in range(0, n_scan, st):
        d = int(day_seg[i])
        w = _wedge_polygons_motion_segment_utm(
            float(xa[i]),
            float(ya[i]),
            float(ha[i]),
            float(xa[i + 1]),
            float(ya[i + 1]),
            float(ha[i + 1]),
            radius_m=r_m,
            hfov_deg=float(cfg.hfov_deg),
            resolution_m=float(cfg.resolution_m),
        )
        by_day.setdefault(d, []).extend(w)

    layers: list[tuple[str, str, Any]] = []
    for d in sorted(by_day.keys()):
        wedges = by_day[d]
        if not wedges:
            continue
        uu = unary_union(wedges)
        if uu.is_empty:
            continue
        g3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        layers.append((f"Coverage — always-on Day {d + 1}", col, g3857))
    return layers if len(layers) > 1 else None


def policy_camera_coverage_layers_3857_by_day(
    rec: dict[str, Any],
    *,
    x_graph: np.ndarray,
    y_graph: np.ndarray,
    h_graph: np.ndarray,
    graph_crs: str,
    stride: int,
    cfg: AdaptiveScanningConfig,
) -> list[tuple[str, str, Any]] | None:
    """
    Motion-integrated sector wedges along **``rec["xs"], rec["ys"]``** (env world metres — the same
    path as map-age stamping), **only for segments where** ``camera_on_effective`` **is true**.

    Uses the same **segment stride** as the merged always-on green layer (``try_save_realworld_*``),
    so wedge sampling density matches green and unions do not over-thicken from every-``dt`` overlap.

    Wedges are built in **graph projected metres** (after mapping segment endpoints from world) so
    ``scan_radius_m`` matches always-on / env scale, then EPSG:3857 for Folium.
    One merged polygon per playback day (or day 0 when playback is absent).
    """
    import geopandas as gpd
    from shapely.ops import unary_union
    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    xs_a = np.asarray(rec.get("xs"), dtype=np.float64).ravel()
    ys_a = np.asarray(rec.get("ys"), dtype=np.float64).ravel()
    if xs_a.size >= 2 and ys_a.size == xs_a.size:
        n_scan = int(xs_a.size) - 1
        hs = rec.get("traj_heading_rad")
        if hs is not None and int(np.asarray(hs).size) == int(xs_a.size):
            h_pol = np.asarray(hs, dtype=np.float64).ravel()[: int(xs_a.size)]
        else:
            h_pol = np.empty(int(xs_a.size), dtype=np.float64)
            for ii in range(int(xs_a.size) - 1):
                h_pol[ii] = math.atan2(
                    float(ys_a[ii + 1] - ys_a[ii]),
                    float(xs_a[ii + 1] - xs_a[ii]),
                )
            h_pol[-1] = h_pol[-2]
        on = np.asarray(rec["camera_on_effective"], dtype=np.float64).ravel()[:n_scan] > 0.5
        if np.any(on):
            day_seg = _segment_day_indices_playback(rec, n_scan)
            if day_seg is None:
                day_seg = np.zeros(n_scan, dtype=np.int32)
            poly = rec.get("polyline_graph_m")
            if poly is not None:
                poly_a = np.asarray(poly, dtype=np.float64)
                world_w_m = float(cfg.nx) * float(cfg.resolution_m)
                world_h_m = float(cfg.ny) * float(cfg.resolution_m)
                r_m = float(cfg.scan_radius_m)
                by_day_actual: dict[int, list[Any]] = {}
                for i in range(n_scan):
                    if not bool(on[i]):
                        continue
                    d = int(day_seg[i])
                    wseg = np.array(
                        [[xs_a[i], ys_a[i]], [xs_a[i + 1], ys_a[i + 1]]],
                        dtype=np.float64,
                    )
                    gseg = inverse_affine_world_to_graph(
                        wseg,
                        poly_a,
                        world_w_m=world_w_m,
                        world_h_m=world_h_m,
                        margin=1.0,
                    )
                    wedges_g = _wedge_polygons_motion_segment_utm(
                        float(gseg[0, 0]),
                        float(gseg[0, 1]),
                        float(h_pol[i]),
                        float(gseg[1, 0]),
                        float(gseg[1, 1]),
                        float(h_pol[i + 1]),
                        radius_m=r_m,
                        hfov_deg=float(cfg.hfov_deg),
                        resolution_m=float(cfg.resolution_m),
                    )
                    if wedges_g:
                        by_day_actual.setdefault(d, []).extend(wedges_g)
                if by_day_actual:
                    layers_actual: list[tuple[str, str, Any]] = []
                    for d in sorted(by_day_actual.keys()):
                        parts = by_day_actual[d]
                        if not parts:
                            continue
                        uu = unary_union(parts)
                        if uu.is_empty:
                            continue
                        g3857 = gpd.GeoDataFrame(
                            geometry=[uu], crs=str(graph_crs)
                        ).to_crs(3857).geometry.iloc[0]
                        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
                        layers_actual.append((f"Coverage — policy camera Day {d + 1}", col, g3857))
                    if layers_actual:
                        return layers_actual

    day_resampled = _home_daily_resampled_polylines_graph(rec)
    if day_resampled is not None:
        r_m = float(cfg.scan_radius_m)
        st = max(1, int(stride))
        layers: list[tuple[str, str, Any]] = []
        for d, xg_d, yg_d, hg_d in day_resampled:
            n_scan = int(xg_d.size) - 1
            on_d = _policy_on_day_segments_from_playback(rec, day_index=d, nseg_day=n_scan)
            if on_d is None or not np.any(on_d):
                continue
            wedges: list[Any] = []
            for i in range(0, n_scan, st):
                if not bool(on_d[i]):
                    continue
                wedges.extend(
                    _wedge_polygons_motion_segment_utm(
                        float(xg_d[i]),
                        float(yg_d[i]),
                        float(hg_d[i]),
                        float(xg_d[i + 1]),
                        float(yg_d[i + 1]),
                        float(hg_d[i + 1]),
                        radius_m=r_m,
                        hfov_deg=float(cfg.hfov_deg),
                        resolution_m=float(cfg.resolution_m),
                    )
                )
            if not wedges:
                continue
            uu = unary_union(wedges)
            if uu.is_empty:
                continue
            g3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            layers.append((f"Coverage — policy camera Day {d + 1}", col, g3857))
        return layers if layers else None

    xg = np.asarray(x_graph, dtype=np.float64).ravel()
    yg = np.asarray(y_graph, dtype=np.float64).ravel()
    hg = np.asarray(h_graph, dtype=np.float64).ravel()
    if xg.size < 2 or yg.size != xg.size or hg.size != xg.size:
        return None
    n_scan = int(xg.size) - 1
    n_eff = _moving_segment_count_from_playback(rec, n_scan)
    if n_eff is not None:
        n_scan = int(n_eff)
        xg = xg[: n_scan + 1]
        yg = yg[: n_scan + 1]
        hg = hg[: n_scan + 1]
    on = _policy_on_moving_segments(rec, n_scan)
    if on is None:
        on = np.asarray(rec["camera_on_effective"], dtype=np.float64).ravel()[:n_scan] > 0.5
    if not np.any(on):
        return None

    day_seg = _canonical_day_segment_indices_from_meta(rec, n_scan)
    if day_seg is None:
        day_seg = _moving_segment_day_indices_playback(rec, n_scan)
    if day_seg is None:
        day_seg = np.zeros(n_scan, dtype=np.int32)

    st = max(1, int(stride))
    r_m = float(cfg.scan_radius_m)
    by_day: dict[int, list[Any]] = {}
    for i in range(0, n_scan, st):
        if not bool(on[i]):
            continue
        d = int(day_seg[i])
        wedges_g = _wedge_polygons_motion_segment_utm(
            float(xg[i]),
            float(yg[i]),
            float(hg[i]),
            float(xg[i + 1]),
            float(yg[i + 1]),
            float(hg[i + 1]),
            radius_m=r_m,
            hfov_deg=float(cfg.hfov_deg),
            resolution_m=float(cfg.resolution_m),
        )
        if not wedges_g:
            continue
        by_day.setdefault(d, []).extend(wedges_g)

    layers: list[tuple[str, str, Any]] = []
    for d in sorted(by_day.keys()):
        parts = by_day[d]
        if not parts:
            continue
        uu = unary_union(parts)
        if uu.is_empty:
            continue
        g3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        layers.append((f"Coverage — policy camera Day {d + 1}", col, g3857))
    return layers if layers else None


def _sector_wedge_polygon(
    ax: float,
    ay: float,
    heading_rad: float,
    radius_m: float,
    hfov_deg: float,
    *,
    n_arc: int = 36,
):
    """Planar forward sector (metres), CCW polygon for unary_union."""
    from shapely.geometry import Polygon

    half = math.radians(0.5 * float(hfov_deg))
    angs = np.linspace(heading_rad - half, heading_rad + half, n_arc)
    ring = [(ax, ay)]
    for a in angs:
        ring.append((ax + radius_m * math.cos(float(a)), ay + radius_m * math.sin(float(a))))
    ring.append((ax, ay))
    return Polygon(ring)


def home_daily_per_day_coverage_layers_3857_v2(
    rec: dict[str, Any],
    *,
    graph_crs: str,
    x_graph: np.ndarray,
    y_graph: np.ndarray,
    h_graph: np.ndarray,
    stride: int,
    cfg: AdaptiveScanningConfig,
) -> list[tuple[str, str, Any]] | None:
    """Clean replacement for per-day always-on coverage."""
    import geopandas as gpd
    from shapely.ops import unary_union

    day_resampled = _home_daily_resampled_polylines_graph(rec)
    if day_resampled is not None:
        layers: list[tuple[str, str, Any]] = []
        for d, xa, ya, ha in day_resampled:
            n_scan = int(xa.size) - 1
            wedges: list[Any] = []
            for i in range(0, n_scan, max(1, int(stride))):
                wedges.extend(
                    _wedge_polygons_motion_segment_utm(
                        float(xa[i]),
                        float(ya[i]),
                        float(ha[i]),
                        float(xa[i + 1]),
                        float(ya[i + 1]),
                        float(ha[i + 1]),
                        radius_m=float(cfg.scan_radius_m),
                        hfov_deg=float(cfg.hfov_deg),
                        resolution_m=float(cfg.resolution_m),
                    )
                )
            if not wedges:
                continue
            uu = unary_union(wedges)
            if uu.is_empty:
                continue
            geom3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            layers.append((f"Coverage — always-on Day {d + 1}", col, geom3857))
        return layers if layers else None

    xa = np.asarray(x_graph, dtype=np.float64).ravel()
    ya = np.asarray(y_graph, dtype=np.float64).ravel()
    ha = np.asarray(h_graph, dtype=np.float64).ravel()
    if xa.size < 2 or ya.size != xa.size or ha.size != xa.size:
        return None
    n_scan = int(xa.size) - 1
    day_seg = _canonical_day_segment_indices_from_meta(rec, n_scan)
    if day_seg is None:
        day_seg = _moving_segment_day_indices_playback(rec, n_scan)
    if day_seg is None or int(day_seg.max()) == int(day_seg.min()):
        return None

    by_day: dict[int, list[Any]] = {}
    for i in range(0, n_scan, max(1, int(stride))):
        by_day.setdefault(int(day_seg[i]), []).extend(
            _wedge_polygons_motion_segment_utm(
                float(xa[i]),
                float(ya[i]),
                float(ha[i]),
                float(xa[i + 1]),
                float(ya[i + 1]),
                float(ha[i + 1]),
                radius_m=float(cfg.scan_radius_m),
                hfov_deg=float(cfg.hfov_deg),
                resolution_m=float(cfg.resolution_m),
            )
        )
    layers: list[tuple[str, str, Any]] = []
    for d in sorted(by_day.keys()):
        uu = unary_union(by_day[d])
        if uu.is_empty:
            continue
        geom3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        layers.append((f"Coverage — always-on Day {d + 1}", col, geom3857))
    return layers if layers else None


def policy_camera_coverage_layers_3857_by_day_v2(
    rec: dict[str, Any],
    *,
    graph_crs: str,
    cfg: AdaptiveScanningConfig,
) -> list[tuple[str, str, Any]] | None:
    """Clean replacement for policy coverage, aligned with map-age env-path stamping."""
    import geopandas as gpd
    from shapely.ops import unary_union
    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    xs_a = np.asarray(rec.get("xs"), dtype=np.float64).ravel()
    ys_a = np.asarray(rec.get("ys"), dtype=np.float64).ravel()
    if xs_a.size < 2 or ys_a.size != xs_a.size:
        return None
    n_scan = int(xs_a.size) - 1
    on = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()[:n_scan] > 0.5
    if not np.any(on):
        return None

    hs = rec.get("traj_heading_rad")
    if hs is not None and int(np.asarray(hs).size) == int(xs_a.size):
        h_pol = np.asarray(hs, dtype=np.float64).ravel()[: int(xs_a.size)]
    else:
        h_pol = np.empty(int(xs_a.size), dtype=np.float64)
        for ii in range(int(xs_a.size) - 1):
            h_pol[ii] = math.atan2(
                float(ys_a[ii + 1] - ys_a[ii]),
                float(xs_a[ii + 1] - xs_a[ii]),
            )
        h_pol[-1] = h_pol[-2]

    day_seg = _segment_day_indices_playback(rec, n_scan)
    if day_seg is None:
        day_seg = np.zeros(n_scan, dtype=np.int32)
    poly = rec.get("polyline_graph_m")
    if poly is None:
        return None
    poly_a = np.asarray(poly, dtype=np.float64)
    world_w_m = float(cfg.nx) * float(cfg.resolution_m)
    world_h_m = float(cfg.ny) * float(cfg.resolution_m)

    by_day: dict[int, list[Any]] = {}
    for i in range(n_scan):
        if not bool(on[i]):
            continue
        wseg = np.array([[xs_a[i], ys_a[i]], [xs_a[i + 1], ys_a[i + 1]]], dtype=np.float64)
        gseg = inverse_affine_world_to_graph(
            wseg,
            poly_a,
            world_w_m=world_w_m,
            world_h_m=world_h_m,
            margin=1.0,
        )
        by_day.setdefault(int(day_seg[i]), []).extend(
            _wedge_polygons_motion_segment_utm(
                float(gseg[0, 0]),
                float(gseg[0, 1]),
                float(h_pol[i]),
                float(gseg[1, 0]),
                float(gseg[1, 1]),
                float(h_pol[i + 1]),
                radius_m=float(cfg.scan_radius_m),
                hfov_deg=float(cfg.hfov_deg),
                resolution_m=float(cfg.resolution_m),
            )
        )

    layers: list[tuple[str, str, Any]] = []
    for d in sorted(by_day.keys()):
        uu = unary_union(by_day[d])
        if uu.is_empty:
            continue
        geom3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        layers.append((f"Coverage — policy camera Day {d + 1}", col, geom3857))
    return layers if layers else None


def policy_camera_coverage_layers_3857_v3(
    rec: dict[str, Any],
    *,
    x_graph: np.ndarray,
    y_graph: np.ndarray,
    h_graph: np.ndarray,
    graph_crs: str,
    stride: int,
    cfg: AdaptiveScanningConfig,
) -> list[tuple[str, str, Any]] | None:
    """Policy coverage on the canonical graph route, aligned to the displayed walking path."""
    import geopandas as gpd
    from shapely.ops import unary_union

    xg = np.asarray(x_graph, dtype=np.float64).ravel()
    yg = np.asarray(y_graph, dtype=np.float64).ravel()
    hg = np.asarray(h_graph, dtype=np.float64).ravel()
    if xg.size < 2 or yg.size != xg.size or hg.size != xg.size:
        return None
    n_scan = int(xg.size) - 1
    on = _policy_on_moving_segments(rec, n_scan)
    if on is None:
        on = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()[:n_scan] > 0.5
    if not np.any(on):
        return None

    day_seg = _canonical_day_segment_indices_from_meta(rec, n_scan)
    if day_seg is None:
        day_seg = _moving_segment_day_indices_playback(rec, n_scan)
    if day_seg is None:
        day_seg = np.zeros(n_scan, dtype=np.int32)

    by_day: dict[int, list[Any]] = {}
    for i in range(0, n_scan, max(1, int(stride))):
        if not bool(on[i]):
            continue
        by_day.setdefault(int(day_seg[i]), []).extend(
            _wedge_polygons_motion_segment_utm(
                float(xg[i]),
                float(yg[i]),
                float(hg[i]),
                float(xg[i + 1]),
                float(yg[i + 1]),
                float(hg[i + 1]),
                radius_m=float(cfg.scan_radius_m),
                hfov_deg=float(cfg.hfov_deg),
                resolution_m=float(cfg.resolution_m),
            )
        )

    layers: list[tuple[str, str, Any]] = []
    for d in sorted(by_day.keys()):
        uu = unary_union(by_day[d])
        if uu.is_empty:
            continue
        geom3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        layers.append((f"Coverage — policy camera Day {d + 1}", col, geom3857))
    return layers if layers else None


def policy_camera_coverage_layers_3857_v4(
    rec: dict[str, Any],
    *,
    x_graph: np.ndarray,
    y_graph: np.ndarray,
    h_graph: np.ndarray,
    graph_crs: str,
    stride: int,
    cfg: AdaptiveScanningConfig,
) -> list[tuple[str, str, Any]] | None:
    """Policy coverage on the same canonical route geometry as the displayed path/always-on layers."""
    import geopandas as gpd
    from shapely.ops import unary_union

    day_resampled = _home_daily_resampled_polylines_graph(rec)
    if day_resampled is not None:
        layers: list[tuple[str, str, Any]] = []
        st = max(1, int(stride))
        for d, xg_d, yg_d, hg_d in day_resampled:
            n_scan_d = int(xg_d.size) - 1
            if n_scan_d < 1:
                continue
            on_d = _policy_on_day_segments_from_playback(rec, day_index=int(d), nseg_day=n_scan_d)
            if on_d is None or not np.any(on_d):
                continue
            wedges: list[Any] = []
            for i in range(0, n_scan_d, st):
                if not bool(on_d[i]):
                    continue
                wedges.extend(
                    _wedge_polygons_motion_segment_utm(
                        float(xg_d[i]),
                        float(yg_d[i]),
                        float(hg_d[i]),
                        float(xg_d[i + 1]),
                        float(yg_d[i + 1]),
                        float(hg_d[i + 1]),
                        radius_m=float(cfg.scan_radius_m),
                        hfov_deg=float(cfg.hfov_deg),
                        resolution_m=float(cfg.resolution_m),
                    )
                )
            if not wedges:
                continue
            uu = unary_union(wedges)
            if uu.is_empty:
                continue
            geom3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            layers.append((f"Coverage — policy camera Day {d + 1}", col, geom3857))
        return layers if layers else None

    xg = np.asarray(x_graph, dtype=np.float64).ravel()
    yg = np.asarray(y_graph, dtype=np.float64).ravel()
    hg = np.asarray(h_graph, dtype=np.float64).ravel()
    if xg.size < 2 or yg.size != xg.size or hg.size != xg.size:
        return None
    n_scan = int(xg.size) - 1
    n_eff = _moving_segment_count_from_playback(rec, n_scan)
    if n_eff is not None:
        n_scan = int(n_eff)
        xg = xg[: n_scan + 1]
        yg = yg[: n_scan + 1]
        hg = hg[: n_scan + 1]
    on = _policy_on_moving_segments(rec, n_scan)
    if on is None:
        on = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()[:n_scan] > 0.5
    if not np.any(on):
        return None

    day_seg = _canonical_day_segment_indices_from_meta(rec, n_scan)
    if day_seg is None:
        day_seg = _moving_segment_day_indices_playback(rec, n_scan)
    if day_seg is None:
        day_seg = np.zeros(n_scan, dtype=np.int32)

    by_day: dict[int, list[Any]] = {}
    for i in range(0, n_scan, max(1, int(stride))):
        if not bool(on[i]):
            continue
        by_day.setdefault(int(day_seg[i]), []).extend(
            _wedge_polygons_motion_segment_utm(
                float(xg[i]),
                float(yg[i]),
                float(hg[i]),
                float(xg[i + 1]),
                float(yg[i + 1]),
                float(hg[i + 1]),
                radius_m=float(cfg.scan_radius_m),
                hfov_deg=float(cfg.hfov_deg),
                resolution_m=float(cfg.resolution_m),
            )
        )

    layers: list[tuple[str, str, Any]] = []
    for d in sorted(by_day.keys()):
        uu = unary_union(by_day[d])
        if uu.is_empty:
            continue
        geom3857 = gpd.GeoDataFrame(geometry=[uu], crs=str(graph_crs)).to_crs(3857).geometry.iloc[0]
        col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
        layers.append((f"Coverage — policy camera Day {d + 1}", col, geom3857))
    return layers if layers else None


def _wrap_pi_scalar(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _wedge_polygons_motion_segment_utm(
    x0: float,
    y0: float,
    h0: float,
    x1: float,
    y1: float,
    h1: float,
    *,
    radius_m: float,
    hfov_deg: float,
    resolution_m: float,
) -> list[Any]:
    """Several sector wedges along one walked segment (matches env scan integration)."""
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return [_sector_wedge_polygon(x0, y0, h0, radius_m, hfov_deg)]
    step_m = max(0.5 * float(resolution_m), 0.35)
    n = max(2, min(40, int(math.ceil(dist / step_m)) + 1))
    dh = _wrap_pi_scalar(h1 - h0)
    out: list[Any] = []
    for j in range(n):
        t = j / (n - 1) if n > 1 else 0.0
        ax = x0 + t * dx
        ay = y0 + t * dy
        hd = _wrap_pi_scalar(h0 + t * dh)
        out.append(_sector_wedge_polygon(ax, ay, hd, radius_m, hfov_deg))
    return out


def _canonical_graph_walk_path(
    rec: dict[str, Any],
    *,
    x_utm: np.ndarray,
    y_utm: np.ndarray,
    h_utm: np.ndarray,
    utm_crs: str,
    graph_crs: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical walked path in graph CRS metres plus per-segment day index."""
    import geopandas as gpd

    ref_g = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            np.asarray(x_utm, dtype=np.float64),
            np.asarray(y_utm, dtype=np.float64),
        ),
        crs=utm_crs,
    ).to_crs(graph_crs)
    xg = np.asarray(ref_g.geometry.x, dtype=np.float64)
    yg = np.asarray(ref_g.geometry.y, dtype=np.float64)
    hg = np.asarray(h_utm, dtype=np.float64).ravel()
    nseg = max(0, int(xg.size) - 1)
    day_seg = _canonical_day_segment_indices_from_meta(rec, nseg)
    if day_seg is None:
        day_seg = _moving_segment_day_indices_playback(rec, nseg)
    if day_seg is None:
        day_seg = np.zeros(nseg, dtype=np.int32)
    return xg, yg, hg, day_seg


def _simulate_sector_walk_projected(
    *,
    x_path: np.ndarray,
    y_path: np.ndarray,
    h_path: np.ndarray,
    path_crs: str,
    day_seg: np.ndarray,
    on_mask: np.ndarray,
    stamps: np.ndarray,
    cfg: AdaptiveScanningConfig,
    zoom_bounds_3857: tuple[float, float, float, float] | None = None,
    nx: int = 640,
    ny: int = 640,
) -> tuple[Any, dict[int, Any], tuple[np.ndarray, tuple[float, float, float, float], int] | None]:
    """Single source of truth for coverage polygons and map-age from one stepped wedge simulation."""
    import geopandas as gpd
    from shapely.ops import unary_union
    from pyproj import Transformer

    xp = np.asarray(x_path, dtype=np.float64).ravel()
    yp = np.asarray(y_path, dtype=np.float64).ravel()
    hp = np.asarray(h_path, dtype=np.float64).ravel()
    nseg = max(0, int(xp.size) - 1)
    on = np.asarray(on_mask, dtype=bool).ravel()[:nseg]
    day_arr = np.asarray(day_seg, dtype=np.int32).ravel()[:nseg]
    st = np.asarray(stamps, dtype=np.float64).ravel()[:nseg]
    if nseg < 1 or yp.size != xp.size or hp.size != xp.size or on.size < nseg or day_arr.size < nseg or st.size < nseg:
        return unary_union([]), {}, None

    r_m = float(cfg.scan_radius_m)
    half = math.radians(0.5 * float(cfg.hfov_deg))
    res_m = float(cfg.resolution_m)
    final_t = float(np.max(st) + float(cfg.dt_s)) if st.size else float(cfg.dt_s)

    build_age = zoom_bounds_3857 is not None
    if build_age:
        zx0, zy0, zx1, zy1 = zoom_bounds_3857
        xmin, xmax = (min(zx0, zx1), max(zx0, zx1))
        ymin, ymax = (min(zy0, zy1), max(zy0, zy1))
        gx = np.linspace(xmin, xmax, nx, dtype=np.float64)
        gy = np.linspace(ymin, ymax, ny, dtype=np.float64)
        Wx, Wy = np.meshgrid(gx, gy)
        flat_m = np.column_stack([Wx.ravel(), Wy.ravel()])
        gdfg = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(flat_m[:, 0], flat_m[:, 1]),
            crs=3857,
        ).to_crs(path_crs)
        PX = np.asarray(gdfg.geometry.x, dtype=np.float64).reshape(Wx.shape)
        PY = np.asarray(gdfg.geometry.y, dtype=np.float64).reshape(Wy.shape)
        last_scan = np.full((ny, nx), -np.inf, dtype=np.float64)
        merc_pad = max(r_m * 2.5, 85.0)
    else:
        xmin = xmax = ymin = ymax = 0.0
        gx = gy = Wx = Wy = PX = PY = last_scan = merc_pad = None  # type: ignore[assignment]

    all_parts: list[Any] = []
    by_day_parts: dict[int, list[Any]] = {}
    step_m = max(0.5 * res_m, 0.35)

    for k in range(nseg):
        if not bool(on[k]):
            continue
        d = int(day_arr[k])
        stamp = float(st[k])
        x0, y0, h0 = float(xp[k]), float(yp[k]), float(hp[k])
        x1, y1, h1 = float(xp[k + 1]), float(yp[k + 1]), float(hp[k + 1])
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        ns = 1 if dist < 1e-6 else max(2, min(40, int(math.ceil(dist / step_m)) + 1))
        dh = _wrap_pi_scalar(h1 - h0)

        if build_age:
            seg3857 = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy([x0, x1], [y0, y1]),
                crs=path_crs,
            ).to_crs(3857)
            sx = np.asarray(seg3857.geometry.x, dtype=np.float64)
            sy = np.asarray(seg3857.geometry.y, dtype=np.float64)
            j0 = max(0, int(np.searchsorted(gx, float(np.min(sx)) - merc_pad)) - 1)
            j1 = min(nx - 1, int(np.searchsorted(gx, float(np.max(sx)) + merc_pad)) + 1)
            i0 = max(0, int(np.searchsorted(gy, float(np.min(sy)) - merc_pad)) - 1)
            i1 = min(ny - 1, int(np.searchsorted(gy, float(np.max(sy)) + merc_pad)) + 1)
            sub_px = PX[i0 : i1 + 1, j0 : j1 + 1]
            sub_py = PY[i0 : i1 + 1, j0 : j1 + 1]

        for j in range(ns):
            t = j / (ns - 1) if ns > 1 else 0.0
            ax = x0 + t * dx
            ay = y0 + t * dy
            hd = _wrap_pi_scalar(h0 + t * dh)
            poly = _sector_wedge_polygon(ax, ay, hd, r_m, float(cfg.hfov_deg))
            all_parts.append(poly)
            by_day_parts.setdefault(d, []).append(poly)

            if build_age:
                ddx = sub_px - ax
                ddy = sub_py - ay
                distm = np.hypot(ddx, ddy)
                ang = np.arctan2(ddy, ddx) - hd
                ang = _wrap_pi(ang)
                m = (distm <= r_m) & (distm >= 1e-3) & (np.abs(ang) <= half)
                slc = (slice(i0, i1 + 1), slice(j0, j1 + 1))
                sub = last_scan[slc]
                last_scan[slc] = np.where(m, np.maximum(sub, stamp), sub)

    cov_union = unary_union(all_parts) if all_parts else unary_union([])
    by_day_union = {
        d: unary_union(parts)
        for d, parts in by_day_parts.items()
        if parts
    }

    age_pack = None
    if build_age:
        cov_3857 = gpd.GeoDataFrame(geometry=[cov_union], crs=path_crs).to_crs(3857).geometry.iloc[0]
        try:
            from shapely import vectorized

            ins = vectorized.contains(cov_3857, Wx, Wy)
            if hasattr(vectorized, "touches"):
                ins = ins | vectorized.touches(cov_3857, Wx, Wy)
        except Exception:
            from shapely.geometry import Point
            from shapely.prepared import prep

            prep_c = prep(cov_3857)
            ins = np.zeros((ny, nx), dtype=bool)
            for r in range(ny):
                for c in range(nx):
                    ins[r, c] = prep_c.covers(Point(float(Wx[r, c]), float(Wy[r, c])))
        never_in_cov = ins & (last_scan < -1e90)
        last_plot = np.full_like(last_scan, np.nan, dtype=np.float64)
        hit = ins & (last_scan > -1e90)
        last_plot[hit] = last_scan[hit]
        last_plot[never_in_cov] = 0.0
        if np.any(np.isfinite(last_plot)):
            rgba = _sim_time_grid_to_rgba_red_green(last_plot, vmin=0.0, vmax=max(final_t, 1.0))
            dx_merc = (xmax - xmin) / max(nx - 1, 1)
            dy_merc = (ymax - ymin) / max(ny - 1, 1)
            xmin_e = xmin - 0.5 * dx_merc
            xmax_e = xmax + 0.5 * dx_merc
            ymin_e = ymin - 0.5 * dy_merc
            ymax_e = ymax + 0.5 * dy_merc
            t4326 = Transformer.from_crs(3857, 4326, always_xy=True)
            lons, lats = t4326.transform(
                [xmin_e, xmax_e, xmax_e, xmin_e],
                [ymin_e, ymin_e, ymax_e, ymax_e],
            )
            age_pack = (rgba, (min(lons), min(lats), max(lons), max(lats)), int(np.sum(hit)))
    return cov_union, by_day_union, age_pack


def _map_extent_webmerc_from_cfg(
    cfg: AdaptiveScanningConfig,
    route_line_3857: Any,
) -> tuple[float, float, float, float]:
    """City / bbox extent in EPSG:3857 for basemap; fallback to padded route."""
    import geopandas as gpd
    from shapely.geometry import box

    from adaptive_scanning.street_trajectories import DEFAULT_OSM_PLACE

    if cfg.osm_bbox is not None:
        w, s, e, n = cfg.osm_bbox
        b = gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs=4326).to_crs(3857)
        return tuple(float(x) for x in b.total_bounds)

    place = (cfg.osm_place or "").strip() or DEFAULT_OSM_PLACE
    try:
        import osmnx as ox

        pl = ox.geocode_to_gdf(place).to_crs(3857)
        return tuple(float(x) for x in pl.total_bounds)
    except Exception:
        r = route_line_3857.total_bounds
        pad = 2000.0
        return (r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad)


def _bounds3857_to_wgs84(
    mx0: float, my0: float, mx1: float, my1: float
) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    t = Transformer.from_crs(3857, 4326, always_xy=True)
    lon0, lat0 = t.transform(mx0, my0)
    lon1, lat1 = t.transform(mx1, my1)
    w, e = min(lon0, lon1), max(lon0, lon1)
    s, n = min(lat0, lat1), max(lat0, lat1)
    return (w, s, e, n)


def _tile_zoom_for_bounds_3857(
    mx0: float,
    my0: float,
    mx1: float,
    my1: float,
    *,
    max_tiles: int,
) -> int:
    """Pick highest OSM zoom that stays under ``max_tiles`` for this Web-Mercator bbox."""
    import mercantile

    w, s, e, n = _bounds3857_to_wgs84(mx0, my0, mx1, my1)
    best = 10
    for z in range(10, 20):
        nt = len(list(mercantile.tiles(w, s, e, n, zooms=[z])))
        if nt > max_tiles:
            break
        best = z
    return best


def _route_zoom_bounds_3857(gwm: Any, r_m: float, *, pad_frac: float = 0.38) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = gwm.total_bounds
    span = max(xmax - xmin, ymax - ymin, 80.0)
    pad = max(span * pad_frac, r_m * 2.8, 70.0)
    return (xmin - pad, ymin - pad, xmax + pad, ymax + pad)


def _staleness_grid_to_rgba_rdylgn(
    value_grid: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: int = 238,
    stale_ref_s: float | None = None,
) -> np.ndarray:
    """
    ``value_grid`` NaN = transparent. Otherwise higher value = staler → red;
    lower = fresher → green (matplotlib ``RdYlGn``).

    If ``vmin`` / ``vmax`` are None, uses **min and max** of finite values so the
    oldest pixel in view maps to red and the newest to green.

    When ages are **nearly uniform**, expanding vmin/vmax symmetrically would map
    every pixel to RdYlGn(0.5) (yellow). Instead we derive a single staleness tone
    from the typical age vs ``stale_ref_s`` (seconds).
    """
    from matplotlib import colormaps

    cmap = colormaps["RdYlGn"]
    mnan = np.isnan(value_grid)
    valid = ~mnan
    if vmin is None or vmax is None:
        if not np.any(valid):
            rgba = np.zeros((*value_grid.shape, 4), dtype=np.uint8)
            return rgba
        vals = value_grid[valid].astype(np.float64).ravel()
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi
    assert vmin is not None and vmax is not None
    span = float(vmax) - float(vmin)
    tol = 1e-9 * max(abs(float(vmax)), abs(float(vmin)), 1.0)
    ref = float(stale_ref_s) if stale_ref_s is not None and float(stale_ref_s) > 0 else 3600.0

    stale = np.zeros_like(value_grid, dtype=np.float64)
    if span <= tol:
        # Uniform (or duplicate min=max): avoid RdYlGn midpoint yellow.
        mid = 0.5 * (float(vmin) + float(vmax))
        stale_u = float(np.clip(mid / (3.0 * ref), 0.0, 1.0))
        stale[~mnan] = stale_u
    else:
        stale[~mnan] = np.clip((value_grid[~mnan] - vmin) / (vmax - vmin), 0.0, 1.0)
    fresh = 1.0 - stale
    rgba = (cmap(np.where(mnan, 0.5, fresh)) * 255.0).astype(np.uint8)
    rgba[mnan, :] = 0
    rgba[~mnan, 3] = alpha
    return rgba


def _sim_time_grid_to_rgba_red_green(
    time_s: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    alpha: int = 238,
) -> np.ndarray:
    """
    ``time_s`` NaN = transparent.

    Finite values are mapped linearly from ``vmin`` to ``vmax`` using one continuous
    red→yellow→green ramp, where later scan times in the session are greener.
    """
    from matplotlib import colormaps

    cmap = colormaps["RdYlGn"]
    mnan = np.isnan(time_s)
    if vmax <= vmin:
        vmax = vmin + 1.0
    t = np.zeros_like(time_s, dtype=np.float64)
    t[~mnan] = np.clip((time_s[~mnan] - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (cmap(np.where(mnan, 0.5, t)) * 255.0).astype(np.uint8)
    rgba[mnan, :] = 0
    rgba[~mnan, 3] = alpha
    return rgba


def _graph_xy_to_world_xy(
    poly: np.ndarray,
    xy_graph: np.ndarray,
    *,
    world_w_m: float,
    world_h_m: float,
    margin: float = 1.0,
) -> np.ndarray:
    """Same letterbox as ``inverse_affine_world_to_graph`` / env, using **poly** for scale."""
    xy = np.asarray(poly, dtype=np.float64)
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
    ug = np.asarray(xy_graph, dtype=np.float64).reshape(-1, 2)
    ex = wx0 + (ug[:, 0] - cx) * scale
    ey = wy0 + (ug[:, 1] - cy) * scale
    return np.column_stack([ex, ey])


def _sector_scan_age_rgba_wgs84(
    rec: dict[str, Any],
    cov_geom_3857: Any,
    zoom_bounds_3857: tuple[float, float, float, float],
    *,
    always_on_reference: bool,
    xw_path: np.ndarray,
    yw_path: np.ndarray,
    h_path: np.ndarray,
    poly: np.ndarray,
    graph_crs: str,
    world_w_m: float,
    world_h_m: float,
    on_override: np.ndarray | None = None,
    stamps_override: np.ndarray | None = None,
    nx: int = 640,
    ny: int = 640,
    stamp_debug_path: Path | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float], int] | None:
    """
    Sector stamping matches ``CameraBudgetEnv`` (**world metres** + trajectory heading).

    The output raster is aligned to the Folium **EPSG:3857** zoom box; each pixel maps
    to world (x,y) via graph CRS so distance/angle tests match the simulator. Using
    Web-Mercator chord headings with Mercator positions skewed sectors and collapsed
    ``last_scan`` to a constant / zeros.

    Folium **policy** coverage uses wedge unions (see ``policy_camera_coverage_layers_3857_by_day``),
    not this raster.
    """
    import geopandas as gpd
    from pyproj import Transformer

    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    cfg: AdaptiveScanningConfig = rec["cfg"]
    final_t = float(rec.get("final_sim_time_s", 0.0))
    dt_s = float(cfg.dt_s)
    r_m = float(cfg.scan_radius_m)
    half = math.radians(0.5 * float(cfg.hfov_deg))
    res_m = float(cfg.resolution_m)
    poly_a = np.asarray(poly, dtype=np.float64)

    zx0, zy0, zx1, zy1 = zoom_bounds_3857
    xmin, xmax = (min(zx0, zx1), max(zx0, zx1))
    ymin, ymax = (min(zy0, zy1), max(zy0, zy1))
    gx = np.linspace(xmin, xmax, nx, dtype=np.float64)
    gy = np.linspace(ymin, ymax, ny, dtype=np.float64)
    Wx, Wy = np.meshgrid(gx, gy)
    flat_m = np.column_stack([Wx.ravel(), Wy.ravel()])
    gdfg = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(flat_m[:, 0], flat_m[:, 1]),
        crs=3857,
    ).to_crs(graph_crs)
    x_graph = np.asarray(gdfg.geometry.x, dtype=np.float64)
    y_graph = np.asarray(gdfg.geometry.y, dtype=np.float64)
    xy_world = _graph_xy_to_world_xy(
        poly_a,
        np.column_stack([x_graph, y_graph]),
        world_w_m=world_w_m,
        world_h_m=world_h_m,
        margin=1.0,
    )
    WWx = xy_world[:, 0].reshape(Wx.shape)
    WWy = xy_world[:, 1].reshape(Wy.shape)

    t_gr_m = Transformer.from_crs(str(graph_crs), 3857, always_xy=True)
    last_scan = np.full((ny, nx), -np.inf, dtype=np.float64)

    nseg = int(np.asarray(xw_path).shape[0]) - 1
    if nseg < 1:
        if stamp_debug_path is not None:
            stamp_debug_path.write_text(
                "status=FAIL\nreason=nseg<1 (path needs at least 2 points)\n", encoding="utf-8"
            )
        return None

    if on_override is not None:
        on = np.asarray(on_override, dtype=bool).ravel()[:nseg]
        if on.size < nseg:
            return None
        if stamps_override is not None:
            stamps = np.asarray(stamps_override, dtype=np.float64).ravel()[:nseg]
            if stamps.size < nseg:
                return None
        else:
            stamps = np.arange(nseg, dtype=np.float64) * dt_s
    elif always_on_reference:
        on = np.ones(nseg, dtype=bool)
        stamps = np.arange(nseg, dtype=np.float64) * dt_s
    else:
        on = np.asarray(rec["camera_on_effective"], dtype=np.float64).ravel()[:nseg] > 0.5
        sim_arr = np.asarray(rec["sim_time_s"], dtype=np.float64).ravel()
        if sim_arr.size < nseg:
            if stamp_debug_path is not None:
                stamp_debug_path.write_text(
                    "status=FAIL\n"
                    f"reason=sim_time_s shorter than nseg\nnseg={nseg}\n"
                    f"len(sim_time_s)={sim_arr.size}\n",
                    encoding="utf-8",
                )
            return None
        stamps = sim_arr[:nseg] - dt_s

    merc_pad = max(r_m * 2.5, 85.0)
    step_m = max(0.5 * res_m, 0.35)
    for k in range(nseg):
        if not bool(on[k]):
            continue
        stamp = float(stamps[k])
        x0, y0, h0 = float(xw_path[k]), float(yw_path[k]), float(h_path[k])
        x1, y1, h1 = float(xw_path[k + 1]), float(yw_path[k + 1]), float(h_path[k + 1])
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            ns = 1
        else:
            ns = max(2, min(40, int(math.ceil(dist / step_m)) + 1))
        dh = _wrap_pi_scalar(h1 - h0)
        for j in range(ns):
            t = j / (ns - 1) if ns > 1 else 0.0
            pw_x = x0 + t * dx
            pw_y = y0 + t * dy
            ph = _wrap_pi_scalar(h0 + t * dh)
            gxy_s = inverse_affine_world_to_graph(
                np.array([[pw_x, pw_y]], dtype=np.float64),
                poly_a,
                world_w_m=world_w_m,
                world_h_m=world_h_m,
                margin=1.0,
            )
            pmx, pmy = t_gr_m.transform(float(gxy_s[0, 0]), float(gxy_s[0, 1]))
            j0 = max(0, int(np.searchsorted(gx, pmx - merc_pad)) - 1)
            j1 = min(nx - 1, int(np.searchsorted(gx, pmx + merc_pad)) + 1)
            i0 = max(0, int(np.searchsorted(gy, pmy - merc_pad)) - 1)
            i1 = min(ny - 1, int(np.searchsorted(gy, pmy + merc_pad)) + 1)
            sub_wwx = WWx[i0 : i1 + 1, j0 : j1 + 1]
            sub_wwy = WWy[i0 : i1 + 1, j0 : j1 + 1]
            ddx = sub_wwx - pw_x
            ddy = sub_wwy - pw_y
            distm = np.hypot(ddx, ddy)
            ang = np.arctan2(ddy, ddx) - ph
            ang = _wrap_pi(ang)
            m = (distm <= r_m) & (distm >= 1e-3) & (np.abs(ang) <= half)
            slc = (slice(i0, i1 + 1), slice(j0, j1 + 1))
            sub = last_scan[slc]
            last_scan[slc] = np.where(m, np.maximum(sub, stamp), sub)

    geom_cov = cov_geom_3857
    try:
        from shapely import vectorized

        ins = vectorized.contains(geom_cov, Wx, Wy)
        if hasattr(vectorized, "touches"):
            ins = ins | vectorized.touches(geom_cov, Wx, Wy)
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep

        prep_c = prep(geom_cov)
        ins = np.zeros((ny, nx), dtype=bool)
        for r in range(ny):
            for c in range(nx):
                ins[r, c] = prep_c.covers(Point(float(Wx[r, c]), float(Wy[r, c])))
    never_in_cov = ins & (last_scan < -1e90)

    last_plot = np.full_like(last_scan, np.nan, dtype=np.float64)
    hit = ins & (last_scan > -1e90)
    last_plot[hit] = last_scan[hit]
    last_plot[never_in_cov] = 0.0

    if not np.any(np.isfinite(last_plot)):
        if stamp_debug_path is not None:
            stamp_debug_path.write_text(
                "status=FAIL\nreason=no finite last_plot values after mask\n",
                encoding="utf-8",
            )
        return None

    ft = max(final_t, 1.0)
    rgba = _sim_time_grid_to_rgba_red_green(last_plot, vmin=0.0, vmax=ft)

    if stamp_debug_path is not None:
        lines = [
            "NOTE: Green Folium coverage = always-on sector wedges along the resampled route.",
            "Policy camera coverage in HTML = motion-integrated wedges (graph m, endpoints from "
            "rec xs/ys), camera on only; GeoJSON per playback day.",
            "",
            "status=OK",
            f"mode={'always_on_reference' if always_on_reference else 'policy'}",
            f"final_sim_time_s={final_t}",
            f"dt_s={dt_s}",
            f"nseg={nseg}",
            f"nx={nx} ny={ny}",
            "",
            "Per segment k: camera_on (1=yes), stamp_sim_s = sim clock at START of interval k "
            "(same as env when applying sector; policy uses sim_time_s[k]-dt_s).",
            "When camera off, stamp is still listed but not applied to the raster.",
            "",
        ]
        for k in range(nseg):
            lines.append(f"k={k}\tcamera_on={int(on[k])}\tstamp_sim_s={float(stamps[k]):.6g}")
        lines.append("")
        if not np.any(on):
            lines.append(
                "WARNING: camera_on is 0 for every segment — no stamps are written, "
                "so last_plot is only discretization fill (0) and the heatmap is uniform."
            )
            lines.append("")
        lv = last_plot[np.isfinite(last_plot)]
        hit_m = hit & np.isfinite(last_plot)
        z_m = never_in_cov
        lines.append(
            f"raster cells: total={nx * ny} inside_cov={int(np.sum(ins))} "
            f"hit_by_sector={int(np.sum(hit_m))} never_hit_fill0={int(np.sum(z_m))} "
            f"outside_cov={int(nx * ny - np.sum(ins))}"
        )
        if lv.size:
            lines.append(
                f"last_plot (sim_s used for red-green color, 0=discretization gap): "
                f"min={float(np.nanmin(lv)):.6g} max={float(np.nanmax(lv)):.6g} "
                f"mean={float(np.nanmean(lv)):.6g} p5={float(np.percentile(lv, 5)):.6g} "
                f"p95={float(np.percentile(lv, 95)):.6g}"
            )
        lines.append("")
        lines.append(f"session_time_vmin=0 session_time_vmax={ft}")
        stamp_debug_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_debug_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dx_merc = (xmax - xmin) / max(nx - 1, 1)
    dy_merc = (ymax - ymin) / max(ny - 1, 1)
    xmin_e = xmin - 0.5 * dx_merc
    xmax_e = xmax + 0.5 * dx_merc
    ymin_e = ymin - 0.5 * dy_merc
    ymax_e = ymax + 0.5 * dy_merc

    t4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    lons, lats = t4326.transform(
        [xmin_e, xmax_e, xmax_e, xmin_e],
        [ymin_e, ymin_e, ymax_e, ymax_e],
    )
    w, e = min(lons), max(lons)
    s, nlat = min(lats), max(lats)
    n_hit = int(np.sum(hit))
    return rgba, (w, s, e, nlat), n_hit


def _sector_scan_age_rgba_projected_wgs84(
    rec: dict[str, Any],
    cov_geom_3857: Any,
    zoom_bounds_3857: tuple[float, float, float, float],
    *,
    x_path: np.ndarray,
    y_path: np.ndarray,
    h_path: np.ndarray,
    path_crs: str,
    on_mask: np.ndarray,
    stamps: np.ndarray,
    nx: int = 640,
    ny: int = 640,
) -> tuple[np.ndarray, tuple[float, float, float, float], int] | None:
    """Projected-CRS sector stamping that matches the vector wedge geometry exactly."""
    import geopandas as gpd

    cfg: AdaptiveScanningConfig = rec["cfg"]
    final_t = float(rec.get("final_sim_time_s", 0.0))
    r_m = float(cfg.scan_radius_m)
    half = math.radians(0.5 * float(cfg.hfov_deg))
    res_m = float(cfg.resolution_m)

    xp = np.asarray(x_path, dtype=np.float64).ravel()
    yp = np.asarray(y_path, dtype=np.float64).ravel()
    hp = np.asarray(h_path, dtype=np.float64).ravel()
    if xp.size < 2 or yp.size != xp.size or hp.size != xp.size:
        return None
    nseg = int(xp.size) - 1
    on = np.asarray(on_mask, dtype=bool).ravel()[:nseg]
    st = np.asarray(stamps, dtype=np.float64).ravel()[:nseg]
    if on.size < nseg or st.size < nseg:
        return None

    zx0, zy0, zx1, zy1 = zoom_bounds_3857
    xmin, xmax = (min(zx0, zx1), max(zx0, zx1))
    ymin, ymax = (min(zy0, zy1), max(zy0, zy1))
    gx = np.linspace(xmin, xmax, nx, dtype=np.float64)
    gy = np.linspace(ymin, ymax, ny, dtype=np.float64)
    Wx, Wy = np.meshgrid(gx, gy)
    flat_m = np.column_stack([Wx.ravel(), Wy.ravel()])
    gdfg = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(flat_m[:, 0], flat_m[:, 1]),
        crs=3857,
    ).to_crs(path_crs)
    PX = np.asarray(gdfg.geometry.x, dtype=np.float64).reshape(Wx.shape)
    PY = np.asarray(gdfg.geometry.y, dtype=np.float64).reshape(Wy.shape)
    last_scan = np.full((ny, nx), -np.inf, dtype=np.float64)

    merc_pad = max(r_m * 2.5, 85.0)
    step_m = max(0.5 * res_m, 0.35)
    for k in range(nseg):
        if not bool(on[k]):
            continue
        stamp = float(st[k])
        x0, y0, h0 = float(xp[k]), float(yp[k]), float(hp[k])
        x1, y1, h1 = float(xp[k + 1]), float(yp[k + 1]), float(hp[k + 1])
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        ns = 1 if dist < 1e-6 else max(2, min(40, int(math.ceil(dist / step_m)) + 1))
        dh = _wrap_pi_scalar(h1 - h0)

        seg3857 = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([x0, x1], [y0, y1]),
            crs=path_crs,
        ).to_crs(3857)
        sx = np.asarray(seg3857.geometry.x, dtype=np.float64)
        sy = np.asarray(seg3857.geometry.y, dtype=np.float64)
        j0 = max(0, int(np.searchsorted(gx, float(np.min(sx)) - merc_pad)) - 1)
        j1 = min(nx - 1, int(np.searchsorted(gx, float(np.max(sx)) + merc_pad)) + 1)
        i0 = max(0, int(np.searchsorted(gy, float(np.min(sy)) - merc_pad)) - 1)
        i1 = min(ny - 1, int(np.searchsorted(gy, float(np.max(sy)) + merc_pad)) + 1)
        sub_px = PX[i0 : i1 + 1, j0 : j1 + 1]
        sub_py = PY[i0 : i1 + 1, j0 : j1 + 1]

        for j in range(ns):
            t = j / (ns - 1) if ns > 1 else 0.0
            px = x0 + t * dx
            py = y0 + t * dy
            ph = _wrap_pi_scalar(h0 + t * dh)
            ddx = sub_px - px
            ddy = sub_py - py
            distm = np.hypot(ddx, ddy)
            ang = np.arctan2(ddy, ddx) - ph
            ang = _wrap_pi(ang)
            m = (distm <= r_m) & (distm >= 1e-3) & (np.abs(ang) <= half)
            slc = (slice(i0, i1 + 1), slice(j0, j1 + 1))
            sub = last_scan[slc]
            last_scan[slc] = np.where(m, np.maximum(sub, stamp), sub)

    geom_cov = cov_geom_3857
    try:
        from shapely import vectorized

        ins = vectorized.contains(geom_cov, Wx, Wy)
        if hasattr(vectorized, "touches"):
            ins = ins | vectorized.touches(geom_cov, Wx, Wy)
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep

        prep_c = prep(geom_cov)
        ins = np.zeros((ny, nx), dtype=bool)
        for r in range(ny):
            for c in range(nx):
                ins[r, c] = prep_c.covers(Point(float(Wx[r, c]), float(Wy[r, c])))

    never_in_cov = ins & (last_scan < -1e90)
    last_plot = np.full_like(last_scan, np.nan, dtype=np.float64)
    hit = ins & (last_scan > -1e90)
    last_plot[hit] = last_scan[hit]
    last_plot[never_in_cov] = 0.0
    if not np.any(np.isfinite(last_plot)):
        return None

    rgba = _sim_time_grid_to_rgba_red_green(last_plot, vmin=0.0, vmax=max(final_t, 1.0))
    dx_merc = (xmax - xmin) / max(nx - 1, 1)
    dy_merc = (ymax - ymin) / max(ny - 1, 1)
    xmin_e = xmin - 0.5 * dx_merc
    xmax_e = xmax + 0.5 * dx_merc
    ymin_e = ymin - 0.5 * dy_merc
    ymax_e = ymax + 0.5 * dy_merc

    t4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    lons, lats = t4326.transform(
        [xmin_e, xmax_e, xmax_e, xmin_e],
        [ymin_e, ymin_e, ymax_e, ymax_e],
    )
    w, e = min(lons), max(lons)
    s, nlat = min(lats), max(lats)
    return rgba, (w, s, e, nlat), int(np.sum(hit))


def _save_coverage_mpl_png(
    out_path: Path,
    mx0: float,
    my0: float,
    mx1: float,
    my1: float,
    cov_gdf: Any,
    gwm: Any,
    *,
    tile_zoom: int,
    fig_inches: float,
    dpi_save: int,
    title: str,
    rec: dict[str, Any] | None = None,
) -> None:
    import contextily as ctx
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi_save)
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    ax.set_aspect("equal", adjustable="box")

    crs_3857 = gwm.crs
    try:
        ctx.add_basemap(
            ax,
            crs=crs_3857,
            zoom=tile_zoom,
            source=ctx.providers.OpenStreetMap.Mapnik,
            attribution_size=6,
        )
    except Exception:
        try:
            ctx.add_basemap(
                ax,
                crs=crs_3857,
                zoom=tile_zoom,
                source=ctx.providers.CartoDB.Positron,
                attribution_size=6,
            )
        except Exception:
            ctx.add_basemap(ax, crs=crs_3857, zoom=tile_zoom, attribution_size=6)

    cov_gdf.plot(ax=ax, color="#2ecc71", alpha=0.52, edgecolor="none", zorder=4)
    hm = rec.get("osm_home_daily_meta") if rec is not None else None
    gcrs = str(rec["graph_crs"]) if rec is not None and rec.get("graph_crs") else None
    day_polys = hm.get("day_polylines_graph_m") if isinstance(hm, dict) else None
    if hm and gcrs and day_polys:
        plot_home_daily_paths_mercator_ax(ax, day_polys, gcrs, linewidth=3.2)
        plot_home_daily_markers_mercator_ax(ax, hm, gcrs, legend=True)
    else:
        gwm.plot(ax=ax, color="red", linewidth=3.2, alpha=0.95, zorder=6)
    if not (hm and gcrs and day_polys):
        x0, y0 = gwm.geometry.iloc[0].coords[0]
        x1, y1 = gwm.geometry.iloc[0].coords[-1]
        ax.scatter([x0], [y0], c="lime", s=55, zorder=7, edgecolors="k", linewidths=0.5)
        ax.scatter([x1], [y1], c="yellow", s=55, marker="s", zorder=7, edgecolors="k", linewidths=0.5)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=dpi_save, bbox_inches="tight")
    plt.close(fig)


def folium_feature_group_policy_decisions(rec: dict[str, Any]) -> Any | None:
    """
    Folium layer: one small dot at each env step where ``policy.act`` ran (``rec["xs"], rec["ys"]``
    at the step index, same frame as ``action_requested``).

    Green = requested camera on; gray = requested off; orange = requested on but not effectively on
    (budget / stationary clamp). Requires ``polyline_graph_m`` + ``graph_crs`` for world→graph affine.
    """
    try:
        import folium
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        return None

    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    poly = rec.get("polyline_graph_m")
    crs = rec.get("graph_crs")
    if poly is None or crs is None:
        return None
    xs = np.asarray(rec.get("xs"), dtype=np.float64).ravel()
    ys = np.asarray(rec.get("ys"), dtype=np.float64).ravel()
    act = np.asarray(rec.get("action_requested"), dtype=np.int32).ravel()
    if xs.size < 1 or act.size < 1:
        return None
    n = int(min(act.size, xs.size))
    if n < 1:
        return None
    sim = np.asarray(rec.get("sim_time_s"), dtype=np.float64).ravel()
    eff = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()
    cfg: AdaptiveScanningConfig = rec["cfg"]
    world_w_m = float(cfg.nx) * float(cfg.resolution_m)
    world_h_m = float(cfg.ny) * float(cfg.resolution_m)
    poly_a = np.asarray(poly, dtype=np.float64)
    wxy = np.column_stack([xs[:n], ys[:n]])
    gxy = inverse_affine_world_to_graph(
        wxy,
        poly_a,
        world_w_m=world_w_m,
        world_h_m=world_h_m,
        margin=1.0,
    )
    pts = [Point(float(gxy[k, 0]), float(gxy[k, 1])) for k in range(n)]
    steps = np.arange(n, dtype=np.int32)
    gdf = gpd.GeoDataFrame(
        {"step": steps, "action": act[:n].copy()},
        geometry=pts,
        crs=str(crs),
    ).to_crs(4326)
    t_end = sim[:n].astype(np.float64) if sim.size >= n else np.full(n, np.nan, dtype=np.float64)
    cam_e = eff[:n].astype(np.float64) if eff.size >= n else np.full(n, np.nan, dtype=np.float64)

    fg = folium.FeatureGroup(name="Policy decisions (dot = each env step)", show=True)
    for k in range(n):
        row = gdf.iloc[k]
        geom = row.geometry
        lat, lon = float(geom.y), float(geom.x)
        a = int(row["action"])
        te = float(t_end[k]) if np.isfinite(t_end[k]) else float("nan")
        ce = float(cam_e[k]) if np.isfinite(cam_e[k]) else -1.0
        wanted_on = a == 1
        actually_on = ce > 0.5
        color = "#27ae60" if wanted_on else "#7f8c8d"
        if ce >= 0.0 and wanted_on and not actually_on:
            color = "#e67e22"
        popup = f"step {k}, action={a}, cam_on_effective={ce:.0f}, sim_time_end_s≈{te:.1f}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#1c2833",
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.85,
            popup=folium.Popup(popup, max_width=240),
        ).add_to(fg)
    return fg


def folium_feature_group_sim_env_grid(rec: dict[str, Any]) -> Any | None:
    """
    Folium layer: **simulation** grid lines (``cfg.nx`` × ``cfg.ny`` × ``resolution_m`` world metres),
    same letterboxed frame as ``last_seen`` / greedy foot cell indexing, mapped through
    ``inverse_affine_world_to_graph`` so they align with the OSM route map.

    Requires ``polyline_graph_m`` and ``graph_crs`` (same as policy decision dots).
    """
    try:
        import folium
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        return None

    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    poly = rec.get("polyline_graph_m")
    crs = rec.get("graph_crs")
    if poly is None or crs is None:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    nx = int(cfg.nx)
    ny = int(cfg.ny)
    res = float(cfg.resolution_m)
    world_w_m = float(nx) * res
    world_h_m = float(ny) * res
    poly_a = np.asarray(poly, dtype=np.float64)

    lines_w: list[np.ndarray] = []
    for j in range(nx + 1):
        x = float(j) * res
        lines_w.append(np.array([[x, 0.0], [x, world_h_m]], dtype=np.float64))
    for i in range(ny + 1):
        y = float(i) * res
        lines_w.append(np.array([[0.0, y], [world_w_m, y]], dtype=np.float64))

    fg = folium.FeatureGroup(
        name=f"Sim grid ({nx}×{ny} @ {res:g} m — last_seen / greedy cells)",
        show=False,
    )
    for wxy in lines_w:
        gxy = inverse_affine_world_to_graph(
            wxy,
            poly_a,
            world_w_m=world_w_m,
            world_h_m=world_h_m,
            margin=1.0,
        )
        ls = LineString(gxy)
        gdf = gpd.GeoDataFrame(geometry=[ls], crs=str(crs)).to_crs(4326)
        geom = gdf.geometry.iloc[0]
        coords = [(float(lat), float(lon)) for lon, lat in geom.coords]
        folium.PolyLine(
            locations=coords,
            color="#7b2cbf",
            weight=1,
            opacity=0.65,
        ).add_to(fg)
    return fg


def _last_seen_stamp_folium_style(
    ls: float,
    *,
    t_final: float,
) -> dict[str, str | float]:
    """
    Fill style for one env cell from ``last_seen`` **stamp** (sim seconds), same semantics as map-age
    raster: RdYlGn with later stamps greener; never scanned = neutral gray.
    """
    from matplotlib import colormaps

    cmap = colormaps["RdYlGn"]
    vmax = max(float(t_final), 1.0)
    if not np.isfinite(ls) or float(ls) < -1e90:
        return {
            "fillColor": "#bdc3c7",
            "fillOpacity": 0.22,
            "color": "#7f8c8d",
            "weight": 0.15,
        }
    u = float(np.clip((float(ls) - 0.0) / vmax, 0.0, 1.0))
    r, g, b, _ = cmap(u)
    hx = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return {
        "fillColor": hx,
        "fillOpacity": 0.52,
        "color": "#34495e",
        "weight": 0.2,
    }


def folium_feature_group_env_last_seen_cells(rec: dict[str, Any]) -> Any | None:
    """
    Folium layer: each **env raster cell** as a filled polygon in map coordinates, colored by
    ``rec["last_seen"][iy, ix]`` **stamp time** (sim s) at episode end — same red→yellow→green idea
    as the map-age overlay (newer scan → greener). Never-scanned cells are light gray.

    Uses the same world→graph affine as the route. Requires ``last_seen`` in ``rec``,
    ``polyline_graph_m``, and ``graph_crs``.
    """
    try:
        import folium
        import geopandas as gpd
        from shapely.geometry import Polygon, mapping
    except ImportError:
        return None

    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    poly = rec.get("polyline_graph_m")
    crs = rec.get("graph_crs")
    ls_arr = rec.get("last_seen")
    if poly is None or crs is None or ls_arr is None:
        return None
    last_seen = np.asarray(ls_arr, dtype=np.float64)
    if last_seen.ndim != 2:
        return None
    cfg: AdaptiveScanningConfig = rec["cfg"]
    nx = int(cfg.nx)
    ny = int(cfg.ny)
    res = float(cfg.resolution_m)
    if last_seen.shape[0] != ny or last_seen.shape[1] != nx:
        return None
    world_w_m = float(nx) * res
    world_h_m = float(ny) * res
    poly_a = np.asarray(poly, dtype=np.float64)
    t_final = float(rec.get("final_sim_time_s", 0.0))

    features: list[dict[str, Any]] = []
    for iy in range(ny):
        for ix in range(nx):
            x0, x1 = float(ix) * res, float(ix + 1) * res
            y0, y1 = float(iy) * res, float(iy + 1) * res
            wxy = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                dtype=np.float64,
            )
            gxy = inverse_affine_world_to_graph(
                wxy,
                poly_a,
                world_w_m=world_w_m,
                world_h_m=world_h_m,
                margin=1.0,
            )
            ga = np.asarray(gxy, dtype=np.float64)
            poly_g = Polygon([(float(ga[i, 0]), float(ga[i, 1])) for i in range(int(ga.shape[0]))])
            if not poly_g.is_valid:
                poly_g = poly_g.buffer(0)
            if poly_g.is_empty:
                continue
            gdf1 = gpd.GeoDataFrame(geometry=[poly_g], crs=str(crs)).to_crs(4326)
            geom4326 = gdf1.geometry.iloc[0]
            ls = float(last_seen[iy, ix])
            st = _last_seen_stamp_folium_style(ls, t_final=t_final)
            age_v = float(t_final - ls) if np.isfinite(ls) and ls > -1e90 else float("nan")
            ls_prop = (
                f"{float(ls):.2f}"
                if np.isfinite(ls) and float(ls) > -1e90
                else "never"
            )
            age_prop = f"{age_v:.2f}" if np.isfinite(age_v) else "—"
            feat: dict[str, Any] = {
                "type": "Feature",
                "geometry": mapping(geom4326),
                "properties": {
                    "ix": ix,
                    "iy": iy,
                    "last_seen_s": ls_prop,
                    "age_s": age_prop,
                    "fillColor": st["fillColor"],
                    "fillOpacity": st["fillOpacity"],
                    "stroke": st["color"],
                    "strokeWeight": st["weight"],
                },
            }
            features.append(feat)

    if not features:
        return None
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    fg = folium.FeatureGroup(
        name=f"Env last_seen cells ({nx}×{ny}, stamp time RdYlGn @ t_end={t_final:.0f}s)",
        show=False,
    )

    def _style(f: dict[str, Any]) -> dict[str, Any]:
        p = f["properties"]
        return {
            "fillColor": p["fillColor"],
            "fillOpacity": float(p["fillOpacity"]),
            "color": p["stroke"],
            "weight": float(p["strokeWeight"]),
        }

    folium.GeoJson(
        data=fc,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["ix", "iy", "last_seen_s", "age_s"],
            aliases=["ix", "iy", "last_seen (sim s)", "age (s)"],
            sticky=True,
        ),
    ).add_to(fg)
    return fg


def try_save_realworld_always_on_coverage(
    rec: dict[str, Any],
    base_image_path: str | Path,
) -> tuple[Path, Path, Path | None] | None:
    """
    Writes:
      1) ``*_coverage_realworld.png`` — full place/bbox extent, sharp OSM tiles
      2) ``*_coverage_realworld_zoom.png`` — cropped around the route
      3) ``*_coverage_realworld_map.html`` — Folium (if ``folium`` installed): path,
         **always-on** coverage (green), **per-day** always-on wedges when multi-day playback,
         **per-day policy** wedge unions when the camera is on (resampled route + stationary scans),
         **policy decision dots** (one per env step at the agent position when the action was chosen),
         optional **sim grid** (``nx``×``ny`` × ``resolution_m`` in world metres, layer off by default),
         optional **env ``last_seen`` cells** (filled polygons colored by scan stamp, layer off by default),
         optional **map-age** raster (session-time red→green).

    **Coverage vs map age:** the green GeoJSON is motion-integrated **UTM** wedges along the
    resampled OSM polyline (with ``stride`` when ``n_scan`` is large), in **graph CRS** — the same
    geometry family as the route ``LineString(poly)`` drawn on the map. **Policy** wedges use the
    same resampled walk plus optional stationary scan points from playback, with camera-on masks
    aligned per-day via ``_policy_on_day_segments_from_playback`` when home-daily metadata exists.
    Map age is the **grid** ``last_scan`` heatmap from ``rec["xs"], rec["ys"]`` (world path).

    Returns ``(full_png, zoom_png, html_or_none)`` or ``None`` if prerequisites missing.
    """
    poly = rec.get("polyline_graph_m")
    crs = rec.get("graph_crs")
    xs = rec.get("xs")
    if poly is None or crs is None or poly.shape[0] < 2 or xs is None:
        return None
    if len(xs) < 2:
        return None
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
        from shapely.ops import unary_union
    except ImportError:
        return None

    from adaptive_scanning.street_trajectories import resample_polyline_at_speed

    cfg: AdaptiveScanningConfig = rec["cfg"]
    base_image_path = Path(base_image_path)
    base_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_full = base_image_path.with_name(
        f"{base_image_path.stem}_coverage_realworld{base_image_path.suffix}"
    )
    out_zoom = base_image_path.with_name(
        f"{base_image_path.stem}_coverage_realworld_zoom{base_image_path.suffix}"
    )
    out_html = base_image_path.with_name(
        f"{base_image_path.stem}_coverage_realworld_map.html"
    )

    line = LineString(poly)
    gdf_line = gpd.GeoDataFrame(geometry=[line], crs=crs)
    gwm = gdf_line.to_crs(3857)

    try:
        utm_crs = gdf_line.estimate_utm_crs()
    except Exception:
        utm_crs = "EPSG:32619"
    g_utm = gdf_line.to_crs(utm_crs)
    coords = np.asarray(g_utm.geometry.iloc[0].coords, dtype=np.float64)

    n_out = int(len(xs) - 1)
    x, y, h = resample_polyline_at_speed(
        coords,
        speed_m_s=float(cfg.walk_speed_m_s),
        dt_s=float(cfg.dt_s),
        n_out=max(1, n_out),
        repeat_path=False,
    )

    r_m = float(cfg.scan_radius_m)
    day_resampled = _home_daily_resampled_polylines_graph(rec)
    if day_resampled is not None:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        h_parts: list[np.ndarray] = []
        day_parts: list[np.ndarray] = []
        on_all_parts: list[np.ndarray] = []
        on_policy_parts: list[np.ndarray] = []
        stamp_parts: list[np.ndarray] = []
        for d, x_d, y_d, h_d in day_resampled:
            x_d = np.asarray(x_d, dtype=np.float64).ravel()
            y_d = np.asarray(y_d, dtype=np.float64).ravel()
            h_d = np.asarray(h_d, dtype=np.float64).ravel()
            if x_d.size < 2 or y_d.size != x_d.size or h_d.size != x_d.size:
                continue
            nseg_d = int(x_d.size) - 1
            on_d = _policy_on_day_segments_from_playback(rec, day_index=int(d), nseg_day=nseg_d)
            if on_d is None:
                on_d = np.zeros(nseg_d, dtype=bool)
            stamps_d = _segment_stamp_times_for_day_from_playback(rec, day_index=int(d), nseg_day=nseg_d)
            if stamps_d is None:
                stamps_d = np.arange(nseg_d, dtype=np.float64) * float(cfg.dt_s)
            if x_parts:
                x_parts.append(x_d[1:])
                y_parts.append(y_d[1:])
                h_parts.append(h_d[1:])
            else:
                x_parts.append(x_d)
                y_parts.append(y_d)
                h_parts.append(h_d)
            day_parts.append(np.full(nseg_d, int(d), dtype=np.int32))
            on_all_parts.append(np.ones(nseg_d, dtype=bool))
            on_policy_parts.append(np.asarray(on_d, dtype=bool))
            stamp_parts.append(np.asarray(stamps_d, dtype=np.float64))
        if not x_parts or not day_parts:
            return None
        xg_walk = np.concatenate(x_parts)
        yg_walk = np.concatenate(y_parts)
        hg_walk = np.concatenate(h_parts)
        day_seg_walk = np.concatenate(day_parts)
        on_all = np.concatenate(on_all_parts)
        on_policy = np.concatenate(on_policy_parts)
        stamps_all = np.concatenate(stamp_parts)
        stamps_policy = stamps_all.copy()
        n_scan = int(day_seg_walk.size)
    else:
        walk_graph_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
            ),
            crs=str(utm_crs),
        ).to_crs(str(crs))
        xg_walk = np.asarray(walk_graph_gdf.geometry.x, dtype=np.float64)
        yg_walk = np.asarray(walk_graph_gdf.geometry.y, dtype=np.float64)
        hg_walk = np.empty(int(xg_walk.size), dtype=np.float64)
        if xg_walk.size >= 2:
            for ii in range(int(xg_walk.size) - 1):
                hg_walk[ii] = math.atan2(
                    float(yg_walk[ii + 1] - yg_walk[ii]),
                    float(xg_walk[ii + 1] - xg_walk[ii]),
                )
            hg_walk[-1] = hg_walk[-2]
        else:
            hg_walk[:] = 0.0
        n_scan = max(0, int(xg_walk.shape[0]) - 1)
        day_seg_walk = _canonical_day_segment_indices_from_meta(rec, n_scan)
        if day_seg_walk is None:
            day_seg_walk = _moving_segment_day_indices_playback(rec, n_scan)
        if day_seg_walk is None:
            day_seg_walk = np.zeros(n_scan, dtype=np.int32)
        n_move = _moving_segment_count_from_playback(rec, n_scan)
        on_all = np.ones(n_scan, dtype=bool)
        if n_move is not None and 0 <= int(n_move) < n_scan:
            on_all[:] = False
            on_all[: int(n_move)] = True
        dt_s = float(cfg.dt_s)
        stamps_all = _moving_segment_stamp_times_playback(rec, n_scan)
        if stamps_all is None:
            stamps_all = np.arange(n_scan, dtype=np.float64) * dt_s
        on_policy = _policy_on_moving_segments(rec, n_scan)
        if on_policy is None:
            on_policy = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()[:n_scan] > 0.5
        stamps_policy = stamps_all.copy()

    coverage_graph, cov_by_day_graph, _age_unused = _simulate_sector_walk_projected(
        x_path=xg_walk,
        y_path=yg_walk,
        h_path=hg_walk,
        path_crs=str(crs),
        day_seg=day_seg_walk,
        on_mask=on_all,
        stamps=stamps_all,
        cfg=cfg,
    )
    cov_gdf = gpd.GeoDataFrame(geometry=[coverage_graph], crs=str(crs)).to_crs(3857)
    cov_geom = cov_gdf.geometry.iloc[0]
    path_geom = gwm.geometry.iloc[0]

    stationary_policy_scans = _stationary_policy_scan_points_from_playback(rec)
    (
        xg_policy,
        yg_policy,
        hg_policy,
        day_seg_policy,
        on_policy_aug,
        stamps_policy_aug,
    ) = _append_stationary_segments_to_simulation(
        x_path=xg_walk,
        y_path=yg_walk,
        h_path=hg_walk,
        day_seg=day_seg_walk,
        on_mask=on_policy,
        stamps=stamps_policy,
        stationary_scans=stationary_policy_scans,
    )

    debug_steps_csv = base_image_path.with_name(f"{base_image_path.stem}_policy_scan_debug_steps.csv")
    debug_segments_csv = base_image_path.with_name(f"{base_image_path.stem}_policy_scan_debug_segments.csv")
    debug_stamps_txt = base_image_path.with_name(f"{base_image_path.stem}_coverage_realworld_map_stamps.txt")
    act_req = np.asarray(rec.get("action_requested"), dtype=np.int32).ravel()
    act_eff = np.asarray(rec.get("camera_on_effective"), dtype=np.float64).ravel()
    move_eff = np.asarray(rec.get("step_interval_is_moving"), dtype=np.float64).ravel()
    budget_arr = np.asarray(rec.get("budget_s"), dtype=np.float64).ravel()
    sim_arr = np.asarray(rec.get("sim_time_s"), dtype=np.float64).ravel()
    step_day_idx = _segment_day_indices_playback(rec, int(sim_arr.size))
    if step_day_idx is None:
        step_day_idx = np.full(int(sim_arr.size), -1, dtype=np.int32)
    lines_steps = [
        "env_step_idx,day_index,step_start_s,sim_time_end_s,step_interval_is_moving,action_requested,camera_on_effective,budget_s_after_step"
    ]
    for k in range(int(sim_arr.size)):
        day_k = int(step_day_idx[k]) if k < step_day_idx.size else -1
        step_start_k = float(sim_arr[k] - float(cfg.dt_s))
        sim_end_k = float(sim_arr[k])
        moving_k = int(move_eff[k] > 0.5) if k < move_eff.size else 1
        act_req_k = int(act_req[k]) if k < act_req.size else -1
        act_eff_k = int(act_eff[k] > 0.5) if k < act_eff.size else 0
        bud_k = float(budget_arr[k]) if k < budget_arr.size else float("nan")
        lines_steps.append(
            f"{k},{day_k},{step_start_k:.6f},{sim_end_k:.6f},{moving_k},{act_req_k},{act_eff_k},{bud_k:.6f}"
        )
    debug_steps_csv.write_text("\n".join(lines_steps) + "\n", encoding="utf-8")

    lines_segments = [
        "seg_idx,day_index,stamp_start_s,policy_on_mask,always_on_mask"
    ]
    n_policy_seg = max(0, int(day_seg_policy.size))
    for k in range(n_policy_seg):
        day_k = int(day_seg_policy[k]) if k < day_seg_policy.size else -1
        stamp_k = float(stamps_policy_aug[k]) if k < stamps_policy_aug.size else float(k) * float(cfg.dt_s)
        pol_on_k = int(on_policy_aug[k]) if k < on_policy_aug.size else 0
        all_on_k = int(on_all[k]) if k < on_all.size else 0
        lines_segments.append(
            f"{k},{day_k},{stamp_k:.6f},{pol_on_k},{all_on_k}"
        )
    debug_segments_csv.write_text("\n".join(lines_segments) + "\n", encoding="utf-8")

    day_counts: dict[int, int] = {}
    for k in range(n_policy_seg):
        if k >= on_policy_aug.size or not bool(on_policy_aug[k]):
            continue
        d = int(day_seg_policy[k]) if k < day_seg_policy.size else -1
        day_counts[d] = int(day_counts.get(d, 0) + 1)
    lines_txt = [
        "status=OK",
        "mode=policy_resampled_route_plus_stationary",
        f"final_sim_time_s={float(rec.get('final_sim_time_s', 0.0))}",
        f"dt_s={float(cfg.dt_s)}",
        f"nseg={n_policy_seg}",
        "note=Green/policy wedges use resampled graph polyline (same as map route); policy adds stationary scans from playback.",
        "",
        f"policy_on_steps={int(np.sum(on_policy_aug.astype(np.int32)))}",
        f"policy_on_seconds={int(np.sum(on_policy_aug.astype(np.int32)) * int(round(float(cfg.dt_s))))}",
        f"policy_on_by_day_steps={json.dumps({int(k): int(v) for k, v in sorted(day_counts.items())})}",
        "",
        "Per segment k: day_index, moving_mask(always_on), policy_on, stamp_start_s",
    ]
    for k in range(n_policy_seg):
        day_k = int(day_seg_policy[k]) if k < day_seg_policy.size else -1
        all_on_k = int(on_all[k]) if k < on_all.size else 0
        pol_on_k = int(on_policy_aug[k]) if k < on_policy_aug.size else 0
        st_k = float(stamps_policy_aug[k]) if k < stamps_policy_aug.size else float(k) * float(cfg.dt_s)
        lines_txt.append(
            f"k={k}\tday={day_k}\tmoving={all_on_k}\tpolicy_on={pol_on_k}\tstamp_sim_s={st_k:.6g}"
        )
    debug_stamps_txt.write_text("\n".join(lines_txt) + "\n", encoding="utf-8")

    mx0, my0, mx1, my1 = _map_extent_webmerc_from_cfg(cfg, gwm.geometry.iloc[0])
    zx0, zy0, zx1, zy1 = _route_zoom_bounds_3857(gwm, r_m)

    z_full = _tile_zoom_for_bounds_3857(mx0, my0, mx1, my1, max_tiles=280)
    z_crop = _tile_zoom_for_bounds_3857(zx0, zy0, zx1, zy1, max_tiles=56)

    base_title = (
        f"Always-on (motion-integrated wedges / {cfg.dt_s:.0f}s step): "
        f"{cfg.hfov_deg:.0f}° × {cfg.scan_radius_m:.0f} m, EPSG:3857 — "
        f"{n_scan} steps  |  {rec.get('trajectory_source', '')}"
    )

    _save_coverage_mpl_png(
        out_full,
        mx0,
        my0,
        mx1,
        my1,
        cov_gdf,
        gwm,
        tile_zoom=z_full,
        fig_inches=15.0,
        dpi_save=220,
        title=base_title + " — full extent",
        rec=rec,
    )
    _save_coverage_mpl_png(
        out_zoom,
        zx0,
        zy0,
        zx1,
        zy1,
        cov_gdf,
        gwm,
        tile_zoom=z_crop,
        fig_inches=12.0,
        dpi_save=240,
        title=base_title + " — zoom to route",
        rec=rec,
    )

    html_path: Path | None = None
    try:
        from adaptive_scanning.interactive_map import save_realworld_folium_html

        age_pack = _simulate_sector_walk_projected(
            x_path=xg_policy,
            y_path=yg_policy,
            h_path=hg_policy,
            path_crs=str(crs),
            day_seg=day_seg_policy,
            on_mask=on_policy_aug,
            stamps=stamps_policy_aug,
            cfg=cfg,
            zoom_bounds_3857=(zx0, zy0, zx1, zy1),
        )[2]
        age_layer_name = (
            "Map age — sim time of last sector scan (policy camera; session seconds from start, red→green)"
        )
        if age_pack is None:
            age_pack = _simulate_sector_walk_projected(
                x_path=xg_walk,
                y_path=yg_walk,
                h_path=hg_walk,
                path_crs=str(crs),
                day_seg=day_seg_walk,
                on_mask=on_all,
                stamps=stamps_all,
                cfg=cfg,
                zoom_bounds_3857=(zx0, zy0, zx1, zy1),
            )[2]
            age_layer_name = (
                "Map age — sim time of last sector scan (always-on reference; "
                "policy raster unavailable)"
            )
        else:
            _a, _b, n_hit = age_pack
            if n_hit == 0:
                ap2 = _simulate_sector_walk_projected(
                    x_path=xg_walk,
                    y_path=yg_walk,
                    h_path=hg_walk,
                    path_crs=str(crs),
                    day_seg=day_seg_walk,
                    on_mask=on_all,
                    stamps=stamps_all,
                    cfg=cfg,
                    zoom_bounds_3857=(zx0, zy0, zx1, zy1),
                )[2]
                if ap2 is not None:
                    age_pack = ap2
                    age_layer_name = (
                        "Map age — sim time of last scan (always-on, **same as green coverage**; "
                        "policy had camera_off every step this episode)"
                    )
        if age_pack is None:
            age_rgba, age_bounds = None, None
        else:
            age_rgba, age_bounds, _n_hit_out = age_pack
        colored_paths = home_daily_colored_paths_3857(rec)
        extra_fg = folium_feature_groups_home_daily(rec)
        dec_fg = folium_feature_group_policy_decisions(rec)
        grid_fg = folium_feature_group_sim_env_grid(rec)
        age_cells_fg = folium_feature_group_env_last_seen_cells(rec)
        extra_for_html: list[Any] = []
        if extra_fg:
            extra_for_html.extend(extra_fg)
        if dec_fg is not None:
            extra_for_html.append(dec_fg)
        if grid_fg is not None:
            extra_for_html.append(grid_fg)
        if age_cells_fg is not None:
            extra_for_html.append(age_cells_fg)
        per_day_cov: list[tuple[str, str, Any]] = []
        for d in sorted(cov_by_day_graph.keys()):
            geom = cov_by_day_graph[d]
            if geom.is_empty:
                continue
            geom3857 = gpd.GeoDataFrame(geometry=[geom], crs=str(crs)).to_crs(3857).geometry.iloc[0]
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            per_day_cov.append((f"Coverage — always-on Day {d + 1}", col, geom3857))
        _policy_cov_graph, policy_by_day_graph, _policy_age_dup = _simulate_sector_walk_projected(
            x_path=xg_policy,
            y_path=yg_policy,
            h_path=hg_policy,
            path_crs=str(crs),
            day_seg=day_seg_policy,
            on_mask=on_policy_aug,
            stamps=stamps_policy_aug,
            cfg=cfg,
        )
        policy_cov_layers: list[tuple[str, str, Any]] = []
        for d in sorted(policy_by_day_graph.keys()):
            geom = policy_by_day_graph[d]
            if geom.is_empty:
                continue
            geom3857 = gpd.GeoDataFrame(geometry=[geom], crs=str(crs)).to_crs(3857).geometry.iloc[0]
            col = _HOME_DAILY_DAY_COLORS[d % len(_HOME_DAILY_DAY_COLORS)]
            policy_cov_layers.append((f"Coverage — policy camera Day {d + 1}", col, geom3857))
        html_done = save_realworld_folium_html(
            out_path=out_html,
            coverage_3857=cov_geom,
            path_line_3857=path_geom,
            age_rgba=age_rgba,
            age_bounds_wgs84=age_bounds,
            title=(
                "Open layers: path, always-on wedge coverage (green + per-day when multi-day), "
                "policy wedge coverage per day (camera on), **policy decision dots** (each env step), "
                "**sim grid** + **env last_seen cells** (toggle in layer control), map age heatmap. "
                "Scroll to zoom, drag to pan — "
                + base_title
            ),
            colored_path_layers_3857=colored_paths,
            extra_feature_groups=extra_for_html if extra_for_html else None,
            age_layer_name=age_layer_name,
            per_day_coverage_layers_3857=per_day_cov,
            policy_coverage_layers_3857=policy_cov_layers,
        )
        if html_done is not None:
            html_path = html_done
    except Exception:
        html_path = None

    return (out_full, out_zoom, html_path)


def _playback_morning_home_end_abs_s_by_day(rec: dict[str, Any]) -> dict[int, float]:
    """Absolute ``sim_time_s`` when ``morning_home`` ends, keyed by ``day_index`` (from playback)."""
    out: dict[int, float] = {}
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return out
    evs = pb.get("events")
    if not isinstance(evs, list):
        return out
    for ev in evs:
        if str(ev.get("phase")) != "morning_home":
            continue
        d = int(ev.get("day_index", -1))
        if d < 0:
            continue
        t1 = float(ev.get("t_end_s", 0.0))
        out[d] = max(out.get(d, -1.0), t1)
    return out


def try_save_realworld_day_prefix_coverage(
    rec: dict[str, Any],
    base_image_path: str | Path,
    *,
    prefix_seconds: float = 180.0,
    affine_margin_m: float = 1.0,
) -> tuple[Path, Path, Path | None] | None:
    """
    Union of **always-on** sector wedges along **actual** ``xs, ys`` / headings.

    With home-daily ``playback``, after each day's ``morning_home`` ends we include
    samples until **walking arc length** reaches ``prefix_seconds * walk_speed_m_s``
    metres along the trajectory (so "3 minutes" ≈ three minutes of **distance at
    average walk speed**, not three minutes of sim clock that can sit idle or crawl).

    Without playback, uses the first ``prefix_seconds`` of each ``day_duration_s``
    window (sim clock).
    """
    poly = rec.get("polyline_graph_m")
    crs = rec.get("graph_crs")
    xs = rec.get("xs")
    ys = rec.get("ys")
    if poly is None or crs is None or xs is None or ys is None or poly.shape[0] < 2:
        return None
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < 2 or xs.size != ys.size:
        return None
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point
        from shapely.ops import unary_union
    except ImportError:
        return None

    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    cfg: AdaptiveScanningConfig = rec["cfg"]
    base_image_path = Path(base_image_path)
    stem = base_image_path.stem
    sfx = base_image_path.suffix if base_image_path.suffix else ".png"
    mid = f"_day_first{int(round(float(prefix_seconds)))}s_coverage_realworld"
    out_full = base_image_path.with_name(f"{stem}{mid}{sfx}")
    out_zoom = base_image_path.with_name(f"{stem}{mid}_zoom{sfx}")
    out_html = base_image_path.with_name(f"{stem}{mid}_map.html")

    world_w_m = float(cfg.nx) * float(cfg.resolution_m)
    world_h_m = float(cfg.ny) * float(cfg.resolution_m)
    xy_w = np.column_stack([xs, ys])
    xy_g = inverse_affine_world_to_graph(
        xy_w,
        np.asarray(poly, dtype=np.float64),
        world_w_m=world_w_m,
        world_h_m=world_h_m,
        margin=float(affine_margin_m),
    )
    line_g = LineString(xy_g)
    gdf_line = gpd.GeoDataFrame(geometry=[line_g], crs=crs)
    try:
        utm_crs = gdf_line.estimate_utm_crs()
    except Exception:
        utm_crs = "EPSG:32619"
    g_utm = gdf_line.to_crs(utm_crs)
    coords = np.asarray(g_utm.geometry.iloc[0].coords, dtype=np.float64)
    n = int(coords.shape[0])
    if n < 2:
        return None

    dt_s = float(cfg.dt_s)
    day_d = float(cfg.day_duration_s)
    pre = float(prefix_seconds)
    if pre <= 0.0 or day_d <= 0.0:
        return None

    hs = rec.get("traj_heading_rad")
    if hs is not None:
        hs = np.asarray(hs, dtype=np.float64).ravel()
        if hs.size != n:
            hs = None
    h_utm = np.zeros(n, dtype=np.float64)
    if hs is not None:
        h_utm[:] = hs[:]
    else:
        for k in range(n - 1):
            dx = float(coords[k + 1, 0] - coords[k, 0])
            dy = float(coords[k + 1, 1] - coords[k, 1])
            h_utm[k] = math.atan2(dy, dx)
        if n >= 2:
            h_utm[-1] = h_utm[-2]

    r_m = float(cfg.scan_radius_m)
    leave_end = _playback_morning_home_end_abs_s_by_day(rec)
    use_after_home = bool(leave_end)
    walk_speed = float(cfg.walk_speed_m_s)
    arc_budget_m = float(pre) * max(walk_speed, 0.05)

    seg_len = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    cum_arc = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_len)])

    idx_w: list[int] = []
    for k in range(n - 1):
        t_k = float(k) * dt_s
        if use_after_home:
            d = int(t_k // day_d)
            t_leave = float(leave_end.get(d, float(d) * day_d))
            k_leave = int(math.ceil(t_leave / dt_s - 1e-9))
            if k < k_leave:
                continue
            walked_m = float(cum_arc[k] - cum_arc[k_leave])
            if walked_m > arc_budget_m + 1e-3:
                continue
        else:
            in_day = t_k - math.floor(t_k / day_d) * day_d
            if in_day >= pre - 1e-9:
                continue
        idx_w.append(k)
    if not idx_w:
        return None
    max_wedges = 9000
    stride = max(1, int(math.ceil(len(idx_w) / max_wedges)))
    wedges: list = []
    for j in range(0, len(idx_w), stride):
        k = idx_w[j]
        k1 = min(k + 1, n - 1)
        wedges.extend(
            _wedge_polygons_motion_segment_utm(
                float(coords[k, 0]),
                float(coords[k, 1]),
                float(h_utm[k]),
                float(coords[k1, 0]),
                float(coords[k1, 1]),
                float(h_utm[k1]),
                radius_m=r_m,
                hfov_deg=float(cfg.hfov_deg),
                resolution_m=float(cfg.resolution_m),
            )
        )
    coverage_utm = unary_union(wedges)
    cov_gdf = gpd.GeoDataFrame(geometry=[coverage_utm], crs=utm_crs).to_crs(3857)
    cov_geom = cov_gdf.geometry.iloc[0]

    if len(idx_w) >= 2:
        sub_coords = coords[np.array(idx_w, dtype=np.int64), :]
        line_sub_g = LineString(sub_coords)
        path_geom = (
            gpd.GeoDataFrame(geometry=[line_sub_g], crs=utm_crs).to_crs(3857).geometry.iloc[0]
        )
    else:
        path_geom = (
            gpd.GeoDataFrame(geometry=[Point(coords[idx_w[0], 0], coords[idx_w[0], 1])], crs=utm_crs)
            .to_crs(3857)
            .geometry.iloc[0]
        )

    gwm = gdf_line.to_crs(3857)
    mx0, my0, mx1, my1 = _map_extent_webmerc_from_cfg(cfg, gwm.geometry.iloc[0])
    zx0, zy0, zx1, zy1 = _route_zoom_bounds_3857(gwm, r_m)
    z_full = _tile_zoom_for_bounds_3857(mx0, my0, mx1, my1, max_tiles=280)
    z_crop = _tile_zoom_for_bounds_3857(zx0, zy0, zx1, zy1, max_tiles=56)

    n_w = len(idx_w)
    stride_note = f", wedge union stride={stride}" if stride > 1 else ""
    win = (
        f"≤{arc_budget_m:.0f} m walk ({pre:.0f}s×{walk_speed:.2f} m/s) after leaving home "
        f"(each {day_d/3600:.0f}h day)"
        if use_after_home
        else f"first {pre:.0f}s of sim clock (each {day_d/3600:.0f}h day)"
    )
    base_title = (
        f"Always-on ({win}): {cfg.hfov_deg:.0f}° × {cfg.scan_radius_m:.0f} m — "
        f"{n_w} samples / {len(wedges)} wedges @ dt={cfg.dt_s:.0f}s{stride_note}  |  "
        f"{rec.get('trajectory_source', '')}"
    )

    _save_coverage_mpl_png(
        out_full,
        mx0,
        my0,
        mx1,
        my1,
        cov_gdf,
        gwm,
        tile_zoom=z_full,
        fig_inches=15.0,
        dpi_save=220,
        title=base_title + " — full extent",
        rec=rec,
    )
    _save_coverage_mpl_png(
        out_zoom,
        zx0,
        zy0,
        zx1,
        zy1,
        cov_gdf,
        gwm,
        tile_zoom=z_crop,
        fig_inches=12.0,
        dpi_save=240,
        title=base_title + " — zoom to route",
        rec=rec,
    )

    html_path: Path | None = None
    try:
        from adaptive_scanning.interactive_map import save_realworld_folium_html

        colored_paths = home_daily_colored_paths_3857(rec)
        extra_fg = folium_feature_groups_home_daily(rec)
        dec_fg2 = folium_feature_group_policy_decisions(rec)
        grid_fg2 = folium_feature_group_sim_env_grid(rec)
        age_cells_fg2 = folium_feature_group_env_last_seen_cells(rec)
        extra_day: list[Any] = []
        if extra_fg:
            extra_day.extend(extra_fg)
        if dec_fg2 is not None:
            extra_day.append(dec_fg2)
        if grid_fg2 is not None:
            extra_day.append(grid_fg2)
        if age_cells_fg2 is not None:
            extra_day.append(age_cells_fg2)
        html_done = save_realworld_folium_html(
            out_path=out_html,
            coverage_3857=cov_geom,
            path_line_3857=path_geom,
            age_rgba=None,
            age_bounds_wgs84=None,
            title="Open layers: path, coverage, policy decision dots. " + base_title,
            colored_path_layers_3857=colored_paths,
            extra_feature_groups=extra_day if extra_day else None,
        )
        if html_done is not None:
            html_path = html_done
    except Exception:
        html_path = None

    return (out_full, out_zoom, html_path)


def _always_on_coverage_geom_3857_from_poly(
    poly: np.ndarray,
    graph_crs: str,
    cfg: AdaptiveScanningConfig,
) -> Any:
    """Union of forward-sector scans along one graph polyline (one pass), returned in EPSG:3857."""
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from adaptive_scanning.street_trajectories import resample_polyline_at_speed

    line = LineString(np.asarray(poly, dtype=np.float64))
    gdf = gpd.GeoDataFrame(geometry=[line], crs=graph_crs)
    try:
        utm_crs = gdf.estimate_utm_crs()
    except Exception:
        utm_crs = "EPSG:32619"
    g_utm = gdf.to_crs(utm_crs)
    coords = np.asarray(g_utm.geometry.iloc[0].coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return gpd.GeoDataFrame(geometry=[line], crs=graph_crs).to_crs(3857).geometry.iloc[0]
    seg = np.diff(coords, axis=0)
    trip = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    step = max(float(cfg.walk_speed_m_s) * float(cfg.dt_s), 1e-4)
    n_out = max(1, min(4000, int(math.ceil(trip / step)) + 1))
    x, y, h = resample_polyline_at_speed(
        coords,
        speed_m_s=float(cfg.walk_speed_m_s),
        dt_s=float(cfg.dt_s),
        n_out=n_out,
        repeat_path=False,
    )
    r_m = float(cfg.scan_radius_m)
    n_scan = len(x) - 1
    max_w = 3500
    stride = max(1, int(math.ceil(n_scan / max_w)))
    wedges = [
        _sector_wedge_polygon(
            float(x[i]),
            float(y[i]),
            float(h[i]),
            r_m,
            float(cfg.hfov_deg),
        )
        for i in range(0, n_scan, stride)
    ]
    cov = unary_union(wedges)
    return gpd.GeoDataFrame(geometry=[cov], crs=utm_crs).to_crs(3857).geometry.iloc[0]


def _four_paths_coverage_union_3857(
    polylines: list[np.ndarray],
    graph_crs: str,
    cfg: AdaptiveScanningConfig,
) -> Any:
    from shapely.ops import unary_union

    parts = [_always_on_coverage_geom_3857_from_poly(p, graph_crs, cfg) for p in polylines]
    return unary_union(parts)


def save_four_overlapping_paths_basemap_png(
    polylines: list[np.ndarray],
    graph_crs: str,
    out_path: str | Path,
    *,
    coverage_3857: Any | None = None,
    cfg: AdaptiveScanningConfig | None = None,
    pad_frac: float = 0.2,
    fig_inches: float = 12.0,
    dpi_save: int = 200,
    title: str = "",
) -> Path:
    """Draw four route polylines (graph CRS) on a sharp OSM Web-Mercator basemap; optional coverage."""
    import contextily as ctx
    import geopandas as gpd
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from shapely.geometry import LineString

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    geoms = [LineString(np.asarray(p, dtype=np.float64)) for p in polylines]
    gdf = gpd.GeoDataFrame({"k": [1, 2, 3, 4]}, geometry=geoms, crs=graph_crs)
    g3857 = gdf.to_crs(3857)

    xmin, ymin, xmax, ymax = g3857.total_bounds
    if coverage_3857 is not None:
        bx = gpd.GeoDataFrame(geometry=[coverage_3857], crs=3857).total_bounds
        xmin, ymin, xmax, ymax = (
            min(xmin, bx[0]),
            min(ymin, bx[1]),
            max(xmax, bx[2]),
            max(ymax, bx[3]),
        )
    span = max(xmax - xmin, ymax - ymin, 200.0)
    pad = max(span * pad_frac, 130.0)
    mx0, my0, mx1, my1 = xmin - pad, ymin - pad, xmax + pad, ymax + pad
    z = _tile_zoom_for_bounds_3857(mx0, my0, mx1, my1, max_tiles=80)

    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi_save)
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    ax.set_aspect("equal", adjustable="box")

    crs_3857 = g3857.crs
    try:
        ctx.add_basemap(
            ax,
            crs=crs_3857,
            zoom=z,
            source=ctx.providers.OpenStreetMap.Mapnik,
            attribution_size=6,
        )
    except Exception:
        try:
            ctx.add_basemap(
                ax,
                crs=crs_3857,
                zoom=z,
                source=ctx.providers.CartoDB.Positron,
                attribution_size=6,
            )
        except Exception:
            ctx.add_basemap(ax, crs=crs_3857, zoom=z, attribution_size=6)

    if coverage_3857 is not None:
        gpd.GeoDataFrame(geometry=[coverage_3857], crs=3857).plot(
            ax=ax,
            color="#2ecc71",
            alpha=0.48,
            edgecolor="none",
            zorder=4,
        )

    for i, geom in enumerate(g3857.geometry):
        gpd.GeoDataFrame(geometry=[geom], crs=3857).plot(
            ax=ax,
            color=colors[i],
            linewidth=4.5,
            alpha=0.92,
            zorder=6 + i,
        )

    handles = [
        mlines.Line2D([], [], color=colors[i], linewidth=3.5, label=f"Route {i + 1}")
        for i in range(len(colors))
    ]
    if coverage_3857 is not None:
        cov_lbl = "Coverage (always-on)"
        if cfg is not None:
            cov_lbl = f"Coverage always-on ({cfg.hfov_deg:.0f}°×{cfg.scan_radius_m:.0f} m)"
        handles.append(
            Patch(
                facecolor="#2ecc71",
                alpha=0.48,
                edgecolor="none",
                label=cov_lbl,
            )
        )
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92)
    ax.set_axis_off()
    fig.suptitle(
        title
        or "Four OSM shortest paths (endpoints may occasionally repeat across paths)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=dpi_save, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_four_overlapping_paths_example(
    cfg: AdaptiveScanningConfig,
    *,
    seed: int,
    out_base: str | Path,
) -> tuple[Path, Path | None]:
    """
    Load OSM graph from ``cfg``, sample four OD shortest paths with probabilistic endpoint reuse, write
    ``{out_base}.png`` and optionally ``{out_base}.html``.
    """
    from adaptive_scanning.street_trajectories import (
        _graph_crs_string,
        load_or_download_osm_graph,
        sample_four_od_paths_with_endpoint_reuse_prob,
    )

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_base.with_suffix(".png")
    out_html = out_base.with_suffix(".html")

    place = (cfg.osm_place or "").strip() or None
    bbox = cfg.osm_bbox
    if not place and bbox is None:
        from adaptive_scanning.street_trajectories import DEFAULT_OSM_PLACE

        place = DEFAULT_OSM_PLACE

    G = load_or_download_osm_graph(
        cache_dir=Path(cfg.osm_cache_dir),
        place=place,
        bbox=bbox,
        network_type=cfg.osm_network_type,
    )
    rng = np.random.default_rng(seed)
    polys = sample_four_od_paths_with_endpoint_reuse_prob(G, rng)
    if polys is None:
        raise RuntimeError(
            "Could not sample four OD paths (try another --seed or a larger --bbox graph)."
        )

    crs_s = _graph_crs_string(G)
    title = (
        f"Four OSM shortest paths (probabilistic start/end reuse) — "
        f"seed={seed}  |  {place or 'bbox'}"
    )
    cov_3857: Any | None = None
    try:
        cov_3857 = _four_paths_coverage_union_3857(polys, crs_s, cfg)
    except Exception:
        cov_3857 = None

    save_four_overlapping_paths_basemap_png(
        polys,
        crs_s,
        out_png,
        coverage_3857=cov_3857,
        cfg=cfg,
        title=title,
    )

    html_path: Path | None = None
    try:
        from shapely.geometry import LineString

        from adaptive_scanning.interactive_map import save_four_paths_folium_html

        geoms_graph = [LineString(np.asarray(p, dtype=np.float64)) for p in polys]
        h = save_four_paths_folium_html(
            out_path=out_html,
            linestrings=geoms_graph,
            source_crs=crs_s,
            coverage_3857=cov_3857,
            title=title + " — open in browser to zoom",
        )
        if h is not None:
            html_path = h
    except Exception:
        html_path = None

    return out_png, html_path


def try_save_episode_basemap(
    rec: dict[str, Any],
    base_image_path: str | Path,
) -> Path | None:
    """
    OSM raster basemap (Web Mercator) + shortest-path polyline in projected metres.
    Requires geopandas, matplotlib, contextily, shapely.
    """
    poly = rec.get("polyline_graph_m")
    crs = rec.get("graph_crs")
    if poly is None or crs is None or poly.shape[0] < 2:
        return None
    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import LineString
    except ImportError:
        return None

    base_image_path = Path(base_image_path)
    out_bm = base_image_path.with_name(
        f"{base_image_path.stem}_basemap{base_image_path.suffix}"
    )
    out_bm.parent.mkdir(parents=True, exist_ok=True)

    line = LineString(poly)
    gdf = gpd.GeoDataFrame(geometry=[line], crs=crs)
    gwm = gdf.to_crs(3857)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    xmin, ymin, xmax, ymax = gwm.total_bounds
    span_m = max(xmax - xmin, ymax - ymin, 50.0)
    pad = max(span_m * 0.15, 80.0)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")

    # OSM Mapnik tiles support zoom ~0–19; clamp inferred zoom
    zoom = int(round(15 - math.log2(max(span_m / 600.0, 0.25))))
    zoom = max(11, min(18, zoom))
    crs_3857 = gwm.crs
    # Contextily needs a set axis extent before tiles download reliably
    try:
        ctx.add_basemap(
            ax,
            crs=crs_3857,
            zoom=zoom,
            source=ctx.providers.OpenStreetMap.Mapnik,
            attribution_size=6,
        )
    except Exception:
        try:
            ctx.add_basemap(
                ax,
                crs=crs_3857,
                zoom=zoom,
                source=ctx.providers.CartoDB.Positron,
                attribution_size=6,
            )
        except Exception:
            ctx.add_basemap(ax, crs=crs_3857, zoom=zoom, attribution_size=6)

    # Optional: same OSM walk network edges under the route (cached graph)
    try:
        import osmnx as ox

        from adaptive_scanning.street_trajectories import (
            DEFAULT_OSM_PLACE,
            load_or_download_osm_graph,
        )

        cfg = rec["cfg"]
        place = (cfg.osm_place or "").strip() or None
        bbox = cfg.osm_bbox
        if not place and bbox is None:
            place = DEFAULT_OSM_PLACE
        G = load_or_download_osm_graph(
            cache_dir=Path(cfg.osm_cache_dir),
            place=place,
            bbox=bbox,
            network_type=cfg.osm_network_type,
        )
        _nodes, edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        if len(edges) > 0:
            ewm = edges.to_crs(3857)
            ewm.plot(ax=ax, color="white", linewidth=0.6, alpha=0.35, zorder=3)
    except Exception:
        pass

    hm = rec.get("osm_home_daily_meta")
    day_polys = hm.get("day_polylines_graph_m") if isinstance(hm, dict) else None
    ts = rec.get("trajectory_source", "?")
    if hm and crs and day_polys:
        plot_home_daily_paths_mercator_ax(ax, day_polys, str(crs), linewidth=5.0)
        plot_home_daily_markers_mercator_ax(ax, hm, str(crs), legend=True)
        fig.suptitle(
            f"Home-commute (one home ★; path & stops colored by day)  |  {ts}",
            fontsize=11,
        )
    else:
        gwm.plot(ax=ax, color="red", linewidth=5, alpha=0.95, zorder=6)
        fig.suptitle(
            f"Sample walk A→B (shortest path) on OpenStreetMap — Cambridge, MA area  |  {ts}",
            fontsize=11,
        )

    ax.set_axis_off()
    fig.savefig(out_bm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_bm


def try_save_playback_json(
    rec: dict[str, Any],
    episode_out_path: str | Path,
) -> Path | None:
    """
    When ``record_episode`` captured home-daily ``playback`` (phase timestamps in sim seconds),
    write a JSON file next to the episode image for external replay tooling.
    """
    pb = rec.get("playback")
    if not isinstance(pb, dict):
        return None
    events = pb.get("events")
    if not isinstance(events, list):
        return None
    episode_out_path = Path(episode_out_path)
    out_pb = episode_out_path.with_name(f"{episode_out_path.stem}_playback.json")
    out_pb.parent.mkdir(parents=True, exist_ok=True)
    with out_pb.open("w", encoding="utf-8") as f:
        json.dump(pb, f, indent=2)
    return out_pb


def policy_from_name(name: str, *, seed: int = 0) -> Policy:
    from adaptive_scanning.policies import (
        AlwaysOffPolicy,
        AlwaysOnPolicy,
        BudgetAwareGreedyPolicy,
        BudgetAwareGreedyUnseenOnlyPolicy,
        GreedyLocalStalenessPolicy,
        RandomPolicy,
    )

    name = name.lower().strip()
    if name == "random":
        return RandomPolicy(np.random.default_rng(seed))
    if name == "always_on":
        return AlwaysOnPolicy()
    if name == "always_off":
        return AlwaysOffPolicy()
    if name == "greedy_stale":
        return GreedyLocalStalenessPolicy()
    if name == "greedy_budget":
        return BudgetAwareGreedyPolicy()
    if name == "greedy_unseen":
        return BudgetAwareGreedyUnseenOnlyPolicy()
    raise ValueError(
        f"unknown policy name: {name!r} (try random, always_on, always_off, greedy_stale, greedy_budget, greedy_unseen)"
    )


def visualize_episode(
    cfg: AdaptiveScanningConfig,
    *,
    policy: Policy | None = None,
    policy_name: str = "random",
    seed: int = 0,
    out_path: str | Path = "outputs/adaptive_scanning/episode_preview.png",
    skip_episode_png: bool = False,
    coverage_first_minutes_per_day: float = 0.0,
) -> tuple[
    Path | None,
    str,
    Path | None,
    tuple[Path, Path, Path | None] | None,
    Path | None,
    tuple[Path, Path, Path | None] | None,
]:
    env = CameraBudgetEnv(cfg, seed=seed)
    if policy is not None:
        pol = policy
        label = policy_name.strip() or "custom"
    else:
        pol = policy_from_name(policy_name, seed=seed)
        label = policy_name
    rec = record_episode(env, pol, seed=seed)
    out_path = Path(out_path)
    src = str(rec.get("trajectory_source", "?"))
    panel_path: Path | None = None
    if not skip_episode_png:
        save_episode_figure(rec, out_path, title=f"policy={label} seed={seed}")
        panel_path = out_path
    basemap_path = try_save_episode_basemap(rec, out_path)
    coverage_pack = try_save_realworld_always_on_coverage(rec, out_path)
    playback_json = try_save_playback_json(rec, out_path)
    day_prefix_pack: tuple[Path, Path, Path | None] | None = None
    if float(coverage_first_minutes_per_day) > 0.0:
        day_prefix_pack = try_save_realworld_day_prefix_coverage(
            rec,
            out_path,
            prefix_seconds=float(coverage_first_minutes_per_day) * 60.0,
        )
    return panel_path, src, basemap_path, coverage_pack, playback_json, day_prefix_pack
