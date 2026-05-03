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
    traj_src = str(info0.get("trajectory_source", "?"))
    act_req: list[int] = []
    act_eff: list[float] = []
    sim_t: list[float] = []
    budget: list[float] = []

    while True:
        a = policy.act(obs, {})
        st = env.step(a)
        inf = st.info
        act_req.append(int(inf["action_requested"]))
        act_eff.append(1.0 if inf["camera_on_effective"] else 0.0)
        sim_t.append(float(inf["sim_time_s"]))
        budget.append(float(inf["budget_s"]))
        obs = st.observation
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
    im = ax1.imshow(
        age,
        origin="lower",
        extent=[0, cfg.nx * cfg.resolution_m, 0, cfg.ny * cfg.resolution_m],
        aspect="auto",
        cmap="magma_r",
        interpolation="nearest",
    )
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


def _policy_camera_coverage_union_3857(
    rec: dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    poly: np.ndarray,
    graph_crs: str,
    utm_crs: Any,
    cfg: AdaptiveScanningConfig,
) -> Any | None:
    """
    Union of motion-integrated sector wedges in **UTM**, same recipe as always-on coverage,
    but only for segments where ``camera_on_effective`` is true. Returned in EPSG:3857.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    from adaptive_scanning.street_trajectories import inverse_affine_world_to_graph

    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < 2 or xs.size != ys.size:
        return None
    nseg = int(xs.size) - 1
    on = np.asarray(rec["camera_on_effective"], dtype=np.float64).ravel()[:nseg] > 0.5
    if not np.any(on):
        return None

    world_w_m = float(cfg.nx) * float(cfg.resolution_m)
    world_h_m = float(cfg.ny) * float(cfg.resolution_m)
    poly_a = np.asarray(poly, dtype=np.float64)
    xy_w = np.column_stack([xs, ys])
    xy_g = inverse_affine_world_to_graph(
        xy_w,
        poly_a,
        world_w_m=world_w_m,
        world_h_m=world_h_m,
        margin=1.0,
    )
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xy_g[:, 0], xy_g[:, 1]),
        crs=str(graph_crs),
    ).to_crs(utm_crs)
    xu = np.asarray(gdf.geometry.x, dtype=np.float64)
    yu = np.asarray(gdf.geometry.y, dtype=np.float64)

    hs = rec.get("traj_heading_rad")
    npt = int(xs.size)
    hu = np.zeros(npt, dtype=np.float64)
    if hs is not None and int(np.asarray(hs).size) == npt:
        hu[:] = np.asarray(hs, dtype=np.float64).ravel()[:npt]
    elif npt >= 2:
        for ii in range(npt - 1):
            hu[ii] = math.atan2(float(yu[ii + 1] - yu[ii]), float(xu[ii + 1] - xu[ii]))
        hu[-1] = hu[-2]

    r_m = float(cfg.scan_radius_m)
    wedges: list[Any] = []
    for k in range(nseg):
        if not bool(on[k]):
            continue
        wedges.extend(
            _wedge_polygons_motion_segment_utm(
                float(xu[k]),
                float(yu[k]),
                float(hu[k]),
                float(xu[k + 1]),
                float(yu[k + 1]),
                float(hu[k + 1]),
                radius_m=r_m,
                hfov_deg=float(cfg.hfov_deg),
                resolution_m=float(cfg.resolution_m),
            )
        )
    if not wedges:
        return None
    uu = unary_union(wedges)
    if uu.is_empty:
        return None
    return gpd.GeoDataFrame(geometry=[uu], crs=utm_crs).to_crs(3857).geometry.iloc[0]


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


def _sim_time_grid_to_rgba_turbo(
    time_s: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    alpha: int = 238,
) -> np.ndarray:
    """``time_s`` NaN = transparent. Finite values mapped linearly ``vmin``…``vmax`` with ``turbo``."""
    from matplotlib import colormaps

    cmap = colormaps["turbo"]
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

    if always_on_reference:
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
    rgba = _sim_time_grid_to_rgba_turbo(last_plot, vmin=0.0, vmax=ft)

    if stamp_debug_path is not None:
        lines = [
            "NOTE: Green Folium coverage = always-on sector wedges along the resampled route.",
            "It is NOT gated by policy camera_on. Policy map-age uses camera_on_effective unless",
            "the HTML builder falls back to always-on when the policy never stamps (n_hit=0).",
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
                f"last_plot (sim_s used for turbo color, 0=discretization gap): "
                f"min={float(np.nanmin(lv)):.6g} max={float(np.nanmax(lv)):.6g} "
                f"mean={float(np.nanmean(lv)):.6g} p5={float(np.percentile(lv, 5)):.6g} "
                f"p95={float(np.percentile(lv, 95)):.6g}"
            )
        lines.append("")
        lines.append(f"turbo_vmin=0 turbo_vmax={ft}")
        stamp_debug_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_debug_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    lons, lats = t4326.transform(
        [xmin, xmax, xmax, xmin],
        [ymin, ymin, ymax, ymax],
    )
    w, e = min(lons), max(lons)
    s, nlat = min(lats), max(lats)
    n_hit = int(np.sum(hit))
    return rgba, (w, s, e, nlat), n_hit


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


def try_save_realworld_always_on_coverage(
    rec: dict[str, Any],
    base_image_path: str | Path,
) -> tuple[Path, Path, Path | None] | None:
    """
    Writes:
      1) ``*_coverage_realworld.png`` — full place/bbox extent, sharp OSM tiles
      2) ``*_coverage_realworld_zoom.png`` — cropped around the route
      3) ``*_coverage_realworld_map.html`` — Folium (if ``folium`` installed): path,
         **always-on** coverage (green), **policy** coverage when camera on (blue), optional
         **map-age raster** (``turbo``).

    **Coverage vs map age:** the green union is **always-on** wedges along the resampled route
    (``repeat_path=False``); it does **not** read ``camera_on_effective``. The heatmap uses the
    policy’s camera first; if the policy never turns the camera on (``n_hit==0``), the HTML
    layer falls back to the same **always-on** stamping as the coverage footprint.

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
    n_scan = len(x) - 1
    max_wedges = 9000
    stride = max(1, int(math.ceil(n_scan / max_wedges)))

    wedges: list = []
    for i in range(0, n_scan, stride):
        wedges.extend(
            _wedge_polygons_motion_segment_utm(
                float(x[i]),
                float(y[i]),
                float(h[i]),
                float(x[i + 1]),
                float(y[i + 1]),
                float(h[i + 1]),
                radius_m=r_m,
                hfov_deg=float(cfg.hfov_deg),
                resolution_m=float(cfg.resolution_m),
            )
        )
    coverage_utm = unary_union(wedges)
    cov_gdf = gpd.GeoDataFrame(geometry=[coverage_utm], crs=utm_crs).to_crs(3857)
    cov_geom = cov_gdf.geometry.iloc[0]
    path_geom = gwm.geometry.iloc[0]

    mx0, my0, mx1, my1 = _map_extent_webmerc_from_cfg(cfg, gwm.geometry.iloc[0])
    zx0, zy0, zx1, zy1 = _route_zoom_bounds_3857(gwm, r_m)

    z_full = _tile_zoom_for_bounds_3857(mx0, my0, mx1, my1, max_tiles=280)
    z_crop = _tile_zoom_for_bounds_3857(zx0, zy0, zx1, zy1, max_tiles=56)

    stride_note = f", segment stride={stride}" if stride > 1 else ""
    base_title = (
        f"Always-on (motion-integrated wedges / {cfg.dt_s:.0f}s step): "
        f"{cfg.hfov_deg:.0f}° × {cfg.scan_radius_m:.0f} m, EPSG:3857 — "
        f"{n_scan} steps{stride_note}  |  {rec.get('trajectory_source', '')}"
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

        stamp_txt = out_html.parent / f"{out_html.stem}_stamps.txt"

        world_w_m = float(cfg.nx) * float(cfg.resolution_m)
        world_h_m = float(cfg.ny) * float(cfg.resolution_m)
        xs_a = np.asarray(xs, dtype=np.float64)
        ys_a = np.asarray(rec["ys"], dtype=np.float64)
        n_path = int(xs_a.shape[0])
        hs = rec.get("traj_heading_rad")
        if hs is not None and int(np.asarray(hs).size) == n_path:
            h_pol = np.asarray(hs, dtype=np.float64).ravel()[:n_path]
        elif n_path >= 2:
            h_pol = np.empty(n_path, dtype=np.float64)
            for ii in range(n_path - 1):
                h_pol[ii] = math.atan2(
                    float(ys_a[ii + 1] - ys_a[ii]),
                    float(xs_a[ii + 1] - xs_a[ii]),
                )
            h_pol[-1] = h_pol[-2]
        else:
            h_pol = np.zeros(max(1, n_path), dtype=np.float64)

        def _reference_sector_age_pack():
            ref_g = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(
                    np.asarray(x, dtype=np.float64),
                    np.asarray(y, dtype=np.float64),
                ),
                crs=utm_crs,
            ).to_crs(str(crs))
            xg = np.asarray(ref_g.geometry.x, dtype=np.float64)
            yg = np.asarray(ref_g.geometry.y, dtype=np.float64)
            xyw_ref = _graph_xy_to_world_xy(
                np.asarray(poly, dtype=np.float64),
                np.column_stack([xg, yg]),
                world_w_m=world_w_m,
                world_h_m=world_h_m,
                margin=1.0,
            )
            xw_ref = xyw_ref[:, 0]
            yw_ref = xyw_ref[:, 1]
            n_ref = int(xw_ref.shape[0])
            if n_ref >= 2:
                h_ref = np.empty(n_ref, dtype=np.float64)
                for ii in range(n_ref - 1):
                    h_ref[ii] = math.atan2(
                        float(yg[ii + 1] - yg[ii]),
                        float(xg[ii + 1] - xg[ii]),
                    )
                h_ref[-1] = h_ref[-2]
            else:
                h_ref = np.zeros(max(1, n_ref), dtype=np.float64)
            return _sector_scan_age_rgba_wgs84(
                rec,
                cov_geom,
                (zx0, zy0, zx1, zy1),
                always_on_reference=True,
                xw_path=xw_ref,
                yw_path=yw_ref,
                h_path=h_ref,
                poly=np.asarray(poly, dtype=np.float64),
                graph_crs=str(crs),
                world_w_m=world_w_m,
                world_h_m=world_h_m,
                stamp_debug_path=stamp_txt,
            )

        age_pack = _sector_scan_age_rgba_wgs84(
            rec,
            cov_geom,
            (zx0, zy0, zx1, zy1),
            always_on_reference=False,
            xw_path=xs_a,
            yw_path=ys_a,
            h_path=h_pol,
            poly=np.asarray(poly, dtype=np.float64),
            graph_crs=str(crs),
            world_w_m=world_w_m,
            world_h_m=world_h_m,
            stamp_debug_path=stamp_txt,
        )
        age_layer_name = (
            "Map age — sim time of last sector scan (policy camera; s from episode start, turbo)"
        )
        if age_pack is None:
            age_pack = _reference_sector_age_pack()
            age_layer_name = (
                "Map age — sim time of last sector scan (always-on reference; "
                "policy raster unavailable)"
            )
        else:
            _a, _b, n_hit = age_pack
            if n_hit == 0:
                ap2 = _reference_sector_age_pack()
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
        policy_cov_3857 = _policy_camera_coverage_union_3857(
            rec,
            xs_a,
            ys_a,
            np.asarray(poly, dtype=np.float64),
            str(crs),
            utm_crs,
            cfg,
        )
        html_done = save_realworld_folium_html(
            out_path=out_html,
            coverage_3857=cov_geom,
            path_line_3857=path_geom,
            age_rgba=age_rgba,
            age_bounds_wgs84=age_bounds,
            title=(
                "Open layers: path, coverage (always-on vs policy), map age (zoom box). "
                "Scroll to zoom, drag to pan — " + base_title
            ),
            colored_path_layers_3857=colored_paths,
            extra_feature_groups=extra_fg,
            age_layer_name=age_layer_name,
            policy_coverage_3857=policy_cov_3857,
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
        html_done = save_realworld_folium_html(
            out_path=out_html,
            coverage_3857=cov_geom,
            path_line_3857=path_geom,
            age_rgba=None,
            age_bounds_wgs84=None,
            title="Open layers: path, coverage. " + base_title,
            colored_path_layers_3857=colored_paths,
            extra_feature_groups=extra_fg,
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
    raise ValueError(f"unknown policy name: {name!r} (try random, always_on, always_off, greedy_stale, greedy_budget)")


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
