from __future__ import annotations

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


def _norm_age_rgba_utm(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    n_scan: int,
    r_m: float,
    hfov_deg: float,
    utm_crs: Any,
    zoom_bounds_3857: tuple[float, float, float, float],
    *,
    nx: int = 112,
    ny: int = 112,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Raster of scan-age (last step index − first hit) in UTM; RGBA + WGS84 overlay bounds."""
    import geopandas as gpd
    from matplotlib import colormaps
    from pyproj import Transformer
    from shapely.geometry import box

    zx0, zy0, zx1, zy1 = zoom_bounds_3857
    utmb = gpd.GeoDataFrame(geometry=[box(zx0, zy0, zx1, zy1)], crs=3857).to_crs(utm_crs)
    ux0, uy0, ux1, uy1 = utmb.total_bounds

    gx = np.linspace(ux0, ux1, nx, dtype=np.float64)
    gy = np.linspace(uy0, uy1, ny, dtype=np.float64)
    wx, wy = np.meshgrid(gx, gy)
    first = np.full((ny, nx), np.inf, dtype=np.float64)
    half = math.radians(0.5 * float(hfov_deg))
    work = n_scan * nx * ny
    inner_stride = max(1, int(math.ceil(work / 12_000_000)))
    for i in range(0, n_scan, inner_stride):
        dx = wx - float(x[i])
        dy = wy - float(y[i])
        dist = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) - float(h[i])
        ang = _wrap_pi(ang)
        m = (dist <= r_m) & (dist >= 1e-3) & (np.abs(ang) <= half)
        first = np.where(m, np.minimum(first, float(i)), first)

    last_i = float(max(0, n_scan - 1))
    age = last_i - first
    age[~np.isfinite(first)] = np.nan
    mx_age = float(np.nanmax(age)) if np.any(np.isfinite(age)) else 1.0
    norm = np.clip(age / max(mx_age, 1.0), 0.0, 1.0)
    cmap = colormaps["magma_r"]
    rgba = (cmap(norm) * 255.0).astype(np.uint8)
    mnan = np.isnan(norm)
    rgba[mnan, :] = 0

    t4326 = Transformer.from_crs(utm_crs, 4326, always_xy=True)
    lons, lats = t4326.transform([ux0, ux1, ux1, ux0], [uy0, uy0, uy1, uy1])
    w, e = min(lons), max(lons)
    s, nlat = min(lats), max(lats)
    return rgba, (w, s, e, nlat)


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
    gwm.plot(ax=ax, color="red", linewidth=3.2, alpha=0.95, zorder=6)
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
      3) ``*_coverage_realworld_map.html`` — Folium (if ``folium`` installed): path, coverage,
         optional map-age raster over the zoom extent (toggle layers, scroll zoom).

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
    max_wedges = 4500
    stride = max(1, int(math.ceil(n_scan / max_wedges)))

    wedges: list = []
    for i in range(0, n_scan, stride):
        wedges.append(
            _sector_wedge_polygon(
                float(x[i]),
                float(y[i]),
                float(h[i]),
                r_m,
                float(cfg.hfov_deg),
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

    stride_note = f", wedge union stride={stride}" if stride > 1 else ""
    base_title = (
        f"Always-on: {cfg.hfov_deg:.0f}° × {cfg.scan_radius_m:.0f} m (UTM wedges), "
        f"EPSG:3857 map — {n_scan} steps @ dt={cfg.dt_s:.0f}s{stride_note}  |  {rec.get('trajectory_source', '')}"
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
    )

    html_path: Path | None = None
    try:
        from adaptive_scanning.interactive_map import save_realworld_folium_html

        age_rgba, age_bounds = _norm_age_rgba_utm(
            x,
            y,
            h,
            n_scan,
            r_m,
            float(cfg.hfov_deg),
            utm_crs,
            (zx0, zy0, zx1, zy1),
        )
        html_done = save_realworld_folium_html(
            out_path=out_html,
            coverage_3857=cov_geom,
            path_line_3857=path_geom,
            age_rgba=age_rgba,
            age_bounds_wgs84=age_bounds,
            title=(
                "Open layers: path, coverage, map age (zoom box). "
                "Scroll to zoom, drag to pan — " + base_title
            ),
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
        or "Four shortest paths: two starts in one cluster × two ends in another (overlap by design)",
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
    Load OSM graph from ``cfg``, sample four overlapping-cluster OD paths, write
    ``{out_base}.png`` and optionally ``{out_base}.html``.
    """
    from adaptive_scanning.street_trajectories import (
        _graph_crs_string,
        load_or_download_osm_graph,
        sample_four_overlapping_od_paths,
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
    polys = sample_four_overlapping_od_paths(G, rng)
    if polys is None:
        raise RuntimeError(
            "Could not sample four overlapping OD paths (try another --seed or a --bbox graph)."
        )

    crs_s = _graph_crs_string(G)
    title = (
        f"Four OSM shortest paths (shared start / end neighbourhoods) — "
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

    gwm.plot(ax=ax, color="red", linewidth=5, alpha=0.95, zorder=6)

    ax.set_axis_off()
    ts = rec.get("trajectory_source", "?")
    fig.suptitle(
        f"Sample walk A→B (shortest path) on OpenStreetMap — Cambridge, MA area  |  {ts}",
        fontsize=11,
    )
    fig.savefig(out_bm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_bm


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
    policy_name: str = "random",
    seed: int = 0,
    out_path: str | Path = "outputs/adaptive_scanning/episode_preview.png",
) -> tuple[Path, str, Path | None, tuple[Path, Path, Path | None] | None]:
    env = CameraBudgetEnv(cfg, seed=seed)
    pol = policy_from_name(policy_name, seed=seed)
    rec = record_episode(env, pol, seed=seed)
    out_path = Path(out_path)
    src = str(rec.get("trajectory_source", "?"))
    save_episode_figure(rec, out_path, title=f"policy={policy_name} seed={seed}")
    basemap_path = try_save_episode_basemap(rec, out_path)
    coverage_pack = try_save_realworld_always_on_coverage(rec, out_path)
    return out_path, src, basemap_path, coverage_pack
