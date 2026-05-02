"""Optional Folium (Leaflet) export for real-world coverage + path + map-age overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import folium.plugins as _folium_plugins

    _HAS_FULLSCREEN = hasattr(_folium_plugins, "Fullscreen")
except Exception:
    _HAS_FULLSCREEN = False


def save_realworld_folium_html(
    *,
    out_path: Path,
    coverage_3857: Any,
    path_line_3857: Any,
    age_rgba: np.ndarray | None,
    age_bounds_wgs84: tuple[float, float, float, float] | None,
    title: str = "Coverage explorer",
) -> Path | None:
    """
    Write a self-contained HTML map with layer control (path, coverage, optional age).

    ``coverage_3857`` / ``path_line_3857``: shapely geometry in EPSG:3857 metres.
    ``age_rgba``: uint8 (H, W, 4) with alpha 0 where no data; ``age_bounds_wgs84`` is
    (west, south, east, north) in degrees for the overlay corners.
    """
    try:
        import folium
        import geopandas as gpd
        from folium import Element
        from folium.raster_layers import ImageOverlay
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cov = gpd.GeoDataFrame(geometry=[coverage_3857], crs=3857).to_crs(4326)
    path = gpd.GeoDataFrame(geometry=[path_line_3857], crs=3857).to_crs(4326)
    w0, s0, e0, n0 = cov.total_bounds
    w1, s1, e1, n1 = path.total_bounds
    w, s = min(w0, w1), min(s0, s1)
    e, n = max(e0, e1), max(n0, n1)

    m = folium.Map(
        location=[0.5 * (s + n), 0.5 * (w + e)],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fg_cov = folium.FeatureGroup(name="Coverage (always-on union)", show=True)
    folium.GeoJson(
        cov.to_json(),
        style_function=lambda _f: {
            "fillColor": "#2ecc71",
            "color": "#1e8449",
            "weight": 1,
            "fillOpacity": 0.45,
        },
    ).add_to(fg_cov)
    fg_cov.add_to(m)

    fg_path = folium.FeatureGroup(name="Path A→B", show=True)
    folium.GeoJson(
        path.to_json(),
        style_function=lambda _f: {
            "color": "#c0392b",
            "weight": 4,
            "opacity": 0.95,
        },
    ).add_to(fg_path)
    fg_path.add_to(m)

    if age_rgba is not None and age_bounds_wgs84 is not None:
        aw, as_, ae, an = age_bounds_wgs84
        img = np.flipud(age_rgba)
        fg_age = folium.FeatureGroup(name="Map age (zoom area)", show=False)
        ImageOverlay(
            image=img,
            bounds=[[as_, aw], [an, ae]],
            opacity=0.55,
            interactive=True,
            cross_origin=False,
        ).add_to(fg_age)
        fg_age.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    if _HAS_FULLSCREEN:
        import folium.plugins

        folium.plugins.Fullscreen(position="topright").add_to(m)

    title_html = (
        f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;'
        f'padding:6px 10px;border:1px solid #333;font:13px sans-serif;max-width:480px;">'
        f"{title}</div>"
    )
    m.get_root().html.add_child(Element(title_html))

    m.fit_bounds([[s, w], [n, e]], padding=(24, 24))
    m.save(str(out_path))
    return out_path


def save_four_paths_folium_html(
    *,
    out_path: Path,
    linestrings: list[Any],
    source_crs: str,
    coverage_3857: Any | None = None,
    title: str = "Four overlapping OD paths",
) -> Path | None:
    """Interactive map with one toggleable layer per route (graph CRS → WGS84 for Folium)."""
    try:
        import folium
        import geopandas as gpd
        from folium import Element
    except ImportError:
        return None

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g_src = gpd.GeoDataFrame(
        {"route": list(range(1, len(linestrings) + 1))},
        geometry=list(linestrings),
        crs=source_crs,
    )
    g4326 = g_src.to_crs(4326)
    min_lon, min_lat, max_lon, max_lat = g4326.total_bounds
    cov4326 = None
    if coverage_3857 is not None:
        cov4326 = gpd.GeoDataFrame(geometry=[coverage_3857], crs=3857).to_crs(4326)
        bx = cov4326.total_bounds
        min_lon, min_lat, max_lon, max_lat = (
            min(min_lon, bx[0]),
            min(min_lat, bx[1]),
            max(max_lon, bx[2]),
            max(max_lat, bx[3]),
        )
    m = folium.Map(
        location=[float(0.5 * (min_lat + max_lat)), float(0.5 * (min_lon + max_lon))],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    if cov4326 is not None:
        fg_cov = folium.FeatureGroup(name="Coverage (always-on, all routes)", show=True)
        folium.GeoJson(
            cov4326.to_json(),
            style_function=lambda _f: {
                "fillColor": "#2ecc71",
                "color": "#1e8449",
                "weight": 1,
                "fillOpacity": 0.45,
            },
        ).add_to(fg_cov)
        fg_cov.add_to(m)

    for i, geom in enumerate(g4326.geometry):
        gj = gpd.GeoDataFrame(geometry=[geom], crs=4326).to_json()
        c = colors[i % len(colors)]
        fg = folium.FeatureGroup(name=f"Route {i + 1}", show=True)
        folium.GeoJson(
            gj,
            style_function=lambda _f, col=c: {"color": col, "weight": 5, "opacity": 0.9},
        ).add_to(fg)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    if _HAS_FULLSCREEN:
        import folium.plugins

        folium.plugins.Fullscreen(position="topright").add_to(m)

    title_html = (
        f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;'
        f'padding:6px 10px;border:1px solid #333;font:13px sans-serif;max-width:520px;">'
        f"{title}</div>"
    )
    m.get_root().html.add_child(Element(title_html))
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(22, 22))
    m.save(str(out_path))
    return out_path
