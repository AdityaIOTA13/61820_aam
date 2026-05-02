"""
Street-following trajectories from OpenStreetMap (OSMnx) + NetworkX shortest paths.

OD pairs are biased toward a small **anchor** node set so routes naturally **overlap**.
Coordinates are affinely mapped into the env world rectangle [0, world_w] x [0, world_h].
"""

from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

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


def sample_four_overlapping_od_paths(
    G: Any,
    rng: np.random.Generator,
    *,
    start_cluster_radius_m: float = 140.0,
    end_cluster_radius_m: float = 140.0,
    min_start_end_separation_m: float = 650.0,
    max_attempts: int = 500,
) -> list[np.ndarray] | None:
    """
    Four shortest-path walks with **shared start neighbourhood** and **shared end neighbourhood**.

    Picks two start nodes within ``start_cluster_radius_m`` of a seed and two end nodes within
    ``end_cluster_radius_m`` of a distant hub, then routes
    ``(s1,e1), (s1,e2), (s2,e1), (s2,e2)`` so initial and final street segments naturally overlap.
    Each polyline is ``(N, 2)`` in the graph projected CRS.
    """
    nodes = _largest_cc_nodes(G)
    if len(nodes) < 12:
        return None
    nd = list(nodes)
    for _ in range(max_attempts):
        rng.shuffle(nd)
        seed = int(rng.choice(nd))
        sx, sy = _node_xy(G, seed)
        pool_s = nodes_within_planar_radius_m(G, sx, sy, start_cluster_radius_m)
        if len(pool_s) < 2:
            continue
        pool_s = list(dict.fromkeys(pool_s))
        rng.shuffle(pool_s)

        pool_e: list[Any] | None = None
        ex = ey = 0.0
        for _ in range(120):
            far = int(rng.choice(nd))
            ex, ey = _node_xy(G, far)
            if math.hypot(ex - sx, ey - sy) < min_start_end_separation_m:
                continue
            cand = nodes_within_planar_radius_m(G, ex, ey, end_cluster_radius_m)
            if len(cand) < 2:
                continue
            pool_e = list(dict.fromkeys(cand))
            rng.shuffle(pool_e)
            break
        if pool_e is None or len(pool_e) < 2:
            continue

        s1, s2 = pool_s[0], pool_s[1]
        if s1 == s2 and len(pool_s) > 2:
            s2 = pool_s[2]
        if s1 == s2:
            continue
        e1, e2 = pool_e[0], pool_e[1]
        if e1 == e2 and len(pool_e) > 2:
            e2 = pool_e[2]
        if e1 == e2:
            continue

        polys: list[np.ndarray] = []
        for o, d in ((s1, e1), (s1, e2), (s2, e1), (s2, e2)):
            p = shortest_path_polyline(G, o, d)
            if p is None or p.shape[0] < 2:
                polys = []
                break
            polys.append(np.asarray(p, dtype=np.float64))
        if len(polys) == 4:
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
