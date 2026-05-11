#!/usr/bin/env python3
"""
Plot bar charts from ``coverage-eval`` JSON output (``run_benchmark`` / ``write_benchmark_json``).

Example::

    python scripts/plot_coverage_eval.py --input outputs/coverage_eval.json \\
        --out outputs/coverage_eval_plots.png

    # One file per chart (folder)
    python scripts/plot_coverage_eval.py -i outputs/coverage_eval.json --separate \\
        --out-dir outputs/coverage_eval_figures

Requires matplotlib (see requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Order for legend / x-axis (baselines → learned → oracle)
METHOD_ORDER = [
    "random",
    "random_iid",  # legacy JSON key
    "greedy_unseen",
    "greedy_unseen_progressive",
    "trained",
    "oracle",
]

DISPLAY_NAMES = {
    "trained": "Trained MLP",
    "random": "Random (full budget)",
    "random_iid": "Random (full budget)",
    "greedy_unseen": "Greedy unseen",
    "greedy_unseen_progressive": "Greedy unseen + late rescan",
    "oracle": "Oracle",
}

# Distinct, colorblind-friendly-ish sequence
COLORS = {
    "random": "#7eb0d5",
    "random_iid": "#7eb0d5",
    "greedy_unseen": "#bd7ebe",
    "greedy_unseen_progressive": "#8c9e5e",
    "trained": "#fd7f6f",
    "oracle": "#ffb55a",
}


def _load_summary(path: Path) -> tuple[dict[str, dict], dict]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise SystemExit("JSON missing 'summary' object")
    return summary, payload


def _ordered_methods(summary: dict[str, dict | None]) -> list[str]:
    """Prefer ``random``; skip legacy ``random_iid`` if ``random`` is present."""
    out: list[str] = []
    seen_random = False
    for key in METHOD_ORDER:
        if key == "random_iid" and seen_random:
            continue
        row = summary.get(key)
        if row is not None and isinstance(row, dict):
            out.append(key)
            if key in ("random", "random_iid"):
                seen_random = True
    return out


def _bar_values(row: dict, metric_base: str) -> tuple[float, float]:
    """Reads ``{metric_base}_mean`` / ``_std``; missing keys → NaN mean (legacy JSON)."""
    key_m = f"{metric_base}_mean"
    key_s = f"{metric_base}_std"
    if key_m not in row:
        return float("nan"), 0.0
    m = float(row[key_m])
    s = float(row[key_s]) if key_s in row else 0.0
    return m, s


def _summary_has_metric(summary: dict[str, dict | None], methods: list[str], metric_base: str) -> bool:
    for m in methods:
        r = summary.get(m)
        if r and f"{metric_base}_mean" in r:
            return True
    return False


def _meta_title(payload: dict) -> str:
    parts: list[str] = []
    if payload.get("n_scenarios") is not None:
        parts.append(f"N = {payload['n_scenarios']} scenarios")
    if payload.get("oracle_mode"):
        parts.append(f"oracle: {payload['oracle_mode']}")
    if payload.get("checkpoint"):
        parts.append(f"checkpoint: {Path(str(payload['checkpoint'])).name}")
    return " · ".join(parts)


def _dt_s(payload: dict) -> float:
    """Episode step duration for legacy JSON that only has ``camera_on_fraction``."""
    return float(payload.get("dt_s", 30.0))


def _panel_camera_sec(
    summary: dict[str, dict | None], methods: list[str], payload: dict
) -> tuple[list[float], list[float]]:
    """Mean effective camera time (s); falls back to fraction × ``dt_s`` for old JSON."""
    dt = _dt_s(payload)
    means: list[float] = []
    stds: list[float] = []
    for m in methods:
        row = summary[m]
        assert row is not None
        if "effective_camera_on_sec_mean" in row:
            mu, sig = _bar_values(row, "effective_camera_on_sec")
        elif "camera_on_fraction_mean" in row:
            mu, sig = _bar_values(row, "camera_on_fraction")
            mu *= dt
            sig *= dt
        else:
            mu, sig = float("nan"), 0.0
        means.append(mu)
        stds.append(sig)
    return means, stds


def _panel_coverage_means(
    summary: dict[str, dict | None], methods: list[str]
) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for m in methods:
        row = summary[m]
        assert row is not None
        mu, sig = _bar_values(row, "coverage_vs_theoretic_max")
        means.append(mu)
        stds.append(sig)
    return means, stds


def _panel_age_hours(
    summary: dict[str, dict | None], methods: list[str]
) -> tuple[list[float], list[float]]:
    means_h, stds_h = [], []
    for m in methods:
        row = summary[m]
        assert row is not None
        mu, sig = _bar_values(row, "mean_age_scanned_sec")
        means_h.append(mu / 3600.0)
        stds_h.append(sig / 3600.0)
    return means_h, stds_h


def _panel_metric(
    summary: dict[str, dict | None], methods: list[str], metric_base: str
) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for m in methods:
        row = summary[m]
        assert row is not None
        mu, sig = _bar_values(row, metric_base)
        means.append(mu)
        stds.append(sig)
    return means, stds


def plot_coverage_eval(
    summary: dict[str, dict | None],
    payload: dict,
    *,
    out_path: Path,
    dpi: int,
    fmt: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = _ordered_methods(summary)
    if not methods:
        raise SystemExit("No non-null methods in summary (nothing to plot).")

    labels = [DISPLAY_NAMES.get(m, m) for m in methods]
    x = np.arange(len(methods))
    colors = [COLORS.get(m, "#888888") for m in methods]

    meta = _meta_title(payload)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    if meta:
        fig.suptitle("Coverage evaluation — " + meta, fontsize=11, y=1.02)

    means, stds = _panel_coverage_means(summary, methods)
    ax = axes[0, 0]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Coverage (fraction of geometric union)")
    ax.set_ylim(0.0, min(1.15, max(0.15, max(means) + max(stds) + 0.08)))
    ax.set_title("Coverage vs. theoretic max")
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
    for i, (mu, sig) in enumerate(zip(means, stds)):
        ax.text(
            i,
            mu + sig + 0.02,
            f"{mu:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    means_h, stds_h = _panel_age_hours(summary, methods)
    ax = axes[0, 1]
    ax.bar(x, means_h, yerr=stds_h, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean age of scanned cells (hours)")
    ax.set_title("Mean scan age (end of episode − last_seen)")
    ymax = max(m + s for m, s in zip(means_h, stds_h)) if means_h else 1.0
    ax.set_ylim(0.0, ymax * 1.18)

    means_lcc, stds_lcc = _panel_metric(summary, methods, "lcc_vs_theoretic_max")
    ax = axes[1, 0]
    if _summary_has_metric(summary, methods, "lcc_vs_theoretic_max"):
        ax.bar(x, means_lcc, yerr=stds_lcc, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        finite_pairs = [(m, s) for m, s in zip(means_lcc, stds_lcc) if np.isfinite(m)]
        ymax_l = max(m + s for m, s in finite_pairs) if finite_pairs else 1.0
        ax.set_ylim(0.0, min(1.12, max(0.12, ymax_l + 0.07)))
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7, alpha=0.55)
    else:
        ax.text(
            0.5,
            0.5,
            "LCC not in JSON — re-run coverage-eval",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#666666",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("LCC / geometric union")
    ax.set_title("Largest connected scanned component")

    means_cam, stds_cam = _panel_camera_sec(summary, methods, payload)
    ax = axes[1, 1]
    ax.bar(x, means_cam, yerr=stds_cam, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Effective camera on (s)")
    ax.set_title("Camera on time (effective, not requested)")
    finite_c = [(m, s) for m, s in zip(means_cam, stds_cam) if np.isfinite(m)]
    ymax_c = max(m + s for m, s in finite_c) if finite_c else 1.0
    ax.set_ylim(0.0, ymax_c * 1.15)

    out_path = out_path.with_suffix(f".{fmt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_coverage_eval_fullpage(
    summary: dict[str, dict | None],
    payload: dict,
    *,
    out_path: Path,
    dpi: int,
    fmt: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = _ordered_methods(summary)
    if not methods:
        raise SystemExit("No non-null methods in summary (nothing to plot).")

    labels = [DISPLAY_NAMES.get(m, m) for m in methods]
    x = np.arange(len(methods))
    colors = [COLORS.get(m, "#888888") for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    meta = _meta_title(payload)
    if meta:
        fig.suptitle("Coverage evaluation — " + meta, fontsize=11, y=1.02)

    means, stds = _panel_coverage_means(summary, methods)
    ax = axes[0, 0]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title("Coverage vs. geometric union")
    ax.set_ylim(0.0, min(1.12, max(0.12, max(means) + max(stds) + 0.07)))
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7, alpha=0.55)

    means_h, stds_h = _panel_age_hours(summary, methods)
    ax = axes[0, 1]
    ax.bar(x, means_h, yerr=stds_h, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Hours")
    ax.set_title("Mean age of scanned regions")

    means_lcc, stds_lcc = _panel_metric(summary, methods, "lcc_vs_theoretic_max")
    ax = axes[1, 0]
    if _summary_has_metric(summary, methods, "lcc_vs_theoretic_max"):
        ax.bar(x, means_lcc, yerr=stds_lcc, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        finite_pairs = [(m, s) for m, s in zip(means_lcc, stds_lcc) if np.isfinite(m)]
        ymax_l = max(m + s for m, s in finite_pairs) if finite_pairs else 1.0
        ax.set_ylim(0.0, min(1.12, max(0.12, ymax_l + 0.07)))
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7, alpha=0.55)
    else:
        ax.text(
            0.5,
            0.5,
            "LCC not in JSON — re-run coverage-eval",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#666666",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title("Largest connected component / union")

    means_cam, stds_cam = _panel_camera_sec(summary, methods, payload)
    ax = axes[1, 1]
    ax.bar(x, means_cam, yerr=stds_cam, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Seconds (effective)")
    ax.set_title("Camera on time (effective, not requested)")
    finite_c = [(m, s) for m, s in zip(means_cam, stds_cam) if np.isfinite(m)]
    ymax_c = max(m + s for m, s in finite_c) if finite_c else 1.0
    ax.set_ylim(0.0, ymax_c * 1.15)

    out_path = out_path.with_suffix(f".{fmt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_coverage_eval_separate(
    summary: dict[str, dict | None],
    payload: dict,
    *,
    out_dir: Path,
    dpi: int,
    fmt: str,
    layout: str,
) -> list[Path]:
    """Write one image per panel into ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = _ordered_methods(summary)
    if not methods:
        raise SystemExit("No non-null methods in summary (nothing to plot).")

    labels = [DISPLAY_NAMES.get(m, m) for m in methods]
    x = np.arange(len(methods))
    colors = [COLORS.get(m, "#888888") for m in methods]
    meta = _meta_title(payload)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(fig, stem: str) -> None:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(p)

    # --- Full layout: four separate figures ---
    if layout == "full":
        means, stds = _panel_coverage_means(summary, methods)
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Fraction")
        ax.set_title("Coverage vs. geometric union")
        ax.set_ylim(0.0, min(1.12, max(0.12, max(means) + max(stds) + 0.07)))
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7, alpha=0.55)
        _save(fig, "01_coverage_vs_geometric_union")

        means_h, stds_h = _panel_age_hours(summary, methods)
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        ax.bar(x, means_h, yerr=stds_h, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Hours")
        ax.set_title("Mean age of scanned regions")
        ymax = max(m + s for m, s in zip(means_h, stds_h)) if means_h else 1.0
        ax.set_ylim(0.0, ymax * 1.18)
        _save(fig, "02_mean_age_scanned_regions")

        means_lcc, stds_lcc = _panel_metric(summary, methods, "lcc_vs_theoretic_max")
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        if _summary_has_metric(summary, methods, "lcc_vs_theoretic_max"):
            ax.bar(x, means_lcc, yerr=stds_lcc, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
            finite_pairs = [(m, s) for m, s in zip(means_lcc, stds_lcc) if np.isfinite(m)]
            ymax_l = max(m + s for m, s in finite_pairs) if finite_pairs else 1.0
            ax.set_ylim(0.0, min(1.12, max(0.12, ymax_l + 0.07)))
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7, alpha=0.55)
        else:
            ax.text(
                0.5,
                0.5,
                "LCC not in JSON — re-run coverage-eval",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#666666",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Fraction")
        ax.set_title("Largest connected component / geometric union")
        _save(fig, "03_lcc_vs_geometric_union")

        means_cam, stds_cam = _panel_camera_sec(summary, methods, payload)
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        ax.bar(x, means_cam, yerr=stds_cam, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Seconds (effective)")
        ax.set_title("Camera on time (effective, not requested)")
        finite_c = [(m, s) for m, s in zip(means_cam, stds_cam) if np.isfinite(m)]
        ymax_c = max(m + s for m, s in finite_c) if finite_c else 1.0
        ax.set_ylim(0.0, ymax_c * 1.15)
        _save(fig, "04_effective_camera_on_sec")

    else:
        # simple: four figures (coverage, age, LCC, camera seconds)
        means, stds = _panel_coverage_means(summary, methods)
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Coverage (fraction of geometric union)")
        ax.set_ylim(0.0, min(1.15, max(0.15, max(means) + max(stds) + 0.08)))
        ax.set_title("Coverage vs. theoretic max")
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
        for i, (mu, sig) in enumerate(zip(means, stds)):
            ax.text(
                i,
                mu + sig + 0.02,
                f"{mu:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        _save(fig, "01_coverage_vs_theoretic_max")

        means_h, stds_h = _panel_age_hours(summary, methods)
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        ax.bar(x, means_h, yerr=stds_h, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Mean age of scanned cells (hours)")
        ax.set_title("Mean scan age (end of episode − last_seen)")
        ymax = max(m + s for m, s in zip(means_h, stds_h)) if means_h else 1.0
        ax.set_ylim(0.0, ymax * 1.18)
        _save(fig, "02_mean_scan_age")

        means_lcc, stds_lcc = _panel_metric(summary, methods, "lcc_vs_theoretic_max")
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        if _summary_has_metric(summary, methods, "lcc_vs_theoretic_max"):
            ax.bar(x, means_lcc, yerr=stds_lcc, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
            finite_pairs = [(m, s) for m, s in zip(means_lcc, stds_lcc) if np.isfinite(m)]
            ymax_l = max(m + s for m, s in finite_pairs) if finite_pairs else 1.0
            ax.set_ylim(0.0, min(1.12, max(0.12, ymax_l + 0.07)))
            ax.axhline(1.0, color="#444444", linestyle="--", linewidth=0.7, alpha=0.55)
        else:
            ax.text(
                0.5,
                0.5,
                "LCC not in JSON — re-run coverage-eval",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#666666",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("LCC / geometric union")
        ax.set_title("Largest connected scanned component")
        _save(fig, "03_lcc_vs_geometric_union")

        means_cam, stds_cam = _panel_camera_sec(summary, methods, payload)
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        if meta:
            fig.suptitle("Coverage evaluation — " + meta, fontsize=10, y=1.03)
        ax.bar(x, means_cam, yerr=stds_cam, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylabel("Effective camera on (s)")
        ax.set_title("Camera on time (effective, not requested)")
        finite_c = [(m, s) for m, s in zip(means_cam, stds_cam) if np.isfinite(m)]
        ymax_c = max(m + s for m, s in finite_c) if finite_c else 1.0
        ax.set_ylim(0.0, ymax_c * 1.15)
        _save(fig, "04_effective_camera_on_sec")

    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Bar charts from coverage-eval JSON")
    p.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="JSON from: python -m adaptive_scanning.run_sim coverage-eval ... --out …",
    )
    p.add_argument(
        "--out",
        "-o",
        type=str,
        default="",
        help="Output image path (combined mode) or ignored if --separate with --out-dir",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="With --separate: folder for individual charts (created if missing)",
    )
    p.add_argument(
        "--separate",
        action="store_true",
        help="Write one image file per panel instead of a single combined figure",
    )
    p.add_argument(
        "--layout",
        choices=("simple", "full"),
        default="full",
        help="simple / full: 2×2 combined (coverage, age, LCC, effective camera seconds). "
        "With --separate, full writes 01–04 charts; simple uses theoretic-max labels for coverage. "
        "Legacy JSON (camera_on_fraction, random_iid) is still supported.",
    )
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--format", choices=("png", "pdf"), default="png")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"not found: {inp.resolve()}")

    summary, payload = _load_summary(inp)
    stem = inp.stem + "_plots"
    out_default = inp.parent / stem

    if args.separate:
        out_dir = Path(args.out_dir) if (args.out_dir or "").strip() else inp.parent / f"{inp.stem}_figures"
        paths = plot_coverage_eval_separate(
            summary,
            payload,
            out_dir=out_dir,
            dpi=args.dpi,
            fmt=args.format,
            layout=args.layout,
        )
        for path in paths:
            print(str(path.resolve()))
    else:
        out_base = Path(args.out) if (args.out or "").strip() else out_default
        if args.layout == "simple":
            path = plot_coverage_eval(
                summary, payload, out_path=out_base, dpi=args.dpi, fmt=args.format
            )
        else:
            path = plot_coverage_eval_fullpage(
                summary, payload, out_path=out_base, dpi=args.dpi, fmt=args.format
            )
        print(str(path.resolve()))


if __name__ == "__main__":
    main()
