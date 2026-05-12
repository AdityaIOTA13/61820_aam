#!/usr/bin/env python3
"""
plot_cambridge_3panel.py
========================
Reads cambridge_coverage_eval.json and produces a clean 1×3 bar chart
(Coverage, Mean Age, Camera On Time) with large fonts. No LCC panel.

Usage
-----
  python scripts/plot_cambridge_3panel.py
  python scripts/plot_cambridge_3panel.py --input outputs/cambridge_coverage_eval.json
  python scripts/plot_cambridge_3panel.py --out outputs/cambridge_3panel.png --dpi 180
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IN  = ROOT / "data_02_outputs" / "cambridge_coverage_eval.json"
DEFAULT_OUT = ROOT / "data_02_outputs" / "cambridge_3panel.png"

METHOD_ORDER = ["random", "greedy_unseen", "greedy_unseen_progressive", "oracle"]
LABELS = {
    "random":                    "Random\n(full budget)",
    "greedy_unseen":             "Greedy\nunseen",
    "greedy_unseen_progressive": "Greedy unseen\n+ late rescan",
    "oracle":                    "Oracle",
}
COLORS = {
    "random":                    "#7eb0d5",
    "greedy_unseen":             "#bd7ebe",
    "greedy_unseen_progressive": "#8c9e5e",
    "oracle":                    "#ffb55a",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default=str(DEFAULT_IN))
    ap.add_argument("--out",   "-o", default=str(DEFAULT_OUT))
    ap.add_argument("--dpi",         type=int, default=180)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload["summary"]

    # Filter to methods present and non-null
    methods = [m for m in METHOD_ORDER if summary.get(m)]
    labels  = [LABELS[m] for m in methods]
    colors  = [COLORS[m] for m in methods]

    def get(m, key):
        return float(summary[m].get(f"{key}_mean", float("nan")))
    def getstd(m, key):
        return float(summary[m].get(f"{key}_std", 0.0))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FONT = 18
    TITLE_FONT = 20
    matplotlib.rcParams.update({
        "font.size":        FONT,
        "axes.titlesize":   TITLE_FONT,
        "axes.labelsize":   FONT,
        "xtick.labelsize":  FONT - 2,
        "ytick.labelsize":  FONT - 2,
        "figure.titlesize": TITLE_FONT + 2,
    })

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    meta_parts = []
    if payload.get("n_scenarios"):
        meta_parts.append(f"N = {payload['n_scenarios']} scenarios")
    if payload.get("oracle_mode"):
        meta_parts.append(f"oracle: {payload['oracle_mode']}")
    if meta_parts:
        fig.suptitle("Coverage evaluation — " + " · ".join(meta_parts), fontsize=TITLE_FONT)

    # ── Panel 1: Coverage ────────────────────────────────────────────────
    ax = axes[0]
    means = [get(m, "coverage_vs_theoretic_max") for m in methods]
    stds  = [getstd(m, "coverage_vs_theoretic_max") for m in methods]
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction")
    ax.set_title("Coverage vs. geometric union")
    ax.set_ylim(0, min(1.13, max(m+s for m,s in zip(means,stds)) + 0.08))
    ax.axhline(1.0, color="#444", linestyle="--", linewidth=0.9, alpha=0.6)
    for i, (mu, sig) in enumerate(zip(means, stds)):
        ax.text(i, mu + sig + 0.01, f"{mu:.2f}", ha="center", va="bottom",
                fontsize=FONT - 2, color="#333")

    # ── Panel 2: Mean age ────────────────────────────────────────────────
    ax = axes[1]
    means_h = [get(m, "mean_age_scanned_sec") / 3600.0 for m in methods]
    stds_h  = [getstd(m, "mean_age_scanned_sec") / 3600.0 for m in methods]
    ax.bar(x, means_h, yerr=stds_h, capsize=5, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Hours")
    ax.set_title("Mean age of scanned regions")
    ymax = max(m+s for m,s in zip(means_h, stds_h)) if means_h else 1.0
    ax.set_ylim(0, ymax * 1.2)

    # ── Panel 3: Camera on time ──────────────────────────────────────────
    ax = axes[2]
    means_c = [get(m, "effective_camera_on_sec") for m in methods]
    stds_c  = [getstd(m, "effective_camera_on_sec") for m in methods]
    ax.bar(x, means_c, yerr=stds_c, capsize=5, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds (effective)")
    ax.set_title("Camera on time (effective)")
    ymax_c = max(m+s for m,s in zip(means_c, stds_c)) if means_c else 1.0
    ax.set_ylim(0, ymax_c * 1.15)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()