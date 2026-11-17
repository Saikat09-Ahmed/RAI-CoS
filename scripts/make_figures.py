#!/usr/bin/env python
"""Figures from the benchmark outputs in results/."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"raicos": "#1f5fbf", "raicos-exact": "#5b8bd9", "perfectshare": "#9a9a9a", "local": "#6b6b6b",
       "fifo": "#3c9a5f", "allshare": "#c43c3c", "aoi-ds": "#e08a00"}


def nominal(res):
    runs = json.load(open(os.path.join(res, "nominal_runs.json")))
    methods = list(dict.fromkeys(r["method"] for r in runs))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in methods:
        rs = [r for r in runs if r["method"] == m]
        x = np.mean([r["tx_mb"] for r in rs])
        y = np.mean([r["continuity_pct"] for r in rs])
        ys = np.std([r["continuity_pct"] for r in rs])
        ax.errorbar(x, y, yerr=ys, fmt="o", color=COL.get(m, "k"), capsize=3, ms=7)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("attempted UAV-originated traffic per episode (MB)")
    ax.set_ylabel("tracking continuity (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(res, "nominal_continuity_vs_traffic.png"), dpi=160)


def grid(res):
    path = os.path.join(res, "grid.json")
    if not os.path.exists(path):
        return
    g = json.load(open(path))
    methods = list(dict.fromkeys(r["method"] for r in g))
    losses = sorted({r["loss"] for r in g})
    delays = sorted({r["delay"] for r in g})
    fig, axes = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 3.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        M = np.array([[np.mean([r["continuity"] for r in g if r["method"] == m and r["loss"] == l and r["delay"] == d])
                       for d in delays] for l in losses])
        im = ax.imshow(M, vmin=20, vmax=80, cmap="Blues", origin="lower", aspect="auto")
        for i in range(len(losses)):
            for j in range(len(delays)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 55 else "black")
        ax.set_xticks(range(len(delays)), [f"{d:g}" for d in delays])
        ax.set_yticks(range(len(losses)), [f"{int(l * 100)}%" for l in losses])
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("mean one-way delay (s)")
    np.atleast_1d(axes)[0].set_ylabel("packet loss")
    fig.colorbar(im, ax=axes, shrink=0.8, label="continuity (%)")
    fig.savefig(os.path.join(res, "loss_delay_grid.png"), dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    nominal(res)
    grid(res)
    print("figures written to", res)
