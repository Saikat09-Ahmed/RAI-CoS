#!/usr/bin/env python
"""Paired multi-seed evaluation with hierarchical bootstrap intervals.

Suites
  nominal   Table 2 analogue: methods at the nominal operating point
  severe    Table 3 analogue: compound-severe network + source stress
  ablation  Table 7 analogue: one-component changes of RAI-CoS
  transfer  held-out maps (chicago-grid, normal)
  drift     abrupt reliability drift (calibration / recovery endpoints)

    python scripts/run_benchmark.py --suite nominal --seeds 0 1 2
    python scripts/run_benchmark.py --suite ablation --seeds 0 1 2 --duration 300
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raicos.config import ExperimentConfig, severe  # noqa: E402
from raicos.env import run_episode  # noqa: E402

SUITES = {
    "nominal": ["local", "fifo", "allshare", "aoi-ds", "raicos", "raicos-exact", "perfectshare"],
    "severe": ["local", "fifo", "aoi-ds", "raicos", "perfectshare"],
    "ablation": ["raicos", "raicos-no-rel", "raicos-no-consistency", "raicos-arrival", "raicos-exp-aoi",
                 "raicos-no-prov", "raicos-naive-fusion", "raicos-fixed-threshold", "raicos-no-receiver-gain",
                 "raicos-no-pdel", "raicos-posterior-mean", "raicos-no-forgetting"],
    "transfer": ["fifo", "aoi-ds", "raicos"],
    "drift": ["raicos", "raicos-no-forgetting", "raicos-posterior-mean", "raicos-no-rel"],
}
KEYS = ["continuity_pct", "captures", "rmse_m", "nees_pos", "reacq_s", "tx_mb", "budget_violation_pct",
        "wh_per_target", "decision_p95_ms", "brier", "ece", "auroc", "harmful_accept_pct", "useful_reject_pct"]


def make_cfg(method, seed, suite, duration, road_map="beijing", actor="scripted", weights=""):
    cfg = ExperimentConfig(method=method, seed=seed, actor=actor, actor_weights=weights)
    cfg.scenario.duration_s = duration
    cfg.scenario.road_map = road_map
    if suite == "severe":
        severe(cfg)
    if suite == "drift":
        cfg.scenario.reliability_drift_at_s = duration / 3
    return cfg


def _job(args):
    return run_episode(make_cfg(*args))


def bootstrap_halfwidth(x, n=10000, seed=0):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float((hi - lo) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="nominal", choices=list(SUITES))
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--maps", nargs="+", default=None)
    ap.add_argument("--actor", default="scripted", choices=["scripted", "learned"])
    ap.add_argument("--weights", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    methods = a.methods or SUITES[a.suite]
    maps = a.maps or (["chicago-grid", "normal"] if a.suite == "transfer" else ["beijing"])
    jobs = [(m, s, a.suite, a.duration, mp, a.actor, a.weights) for mp in maps for m in methods for s in a.seeds]
    with Pool(a.workers) as p:
        runs = p.map(_job, jobs)
    os.makedirs(a.out, exist_ok=True)
    tag = a.suite
    with open(os.path.join(a.out, f"{tag}_runs.json"), "w") as fh:
        json.dump(runs, fh, indent=1)
    lines = ["| Map | Method | " + " | ".join(KEYS) + " |", "|" + "---|" * (len(KEYS) + 2)]
    for mp in maps:
        for m in methods:
            rs = [r for r in runs if r["method"] == m and r["map"] == mp]
            cells = []
            for k in KEYS:
                v = [r[k] for r in rs]
                mu = np.nanmean(v) if any(x == x for x in v) else float("nan")
                hw = bootstrap_halfwidth(v)
                cells.append(f"{mu:.1f} ± {hw:.1f}" if k in ("continuity_pct", "rmse_m", "reacq_s") and hw == hw
                             else (f"{mu:.3f}" if k in ("brier", "ece", "auroc") else f"{mu:.2f}"))
            lines.append(f"| {mp} | {m} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(a.out, f"{tag}_summary.md"), "w") as fh:
        fh.write(f"suite={a.suite} seeds={a.seeds} duration={a.duration}s actor={a.actor}\n\n{table}\n")


if __name__ == "__main__":
    main()
