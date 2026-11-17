#!/usr/bin/env python
"""Packet-loss x one-way-delay continuity grid (Figure 4 analogue).

    python scripts/run_grid.py --losses 0 0.2 0.4 --delays 0 2 5 --seeds 0 --duration 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from raicos.config import ExperimentConfig  # noqa: E402
from raicos.env import run_episode  # noqa: E402


def _job(args):
    m, s, loss, delay, dur = args
    cfg = ExperimentConfig(method=m, seed=s)
    cfg.network.loss, cfg.network.delay_mean_s = loss, delay
    cfg.scenario.duration_s = dur
    r = run_episode(cfg)
    return dict(method=m, seed=s, loss=loss, delay=delay, continuity=r["continuity_pct"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["fifo", "aoi-ds", "raicos"])
    ap.add_argument("--losses", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--delays", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="results/grid.json")
    a = ap.parse_args()
    jobs = [(m, s, l, d, a.duration) for m in a.methods for s in a.seeds for l in a.losses for d in a.delays]
    with Pool(a.workers) as p:
        res = p.map(_job, jobs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for m in a.methods:
        print(f"\n{m}: rows = loss, columns = delay (s) {a.delays}")
        for l in a.losses:
            row = [np.mean([r["continuity"] for r in res if r["method"] == m and r["loss"] == l and r["delay"] == d])
                   for d in a.delays]
            print(f"  {int(l * 100):3d}%  " + "  ".join(f"{v:5.1f}" for v in row))


if __name__ == "__main__":
    main()
