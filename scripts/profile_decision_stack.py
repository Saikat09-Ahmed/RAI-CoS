#!/usr/bin/env python
"""Decision-stack trace replay: per-UAV time from received-report processing
through action output, after warm-up (Sec. 5.9). The article measured 200
warm-up + 5000 decisions on a Jetson Orin NX 16 GB in 25 W mode; run this on
the target board for the same style of measurement.

    python scripts/profile_decision_stack.py --decisions 5000 --warmup 200
"""
from __future__ import annotations

import argparse
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from raicos.config import ExperimentConfig  # noqa: E402
from raicos.env import Episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="raicos")
    ap.add_argument("--decisions", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--actor", default="scripted")
    ap.add_argument("--weights", default="")
    a = ap.parse_args()
    cfg = ExperimentConfig(method=a.method, actor=a.actor, actor_weights=a.weights)
    cfg.scenario.duration_s = 1e9
    ep = Episode(cfg)
    while len(ep.decision_ms) < a.warmup + a.decisions:
        ep.step()
    d = np.array(ep.decision_ms[a.warmup:a.warmup + a.decisions])
    print(f"host {platform.machine()} python {platform.python_version()}  decisions {len(d)}")
    print(f"per-UAV decision stack: mean {d.mean():.2f} ms  P50 {np.percentile(d, 50):.2f}  P95 {np.percentile(d, 95):.2f}  P99 {np.percentile(d, 99):.2f}")


if __name__ == "__main__":
    main()
