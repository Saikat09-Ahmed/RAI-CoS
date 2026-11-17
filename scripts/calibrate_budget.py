#!/usr/bin/env python
"""Apply the article's budget rule in this simulator: B_bar is 50% of the
uncapped AllShare per-UAV attempted-byte rate at the nominal point (the
article's 5.17 KB/s came from its own 49.6 MB AllShare reference).

    python scripts/calibrate_budget.py --seeds 0 1 --fraction 0.5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from raicos.config import ExperimentConfig  # noqa: E402
from raicos.env import run_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=600.0)
    a = ap.parse_args()
    rates = []
    for s in a.seeds:
        cfg = ExperimentConfig(method="allshare", seed=s)
        cfg.scenario.duration_s = a.duration
        rates.append(run_episode(cfg)["tx_kb_per_uav_s"] * 1000)
    m = float(np.mean(rates))
    print(f"uncapped AllShare: {m:.0f} B/s per UAV -> B_bar = {a.fraction * m:.0f} B/s "
          f"(set PacketConfig.budget_bytes_per_s)")


if __name__ == "__main__":
    main()
