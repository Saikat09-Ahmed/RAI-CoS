#!/usr/bin/env python
"""Train the recurrent set-attention actor with MAPPO + PopArt.

Article protocol: 5 seeds x 10,000,000 environment steps, 32 parallel
environments, checkpoints screened every 0.5 M steps from 2 M to 10 M on
validation episodes, selection by validation continuity among byte-feasible
checkpoints. Reproducing that needs a cluster; the defaults below run a short
demonstration.

    python scripts/train_mappo.py --steps 2000 --envs 2 --out results/actor.pt
    python scripts/run_benchmark.py --suite nominal --methods raicos --actor learned --weights results/actor.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from raicos.config import ExperimentConfig  # noqa: E402
from raicos.mappo import MAPPOTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="raicos")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--episode-s", type=float, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/actor.pt")
    a = ap.parse_args()
    cfg = ExperimentConfig(method=a.method)
    cfg.scenario.duration_s = a.episode_s
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    MAPPOTrainer(cfg, n_envs=a.envs, seq_len=a.seq_len, seed=a.seed).train(a.steps, ckpt=a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
