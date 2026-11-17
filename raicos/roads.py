"""Road networks and road-constrained targets with four motion regimes.

The article couples Eclipse SUMO mobility (TraCI) with OpenStreetMap roads
for Beijing (train/validation/test), Chicago-grid and Normal (held-out
transfer). This repository ships synthetic stand-ins with the same roles so
it runs offline; ``RoadMap.from_edge_list`` accepts any road graph (e.g. one
exported from SUMO/OSM) as a list of node coordinates and edges.

Targets switch among the four IMM regimes of Eq. (3): constant velocity,
slow/stop, acceleration and coordinated turn. Road-transition feasibility is
known to the map, never a target identity.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

CV, SLOW, ACCEL, TURN = 0, 1, 2, 3


class RoadMap:
    def __init__(self, nodes: np.ndarray, edges: List[Tuple[int, int]], name: str):
        self.nodes = np.asarray(nodes, float)
        self.name = name
        self.adj: Dict[int, List[int]] = {i: [] for i in range(len(nodes))}
        for a, b in edges:
            if b not in self.adj[a]:
                self.adj[a].append(b)
                self.adj[b].append(a)
        self.extent = float(self.nodes.max())

    @staticmethod
    def from_edge_list(nodes, edges, name="custom") -> "RoadMap":
        return RoadMap(np.asarray(nodes, float), list(edges), name)

    @staticmethod
    def build(name: str, seed: int = 0) -> "RoadMap":
        rng = np.random.default_rng(10_000 + seed)
        if name == "chicago-grid":            # regular orthogonal grid
            n, s, jitter, drop = 7, 150.0, 0.0, 0.0
        elif name == "normal":                 # sparse small-town grid
            n, s, jitter, drop = 5, 220.0, 12.0, 0.18
        else:                                  # "beijing": irregular arterial grid
            n, s, jitter, drop = 6, 180.0, 25.0, 0.08
        idx = {}
        nodes = []
        for i in range(n):
            for j in range(n):
                idx[(i, j)] = len(nodes)
                nodes.append([i * s + rng.normal(0, jitter), j * s + rng.normal(0, jitter)])
        nodes = np.array(nodes) - np.min(nodes, 0)
        edges = []
        for i in range(n):
            for j in range(n):
                for di, dj in ((1, 0), (0, 1)):
                    if i + di < n and j + dj < n:
                        border = i in (0, n - 1) or j in (0, n - 1)
                        if border or rng.random() > drop:
                            edges.append((idx[(i, j)], idx[(i + di, j + dj)]))
        return RoadMap(nodes, edges, name)


class RoadTarget:
    """A vehicle moving along road edges with regime switching."""

    def __init__(self, rmap: RoadMap, rng: np.random.Generator, tid: int):
        self.map, self.rng, self.tid = rmap, rng, tid
        self.u = int(rng.integers(len(rmap.nodes)))
        self.v = int(rng.choice(rmap.adj[self.u]))
        self.s = rng.uniform(0, self._len())
        self.cruise = rng.uniform(7.0, 13.0)
        self.speed = self.cruise
        self.mode = CV
        self.stop_left = 0.0
        self.turn_left = 0.0
        self.pos = self._xy()
        self.vel = self._dir() * self.speed

    def _len(self) -> float:
        return float(np.linalg.norm(self.map.nodes[self.v] - self.map.nodes[self.u]))

    def _dir(self) -> np.ndarray:
        d = self.map.nodes[self.v] - self.map.nodes[self.u]
        return d / (np.linalg.norm(d) + 1e-9)

    def _xy(self) -> np.ndarray:
        return self.map.nodes[self.u] + self._dir() * self.s

    def step(self, dt: float) -> None:
        rng = self.rng
        if self.mode == SLOW:
            self.speed = max(0.0, self.speed - 2.5 * dt)
            self.stop_left -= dt
            if self.stop_left <= 0:
                self.mode = ACCEL
        elif self.mode == ACCEL:
            self.speed = min(self.cruise, self.speed + 2.0 * dt)
            if self.speed >= self.cruise - 1e-6:
                self.mode = CV
        elif self.mode == TURN:
            self.turn_left -= dt
            if self.turn_left <= 0:
                self.mode = ACCEL
        else:
            self.speed = float(np.clip(self.speed + rng.normal(0, 0.15), 3.0, 15.0))
        old_dir = self._dir()
        self.s += self.speed * dt
        while self.s >= self._len():
            self.s -= self._len()
            prev = self.u
            self.u = self.v
            options = [w for w in self.map.adj[self.u] if w != prev] or self.map.adj[self.u]
            self.v = int(rng.choice(options))
            turn = float(np.dot(old_dir, self._dir())) < 0.7
            if turn:
                self.mode, self.turn_left = TURN, 3.0
                self.speed = min(self.speed, 6.0)
                self.s = min(self.s, 3.0)
            elif rng.random() < 0.25:
                self.mode, self.stop_left = SLOW, rng.uniform(3.0, 8.0)
        new = self._xy()
        self.vel = (new - self.pos) / dt
        self.pos = new
