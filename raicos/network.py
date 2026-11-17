"""Intermittent network, byte ledger, acknowledgments, link estimates and
clock filtering (Sec. 3.3, 3.4, 5.3).

* i.i.d. or Gilbert-Elliott burst loss, Gaussian delay with jitter
  (reordering emerges naturally), blackout windows, radio-range mask and a
  per-link per-epoch byte capacity. Every attempted packet is charged to its
  originating UAV, including packets that are dropped.
* Delivered UAV-to-UAV packets are acknowledged with a counted 32-byte
  envelope. Senders learn a per-link delivery estimate and one-way delay
  moments from ACKs (EWMA) and time out unacknowledged packets.
* Two-way timestamps in ACKs feed a per-neighbour clock filter (offset and
  offset variance). After 5 s without acknowledged traffic a 32-byte sync
  envelope is exchanged in each direction.
* A replay trace (CSV: time_s,src,dst,delivered,delay_s) can replace the
  stochastic channel so all methods see identical delivery outcomes.
"""
from __future__ import annotations

import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class LinkEstimate:
    p_link: float = 0.8
    d_mean: float = 2.0
    d_var: float = 0.5
    last_ack_t: float = -1e9

    def p_deliver_by(self, horizon_s: float) -> float:
        """Probability of delivery within `horizon_s` (loss x delay CDF)."""
        if horizon_s <= 0:
            return 0.0
        return float(self.p_link * norm.cdf(horizon_s, self.d_mean, np.sqrt(max(self.d_var, 1e-4))))


class ClockFilter:
    """Offset of a neighbour's clock relative to ours, from two-way timestamps."""

    def __init__(self):
        self.offset = 0.0
        self.var = 0.05 ** 2
        self.n = 0
        self.best_rtt = np.inf

    def observe(self, t1, t2, t3, t4) -> None:
        rtt = (t4 - t1) - (t3 - t2)
        theta = ((t2 - t1) + (t3 - t4)) / 2.0          # their clock - our clock
        self.best_rtt = min(self.best_rtt, rtt)
        w = 0.3 if rtt <= self.best_rtt * 1.5 + 0.2 else 0.05   # favour minimum-delay samples
        self.n += 1
        d = theta - self.offset
        self.offset += w * d
        self.var = (1 - w) * self.var + w * d * d


@dataclass
class Packet:
    kind: str           # point | digest | ack | sync | ingress
    src: str
    dst: str
    payload: Any
    nbytes: int
    t_send: float       # true send time
    t_send_src_clock: float
    pid: int = 0


class Network:
    def __init__(self, cfg, pcfg, seed: int, ideal: bool = False):
        self.cfg, self.pcfg, self.ideal = cfg, pcfg, ideal
        self.rng = np.random.default_rng(seed + 7777)
        self.q: List[Tuple[float, int, Packet]] = []
        self._pid = 0
        self.ge_state: Dict[Tuple[str, str], int] = defaultdict(int)
        self.epoch_bytes: Dict[Tuple[str, str], int] = defaultdict(int)
        self.epoch_t = -1.0
        self.tx_bytes: Dict[str, int] = defaultdict(int)      # attempted, UAV-originated
        self.rx_bytes: Dict[str, int] = defaultdict(int)
        self.ingress_bytes = 0
        self.trace = None
        if cfg.trace_file:
            self.trace = defaultdict(list)
            with open(cfg.trace_file) as fh:
                for r in csv.DictReader(fh):
                    self.trace[(round(float(r["time_s"])), r["src"], r["dst"])].append(
                        (r["delivered"] == "1", float(r["delay_s"])))

    def _blackout(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.cfg.blackouts)

    def _lost(self, link, t) -> bool:
        p = self.cfg.loss
        if not self.cfg.burst:
            return self.rng.random() < p
        # Gilbert-Elliott with mean loss p: bad-state loss 0.8, good 0.02
        s = self.ge_state[link]
        pb = np.clip((p - 0.02) / 0.78, 0, 1)
        to_bad = 0.1 * pb / max(1 - pb, 1e-3)
        if s == 0 and self.rng.random() < to_bad:
            s = 1
        elif s == 1 and self.rng.random() < 0.1:
            s = 0
        self.ge_state[link] = s
        return self.rng.random() < (0.8 if s else 0.02)

    def send(self, pkt: Packet, dist: float, t: float, charge: bool = True) -> None:
        if t != self.epoch_t:
            self.epoch_t = t
            self.epoch_bytes.clear()
        self._pid += 1
        pkt.pid = self._pid
        if pkt.kind == "ingress":
            self.ingress_bytes += pkt.nbytes
        elif charge:
            self.tx_bytes[pkt.src] += pkt.nbytes
        if self.ideal:
            heapq.heappush(self.q, (t, pkt.pid, pkt))
            return
        if dist > self.cfg.radio_range_m or self._blackout(t):
            return
        link = (pkt.src, pkt.dst)
        self.epoch_bytes[link] += pkt.nbytes
        if self.epoch_bytes[link] > self.cfg.link_capacity_bytes:
            return
        if self.trace is not None:
            outs = self.trace.get((round(t), pkt.src, pkt.dst))
            if outs:
                ok, delay = outs.pop(0)
                if ok:
                    heapq.heappush(self.q, (t + delay, pkt.pid, pkt))
                return
        if self._lost(link, t):
            return
        delay = max(0.02, self.cfg.delay_mean_s + self.rng.normal(0, self.cfg.jitter_s))
        heapq.heappush(self.q, (t + delay, pkt.pid, pkt))

    def deliver(self, t: float) -> List[Tuple[float, Packet]]:
        out = []
        while self.q and self.q[0][0] <= t + 1e-9:
            ta, _, p = heapq.heappop(self.q)
            if p.kind != "ingress":
                self.rx_bytes[p.dst] += p.nbytes
            out.append((ta, p))
        return out
