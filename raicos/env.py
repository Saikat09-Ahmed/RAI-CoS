"""Closed-loop RAI-CoS episode: sensing, ingress, intermittent delivery,
belief reconstruction, digests, scheduling, decentralized control and
evaluation (Sec. 3-5).

Methods (`ExperimentConfig.method`):

  local          only local sensing, no communication (Local R-MAPPO analogue)
  fifo           budget-matched FIFO queue, fixed source-type weights,
                 arrival-time fusion (BaseActor-FIFO analogue)
  allshare       attempt every report to every neighbour, uncapped (AllShare)
  aoi-ds         freshest-first budget-matched queue + freshness-weighted
                 evidence (AoI-DS analogue)
  raicos         complete RAI-CoS
  raicos-exact   RAI-CoS with the 16-byte-quantised exact scheduler
  perfectshare   ideal synchronized links and information (oracle)
Ablations (Table 7): raicos-no-rel, raicos-no-consistency, raicos-arrival,
raicos-exp-aoi, raicos-no-prov, raicos-naive-fusion, raicos-fixed-threshold,
raicos-no-receiver-gain, raicos-no-pdel, raicos-posterior-mean,
raicos-no-forgetting.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .admission import ACCEPT, DOWNWEIGHT, REJECT, r_src, r_time
from .config import ExperimentConfig
from .network import ClockFilter, LinkEstimate, Network, Packet
from .reliability import source_type
from .roads import RoadMap, RoadTarget
from .scheduler import (BROAD_P, Candidate, VirtualQueue, duplicate_risk, exact_select,
                        info_gain, urgency)
from .sources import SourceField
from .tracker import BeliefManager, Flags

ACTIONS = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], float)    # hover, N, S, E, W
P_LOSS = {"active": 0.0, "reacquiring": 0.5, "uncertain": 0.5, "lost": 1.0}

METHODS = {
    "local": dict(comm="none", flags=dict()),
    "fifo": dict(comm="fifo", flags=dict(reliability=False, consistency=False, generation_time=False)),
    "allshare": dict(comm="all", flags=dict(reliability=False, consistency=False, generation_time=False)),
    "aoi-ds": dict(comm="fresh", flags=dict(reliability=False, consistency=False, generation_time=False,
                                            freshness_tau=6.0)),
    "raicos": dict(comm="value", flags=dict()),
    "raicos-exact": dict(comm="value", exact=True, flags=dict()),
    "perfectshare": dict(comm="all", ideal=True, flags=dict()),
    # one-component ablations
    "raicos-no-rel": dict(comm="value", flags=dict(reliability=False)),
    "raicos-no-consistency": dict(comm="value", flags=dict(consistency=False)),
    "raicos-arrival": dict(comm="value", flags=dict(generation_time=False)),
    "raicos-exp-aoi": dict(comm="value", value="exp-aoi", flags=dict()),
    "raicos-no-prov": dict(comm="value", flags=dict(provenance=False)),
    "raicos-naive-fusion": dict(comm="value", flags=dict(ci=False)),
    "raicos-fixed-threshold": dict(comm="value", value="fixed-threshold", flags=dict()),
    "raicos-no-receiver-gain": dict(comm="value", value="no-receiver", flags=dict()),
    "raicos-no-pdel": dict(comm="value", value="no-pdel", flags=dict()),
    "raicos-posterior-mean": dict(comm="value", flags=dict(lower_bound=False)),
    "raicos-no-forgetting": dict(comm="value", flags=dict(forgetting=False)),
    "raicos-no-digest-fusion": dict(comm="value", no_digest=True, flags=dict()),
}


@dataclass
class Digest:
    sender: str
    x: np.ndarray
    P: np.ndarray
    t_ref: float                 # sender clock
    status: str
    n_accept: int
    ledger: tuple
    last_update_t: float         # sender clock
    sender_pos: np.ndarray
    prioritized: bool = True
    label: int = -1                                      # evaluator-only metadata


@dataclass
class Held:
    report: object
    tg: float
    tid: Optional[int]
    t_recv: float
    decision: str


class UAVAgent:
    def __init__(self, uid: int, cfg: ExperimentConfig, flags: Flags, pos, clock, rng):
        self.uid, self.name = uid, f"uav{uid}"
        self.cfg = cfg
        self.pos = np.asarray(pos, float)
        self.vel = np.zeros(2)
        self.clock = clock                        # hidden offset of this UAV's clock
        self.bm = BeliefManager(uid, cfg, flags, rng)
        self.held: Dict[int, Held] = {}
        self.fifo: List[Tuple[int, str]] = []
        self.sent: Dict[Tuple[int, str], float] = {}
        self.inflight: Dict[int, Tuple[str, float, int]] = {}    # pid -> (peer, t_send_local, uuid)
        self.links: Dict[str, LinkEstimate] = defaultdict(lambda: LinkEstimate(
            p_link=1 - cfg.network.loss, d_mean=cfg.network.delay_mean_s, d_var=cfg.network.jitter_s ** 2 + 0.05))
        self.clocks: Dict[str, ClockFilter] = defaultdict(ClockFilter)
        self.peer_digest: Dict[Tuple[str, int], Tuple[Digest, float]] = {}
        self.peer_last: Dict[str, Tuple[Digest, float]] = {}
        self.vq = VirtualQueue(cfg.packets.budget_bytes_per_s)
        self.bucket = 0.0
        self.priority: Optional[int] = None
        self.hold = 0
        self.done: set = set()
        self.energy_wh = cfg.energy.battery_wh
        self.flight_wh = self.compute_wh = 0.0
        self.bytes_epoch = 0
        self.ctrl_bytes_to: Dict[str, int] = defaultdict(int)
        self.patrol = None
        self.last_ack: Dict[str, float] = defaultdict(lambda: 0.0)

    def local_time(self, t: float) -> float:
        return t + self.clock


class Episode:
    def __init__(self, cfg: ExperimentConfig, record: bool = False):
        self.cfg = cfg
        sc = cfg.scenario
        spec = METHODS[cfg.method]
        self.spec = spec
        self.rng = np.random.default_rng(cfg.seed)
        self.map = RoadMap.build(sc.road_map, cfg.seed if sc.road_map == "beijing" else 0)
        self.targets = [RoadTarget(self.map, self.rng, j) for j in range(sc.num_targets)]
        # declared births: staggered over the first 60% of the episode
        self.birth = np.sort(self.rng.uniform(0, 0.6 * sc.duration_s, sc.num_targets))
        self.birth[: max(1, sc.num_targets // 5)] = 0.0
        self.field = SourceField(sc, self.map, self.rng)
        self.net = Network(cfg.network, cfg.packets, cfg.seed, ideal=spec.get("ideal", False))
        flags = Flags(**spec["flags"])
        ext = self.map.extent
        g = int(np.ceil(np.sqrt(sc.num_uavs)))
        self.sectors = [np.array([(i % g + 0.5) * ext / g, (i // g + 0.5) * ext / g]) for i in range(sc.num_uavs)]
        self.agents = [UAVAgent(i, cfg, flags, self.sectors[i] + self.rng.normal(0, 20, 2),
                                self.field.clock[f"uav{i}"], self.rng) for i in range(sc.num_uavs)]
        self.by_name = {a.name: a for a in self.agents}
        self.t = 0.0
        self.captured: Dict[int, float] = {}
        self.cap_hold: Dict[Tuple[int, int], int] = defaultdict(int)
        self.m = defaultdict(list)
        self.cont_state = {j: False for j in range(sc.num_targets)}
        self.gap_start: Dict[int, float] = {}
        self.reacq_times: List[float] = []
        self.track_losses = 0
        self.decision_ms: List[float] = []
        self.collisions = 0
        self.record = record
        self.series = []
        self.actor = None
        if cfg.actor == "learned":
            from .policy import LearnedActor
            self.actor = LearnedActor(cfg.actor_weights)

    # ------------------------------------------------------------ utilities
    def neighbours(self, a: UAVAgent) -> List[UAVAgent]:
        others = [b for b in self.agents if b is not a and
                  np.linalg.norm(b.pos - a.pos) <= self.cfg.network.radio_range_m]
        others.sort(key=lambda b: (np.linalg.norm(b.pos - a.pos), b.uid))
        return others[: self.cfg.network.neighbourhood]

    def _send(self, a: UAVAgent, b: UAVAgent, kind: str, payload, nbytes: int, uuid: int = -1) -> None:
        tl = a.local_time(self.t)
        pkt = Packet(kind, a.name, b.name, payload, nbytes, self.t, tl)
        self.net.send(pkt, float(np.linalg.norm(a.pos - b.pos)), self.t)
        a.bytes_epoch += nbytes
        a.ctrl_bytes_to[b.name] += nbytes if kind != "point" else 0
        if kind in ("point", "digest"):
            a.inflight[pkt.pid] = (b.name, tl, uuid)

    def _convert(self, a: UAVAgent, source: str, tau: float) -> Tuple[float, float]:
        """Map a source timestamp into a's clock; returns (time, timing sd)."""
        st = source_type(source)
        if st == "uav":
            if source == a.name:
                return tau, 0.0
            cf = a.clocks[source]
            return tau - cf.offset, float(np.sqrt(cf.var))
        # exogenous sources: no two-way exchange, conservative default
        return tau, 0.1 if st == "monitor" else 0.25

    # --------------------------------------------------------------- phases
    def alive(self):
        return [tg for j, tg in enumerate(self.targets) if self.birth[j] <= self.t]

    def _ingress(self) -> None:
        if self.spec["comm"] == "none":
            return
        for rep in self.field.exogenous_reports(self.alive(), self.t):
            sp = self.field.position_of(rep.source)
            d = [np.linalg.norm(a.pos - sp) for a in self.agents]
            k = int(np.argmin(d))
            pkt = Packet("ingress", rep.source, self.agents[k].name, rep, self.cfg.packets.point_payload, self.t, self.t)
            self.net.send(pkt, float(d[k]), self.t)

    def _receive(self, a: UAVAgent, items, now_l: float) -> None:
        cfg = self.cfg
        batch, digests, ingress_ids = [], [], set()
        for ta, p in items:
            if p.kind == "ack":
                pid, t2 = p.payload
                info = a.inflight.pop(pid, None)
                if info:
                    peer, t1, uuid = info
                    le = a.links[peer]
                    t4 = now_l
                    a.clocks[peer].observe(t1, t2, t2, t4)
                    d = max((t4 - t1) / 2.0, 0.01)
                    e = cfg.scheduler.link_ewma
                    le.p_link = (1 - e) * le.p_link + e
                    diff = d - le.d_mean
                    le.d_mean += e * diff
                    le.d_var = (1 - e) * le.d_var + e * diff * diff
                    le.last_ack_t = now_l
                    a.last_ack[peer] = now_l
                continue
            if p.kind == "sync":
                a.clocks[p.src].observe(p.t_send_src_clock, now_l, now_l, now_l)
                a.last_ack[p.src] = now_l
                continue
            if p.kind == "ingress":
                rep = p.payload
                tg, sd = self._convert(a, rep.source, rep.tau_g)
                batch.append((rep, tg, sd, now_l))
                ingress_ids.add(rep.uuid)
                continue
            # UAV-to-UAV payloads are acknowledged with a counted envelope
            src = self.by_name[p.src]
            ack = Packet("ack", a.name, p.src, (p.pid, now_l), cfg.packets.ack, self.t, now_l)
            self.net.send(ack, float(np.linalg.norm(a.pos - src.pos)), self.t)
            a.bytes_epoch += cfg.packets.ack
            if p.kind == "point":
                rep = p.payload
                tg, sd = self._convert(a, rep.source, rep.tau_g)
                batch.append((rep, tg, sd, now_l))
            elif p.kind == "digest":
                digests.append(p.payload)
        t0 = time.perf_counter()
        for rep, tid, dec in a.bm.ingest_points(batch, now_l):
            if rep.uuid in ingress_ids and rep.uuid not in a.held:
                a.held[rep.uuid] = Held(rep, self._convert(a, rep.source, rep.tau_g)[0], tid, now_l, dec)
        for dg in digests:
            t_ref = dg.t_ref - a.clocks[dg.sender].offset
            dg_local = Digest(dg.sender, dg.x, dg.P, t_ref, dg.status, dg.n_accept, dg.ledger,
                              dg.last_update_t - a.clocks[dg.sender].offset, dg.sender_pos, dg.prioritized, dg.label)
            a.peer_last[dg.sender] = (dg_local, now_l)
            res, tid = a.bm.ingest_digest(dg_local, t_ref, now_l)
            if tid is not None:
                a.peer_digest[(dg.sender, tid)] = (dg_local, now_l)
        self._proc_ms[a.uid] += 1000 * (time.perf_counter() - t0)

    def _own_sensing(self, a: UAVAgent, now_l: float) -> None:
        reps = self.field.uav_reports(a.uid, a.pos, self.alive(), now_l)
        t0 = time.perf_counter()
        for rep, tid, dec in a.bm.ingest_points([(r, r.tau_g, 0.0, now_l) for r in reps], now_l):
            a.held[rep.uuid] = Held(rep, rep.tau_g, tid, now_l, dec)
        self._proc_ms[a.uid] += 1000 * (time.perf_counter() - t0)

    def _digest_of(self, a: UAVAgent, now_l: float) -> Optional[Digest]:
        if a.priority is None or a.priority not in a.bm.tracks:
            return None
        tr = a.bm.tracks[a.priority]
        x, P = a.bm.belief(a.priority, now_l)
        led = tuple(sorted(tr.ledger)[-self.cfg.packets.provenance_ledger:])
        return Digest(a.name, x, P, now_l, tr.status, tr.n_accept, led, tr.last_update_t, a.pos.copy(), True,
                      a.bm.label(a.priority))

    # ---------------------------------------------------------- scheduling
    def _candidates(self, a: UAVAgent, nbrs: List[UAVAgent], now_l: float) -> List[Candidate]:
        cfg, sc = self.cfg, self.cfg.scheduler
        mode = self.spec.get("value", "eig")
        L, life = cfg.packets.fixed_lag_s, cfg.packets.lifetime_s
        nbytes = cfg.packets.point_payload + cfg.packets.envelope
        lmax = cfg.packets.digest_payload + cfg.packets.envelope
        out = []
        for uuid, h in list(a.held.items()):
            age = now_l - h.tg
            if age > min(L, life):
                del a.held[uuid]
                continue
            if h.decision not in (ACCEPT, DOWNWEIGHT) or h.tid is None or h.tid not in a.bm.tracks:
                continue
            rep = h.report
            r, _, _ = a.bm._score(rep.source, rep.ctx)
            x_own, P_own = a.bm.belief(h.tid, now_l)
            R_eff = r_time(r_src(rep.R_raw, rep.xi, cfg.admission.lambda_xi), x_own[2:], 0.05) / max(r, cfg.admission.eps_r)
            for b in nbrs:
                key = (uuid, b.name)
                if key in a.sent and now_l - a.sent[key] < sc.ack_timeout_s:
                    continue
                le = a.links[b.name]
                p_del = 1.0 if mode == "no-pdel" else le.p_deliver_by(min(life, L) - age)
                t_arr = now_l + le.d_mean
                dg = a.peer_digest.get((b.name, h.tid))
                if mode == "no-receiver":
                    P_ref, t_ref, A, ploss, pcap, led, evt = P_own, now_l, 0.0, 0.0, 0.0, set(), []
                elif dg is not None:
                    d, _ = dg
                    P_ref, t_ref = d.P, d.t_ref
                    A, ploss = now_l - d.last_update_t, P_LOSS.get(d.status, 1.0)
                    pcap = 1.0 if np.linalg.norm(d.sender_pos - d.x[:2]) < 60 else 0.0
                    led, evt = set(d.ledger), [d.t_ref, d.last_update_t]
                else:
                    P_ref, t_ref, A, ploss, pcap, led, evt = BROAD_P, now_l, sc.age_norm_s, 1.0, 0.0, set(), []
                dI = info_gain(P_ref, t_ref, h.tg, t_arr, R_eff, sc.eps_I)
                if mode == "exp-aoi":
                    dI = float(np.exp(-age / 5.0)) * 0.5
                u = urgency(A, ploss, pcap, sc)
                D = duplicate_risk(uuid, h.tg, led, evt, sc)
                g = p_del * u * max(0.0, dI) - sc.lambda_dup * D - sc.lambda_E * nbytes / lmax
                if p_del < sc.p_del_min or dI < sc.dI_min_nats:
                    continue
                out.append(Candidate(rep, h.tg, b.name, nbytes, p_del, u, dI, D, g, h.tid))
        return out

    def _communicate(self, a: UAVAgent, now_l: float) -> None:
        cfg = self.cfg
        comm = self.spec["comm"]
        if comm == "none":
            return
        nbrs = self.neighbours(a)
        # deterministic digest traffic (control bytes)
        dg = self._digest_of(a, now_l) if not self.spec.get("no_digest") else None
        if dg is not None:
            for b in nbrs:
                self._send(a, b, "digest", dg, cfg.packets.digest_payload + cfg.packets.envelope)
        for b in nbrs:                              # fallback clock synchronisation
            if now_l - a.last_ack[b.name] > cfg.packets.sync_idle_s:
                self._send(a, b, "sync", None, cfg.packets.envelope)
                a.last_ack[b.name] = now_l
        nbytes = cfg.packets.point_payload + cfg.packets.envelope
        budget_kb = cfg.packets.budget_bytes_per_s / 1000.0
        if comm == "all":
            for uuid, h in list(a.held.items()):
                if h.decision == "init" or now_l - h.tg > cfg.packets.fixed_lag_s:
                    a.held.pop(uuid, None)
                    continue
                for b in nbrs:
                    if (uuid, b.name) not in a.sent:
                        self._send(a, b, "point", h.report, nbytes, uuid)
                        a.sent[(uuid, b.name)] = now_l
        elif comm in ("fifo", "fresh"):
            a.bucket = min(a.bucket + budget_kb * 1000 - a.bytes_epoch, 2 * budget_kb * 1000)
            for uuid, h in a.held.items():
                for b in nbrs:
                    if (uuid, b.name) not in a.sent:
                        a.sent[(uuid, b.name)] = -1.0
                        a.fifo.append((uuid, b.name))
            live = [(u, q) for u, q in a.fifo if u in a.held and now_l - a.held[u].tg <= cfg.packets.lifetime_s]
            if comm == "fresh":
                live.sort(key=lambda uq: -a.held[uq[0]].tg)
            sent = []
            for u, q in live:
                if a.bucket < nbytes:
                    break
                self._send(a, self.by_name[q], "point", a.held[u].report, nbytes, u)
                a.sent[(u, q)] = now_l
                a.bucket -= nbytes
                sent.append((u, q))
            sent = set(sent)
            a.fifo = [uq for uq in live if uq not in sent]
            for uuid in [u for u, h in a.held.items() if now_l - h.tg > cfg.packets.lifetime_s]:
                del a.held[uuid]
        else:   # RAI-CoS value-based scheduling
            t0 = time.perf_counter()
            if a.energy_wh <= cfg.energy.reserve * cfg.energy.battery_wh:
                cands = []
            else:
                cands = self._candidates(a, nbrs, now_l)
            cap = {b.name: max(0, cfg.network.link_capacity_bytes - a.ctrl_bytes_to[b.name]) for b in nbrs}
            mode = self.spec.get("value", "eig")
            if mode == "fixed-threshold":
                chosen = [c for c in cands if c.dI >= 0.05]
            elif self.spec.get("exact"):
                chosen = exact_select(cands, a.vq.Q, cfg.scheduler.V_dpp, cap, cfg.scheduler.quantize_bytes)
            else:
                chosen = self._greedy(cands, a.vq.Q, cfg.scheduler.V_dpp, cap)
            for c in chosen:
                self._send(a, self.by_name[c.receiver], "point", c.report, c.nbytes, c.report.uuid)
                a.sent[(c.report.uuid, c.receiver)] = now_l
            self._proc_ms[a.uid] += 1000 * (time.perf_counter() - t0)

    @staticmethod
    def _greedy(cands, Q, V, cap):
        """Deployed greedy rule of Eq. (23): positive densities (V g - Q l)/l,
        on-wire bytes l, deterministic tie-breaks, per-link capacity."""
        scored = []
        for c in cands:
            dens = (V * c.g - Q * c.nbytes) / c.nbytes
            if dens > 0:
                scored.append((-dens, -c.urgency, c.nbytes, c.tg, c.report.uuid, c.receiver, c))
        scored.sort(key=lambda s: s[:6])
        cap = dict(cap)
        out = []
        for *_, c in scored:
            if cap.get(c.receiver, 0) >= c.nbytes:
                cap[c.receiver] -= c.nbytes
                out.append(c)
        return out

    def _timeouts(self, a: UAVAgent, now_l: float) -> None:
        to = self.cfg.scheduler.ack_timeout_s
        e = self.cfg.scheduler.link_ewma
        for pid, (peer, t1, uuid) in list(a.inflight.items()):
            if now_l - t1 > to:
                le = a.links[peer]
                le.p_link = (1 - e) * le.p_link
                del a.inflight[pid]

    # ---------------------------------------------------------------- control
    def _scripted(self, a: UAVAgent, now_l: float) -> Tuple[int, Optional[int]]:
        ext = self.map.extent
        peers = [(d.sender_pos, d.x[:2]) for d, tr in a.peer_last.values() if now_l - tr < 8]
        best, best_s = None, -np.inf
        for tid, tr in a.bm.tracks.items():
            if tid in a.done or tr.n_accept < 2:
                continue
            x, P = a.bm.belief(tid, now_l)
            dist = np.linalg.norm(x[:2] - a.pos)
            s = -dist / 100.0 + (0.3 if tid == a.priority else 0.0)
            for ppos, px in peers:
                if np.linalg.norm(px - x[:2]) < 30 and np.linalg.norm(ppos - x[:2]) < dist:
                    s -= 3.0
            if s > best_s:
                best, best_s = tid, s
        if best is not None:
            x, _ = a.bm.belief(best, now_l)
            aim = x[:2] + x[2:] * 1.0
        else:
            if a.patrol is None or np.linalg.norm(a.patrol - a.pos) < 30:
                g = int(np.ceil(np.sqrt(len(self.agents))))
                a.patrol = self.sectors[a.uid] + self.rng.uniform(-0.5, 0.5, 2) * ext / g
            aim = a.patrol
        sp = self.cfg.scenario.uav_speed_mps
        cand = [a.pos + ACTIONS[k] * sp for k in range(5)]
        k = int(np.argmin([np.linalg.norm(c - aim) for c in cand]))
        return k, best

    def _safety_and_mask(self, a: UAVAgent, k: int) -> int:
        sp, ext = self.cfg.scenario.uav_speed_mps, self.map.extent
        order = [k] + [j for j in range(5) if j != k]
        for j in order:
            nxt = a.pos + ACTIONS[j] * sp
            if not (-50 <= nxt[0] <= ext + 50 and -50 <= nxt[1] <= ext + 50):
                continue
            if all(np.linalg.norm(nxt - b.pos) >= self.cfg.scenario.collision_sep_m for b in self.agents if b is not a):
                return j
        return 0

    def act(self, a: UAVAgent, now_l: float, external: Optional[Tuple[int, Optional[int]]] = None) -> None:
        if external is not None:
            k, prio = external
        elif self.actor is not None:
            k, prio = self.actor.act(self, a, now_l)
        else:
            k, prio = self._scripted(a, now_l)
        k = self._safety_and_mask(a, k)
        if prio != a.priority:
            a.hold = 0
        a.priority = prio
        new_v = ACTIONS[k] * self.cfg.scenario.uav_speed_mps
        acc = (new_v - a.vel) / self.cfg.scenario.dt
        a.vel = new_v
        ec = self.cfg.energy
        dt = self.cfg.scenario.dt
        fl = (ec.P0_w + ec.Pv * float(new_v @ new_v) + ec.Pa * float(acc @ acc)) * dt / 3600
        cp = ec.compute_w * dt / 3600
        a.flight_wh += fl
        a.compute_wh += cp
        a.energy_wh -= fl + cp
        # own-belief capture bookkeeping (no truth): hold the priority within 20 m for 5 s
        if prio is not None and prio in a.bm.tracks:
            x, _ = a.bm.belief(prio, now_l)
            if np.linalg.norm(x[:2] - a.pos) <= self.cfg.scenario.capture_radius_m:
                a.hold += 1
                if a.hold >= self.cfg.scenario.capture_hold_s:
                    a.done.add(prio)
                    a.hold = 0
            else:
                a.hold = 0

    # -------------------------------------------------------------- evaluate
    def _evaluate(self) -> None:
        sc = self.cfg.scenario
        cont_now = 0
        n_active = 0
        for j, tg in enumerate(self.targets):
            if j in self.captured or self.birth[j] > self.t:
                continue
            n_active += 1
            best = None
            for a in self.agents:
                now_l = a.local_time(self.t)
                for tid, tr in a.bm.tracks.items():
                    if a.bm.label(tid) != j or tr.status == "lost":
                        continue
                    x, P = a.bm.belief(tid, now_l)
                    key = (np.trace(P[:2, :2]), now_l - tr.last_update_t, a.uid)
                    if best is None or key < best[0]:
                        best = (key, x, P)
            ok = False
            if best is not None:
                _, x, P = best
                e = x[:2] - tg.pos
                err = float(np.linalg.norm(e))
                self.m["sq_err"].append(err ** 2)
                self.m["nees"].append(float(e @ np.linalg.solve(P[:2, :2] + 1e-9 * np.eye(2), e)))
                ok = err <= sc.capture_radius_m and np.trace(P[:2, :2]) <= 400.0
            self.m["cont"].append(ok)
            cont_now += ok
            if self.cont_state[j] and not ok:
                self.track_losses += 1
                self.gap_start[j] = self.t
            if ok and not self.cont_state[j] and j in self.gap_start:
                self.reacq_times.append(self.t - self.gap_start.pop(j))
            self.cont_state[j] = ok
        # captures: same UAV prioritised the target, active track, within 20 m, 5 s
        for a in self.agents:
            if a.priority is None or a.priority not in a.bm.tracks:
                continue
            j = a.bm.label(a.priority)
            if j < 0 or j in self.captured:
                continue
            tr = a.bm.tracks[a.priority]
            if tr.status in ("active", "reacquiring") and np.linalg.norm(a.pos - self.targets[j].pos) <= sc.capture_radius_m:
                self.cap_hold[(a.uid, j)] += 1
                if self.cap_hold[(a.uid, j)] >= sc.capture_hold_s:
                    self.captured[j] = self.t
            else:
                self.cap_hold[(a.uid, j)] = 0
        for a in self.agents:
            for b in self.agents:
                if a.uid < b.uid and np.linalg.norm(a.pos - b.pos) < 5.0:
                    self.collisions += 1
        if self.record:
            self.series.append({"t": self.t, "cont": cont_now / max(n_active, 1), "captures": len(self.captured)})

    # ------------------------------------------------------------------- step
    def step(self, external_actions: Optional[Dict[int, Tuple[int, Optional[int]]]] = None) -> float:
        cfg, sc = self.cfg, self.cfg.scenario
        dt = sc.dt
        self._proc_ms = defaultdict(float)
        for tg in self.targets:
            tg.step(dt)
        self.field.step(self.t, dt)
        for a in self.agents:
            a.pos = a.pos + a.vel * dt
            a.bytes_epoch = 0
            a.ctrl_bytes_to.clear()
        self.t += dt
        self._ingress()
        arrivals = defaultdict(list)
        for ta, p in self.net.deliver(self.t):
            arrivals[p.dst].append((ta, p))
        for a in self.agents:
            now_l = a.local_time(self.t)
            self._own_sensing(a, now_l)
            self._receive(a, arrivals.get(a.name, []), now_l)
            a.bm.maintain(now_l)
            self._timeouts(a, now_l)
        for a in self.agents:
            now_l = a.local_time(self.t)
            self._communicate(a, now_l)
            t0 = time.perf_counter()
            self.act(a, now_l, None if external_actions is None else external_actions.get(a.uid))
            self._proc_ms[a.uid] += 1000 * (time.perf_counter() - t0)
            ec = cfg.energy
            a.energy_wh -= a.bytes_epoch * ec.tx_uj_per_byte * 1e-6 / 3600
            a.vq.update(a.bytes_epoch)
        self.decision_ms.extend(self._proc_ms.values())
        before = len(self.captured)
        self._evaluate()
        tail = self.m["cont"][-max(1, sc.num_targets - before):]
        return float(np.mean(tail)) + 1.0 * (len(self.captured) - before) \
            - 0.1 * sum(a.bytes_epoch for a in self.agents) / (len(self.agents) * cfg.packets.budget_bytes_per_s)

    def run(self) -> dict:
        steps = int(self.cfg.scenario.duration_s / self.cfg.scenario.dt)
        for _ in range(steps):
            self.step()
        return self.summary()

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict:
        cfg = self.cfg
        dur = self.t
        nU = len(self.agents)
        tx = np.array([self.net.tx_bytes[a.name] for a in self.agents], float)
        rx = np.array([self.net.rx_bytes[a.name] for a in self.agents], float)
        ec = cfg.energy
        e_flight = sum(a.flight_wh for a in self.agents)
        e_comp = sum(a.compute_wh for a in self.agents)
        e_tx = tx.sum() * ec.tx_uj_per_byte * 1e-6 / 3600
        e_rx = rx.sum() * ec.rx_uj_per_byte * 1e-6 / 3600
        calib = [c for a in self.agents for c in a.bm.calib if c[1] >= 0]
        decs = [d for a in self.agents for d in a.bm.decisions if d[1] >= 0]
        stats = Counter()
        for a in self.agents:
            stats.update(a.bm.stats)
        p = np.array([c[0] for c in calib]) if calib else np.zeros(0)
        y = np.array([c[1] for c in calib]) if calib else np.zeros(0)
        harmful = [d for d in decs if d[1] == 0]
        useful = [d for d in decs if d[1] == 1]
        res = {
            "method": cfg.method, "seed": cfg.seed, "map": cfg.scenario.road_map,
            "continuity_pct": 100 * float(np.mean(self.m["cont"])) if self.m["cont"] else 0.0,
            "captures": len(self.captured),
            "rmse_m": float(np.sqrt(np.mean(self.m["sq_err"]))) if self.m["sq_err"] else float("nan"),
            "nees_pos": float(np.mean(np.clip(self.m["nees"], 0, 1e3))) if self.m["nees"] else float("nan"),
            "track_losses": self.track_losses,
            "reacq_s": float(np.mean(self.reacq_times)) if self.reacq_times else float("nan"),
            "tx_mb": float(tx.sum() / 1e6),
            "tx_kb_per_uav_s": float(tx.mean() / dur / 1000),
            "budget_violation_pct": 100 * float(np.mean(tx / dur > cfg.packets.budget_bytes_per_s)),
            "wh_per_target": float((e_flight + e_comp + e_tx + e_rx) / cfg.scenario.num_targets),
            "decision_p95_ms": float(np.percentile(self.decision_ms, 95)) if self.decision_ms else 0.0,
            "brier": float(np.mean((p - y) ** 2)) if len(p) else float("nan"),
            "ece": _ece(p, y), "auroc": _auroc(p, y),
            "harmful_accept_pct": 100 * float(np.mean([d[0] in (ACCEPT, DOWNWEIGHT) for d in harmful])) if harmful else float("nan"),
            "useful_reject_pct": 100 * float(np.mean([d[0] == REJECT for d in useful])) if useful else float("nan"),
            "replays": int(stats["replay"]), "duplicates": int(stats["duplicate"] + stats["digest_duplicate"]),
            "audit_only": int(sum(a.bm.audit_only for a in self.agents)),
            "collisions": self.collisions,
        }
        if self.record:
            res["series"] = self.series
        return res


def _ece(p, y, bins: int = 10) -> float:
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _auroc(p, y) -> float:
    if len(p) == 0 or y.min() == y.max():
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_episode(cfg: ExperimentConfig, record: bool = False) -> dict:
    return Episode(cfg, record).run()
