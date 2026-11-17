"""Per-UAV belief management (Sec. 3.2, 4.4-4.6, 5.4; Algorithm 1).

Executable order for every delivered batch:
  exact-duplicate suppression (UUID / ancestry ledger)
  -> clock conversion to the receiver's time and live-lag check (L = 15 s)
  -> common association front end: R_assoc, 99.5% gate, Hungarian assignment
     on negative log-likelihood per source scan, deferral of near ties
  -> reliability score + report-consistency admission (ACCEPT/DOWNWEIGHT/REJECT)
  -> generation-time Joseph update inserted in a 15 s ring buffer and
     chronological replay of later typed events (decisions stay fixed;
     CI weights are re-optimised during replay)
Track summaries (digests) are aligned at their reference time and fused by
covariance intersection; digests whose ancestry is already represented are
exact duplicates.

Track lifecycle: two associated reports within 5 s initiate a track;
uncertain after 6 s, lost after 12 s, deleted after 30 s without an accepted
update; at most 12 tracks.

Reliability confirmation: a report is later compared with provenance-
independent evidence (UAV or monitor point, or a low-covariance local
estimate), y = 1[e^T Sigma^-1 e <= chi2_{2,0.95}], weighted by independence
and geometry, and fed to the receiver's Beta posterior.
"""
from __future__ import annotations

import bisect
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .admission import ACCEPT, DOWNWEIGHT, GATED, REJECT, admit, psd, r_src, r_time
from .imm import IMMState, ci_fuse, info_fuse, joseph_update, predict
from .reliability import ReliabilityModel, source_type

FIXED_TYPE_WEIGHTS = {"uav": 0.95, "monitor": 0.85, "mobile": 0.60}
_tid = itertools.count(1)


def _vote(labels: Counter, j: int, w: float = 1.0, decay: float = 0.85) -> None:
    """Evaluator-only label of a track: exponentially decayed votes of the
    hidden truth ids behind its accepted evidence."""
    for k in list(labels):
        labels[k] *= decay
    labels[j] += w


@dataclass
class Event:
    t: float
    seq: int
    kind: str                 # point | digest
    z: np.ndarray             # point: observation; digest: mean
    R: np.ndarray             # point: R_eff; digest: covariance
    source: str = ""


@dataclass
class Track:
    tid: int
    state: IMMState
    t: float
    base: IMMState
    base_t: float
    events: List[Event] = field(default_factory=list)
    snaps: List[IMMState] = field(default_factory=list)       # snaps[k]: after events[k]
    last_update_t: float = 0.0
    created_t: float = 0.0
    status: str = "active"
    reacq_until: float = -1.0
    ledger: set = field(default_factory=set)
    n_accept: int = 0
    rel_sum: float = 0.0
    labels: Counter = field(default_factory=Counter)          # evaluator only (decayed)
    src_updates: Deque[Tuple[float, str]] = field(default_factory=lambda: deque(maxlen=64))
    origin: str = "points"


@dataclass
class Pending:
    source: str
    ctx: tuple
    z: np.ndarray
    R_time: np.ndarray
    t: float
    tid: int
    t_recv: float


@dataclass
class Flags:
    reliability: bool = True
    consistency: bool = True
    generation_time: bool = True
    provenance: bool = True
    ci: bool = True
    lower_bound: bool = True
    forgetting: bool = True
    freshness_tau: Optional[float] = None       # AoI-DS freshness weighting


class BeliefManager:
    def __init__(self, uid: int, cfg, flags: Flags, rng: np.random.Generator):
        self.uid, self.cfg, self.flags, self.rng = uid, cfg, flags, rng
        tc = cfg.tracker
        self.sa, self.p_sw = tc.mode_sigma_a, tc.mode_switch
        self.tracks: Dict[int, Track] = {}
        self.rel = ReliabilityModel(cfg.reliability, flags.lower_bound, flags.forgetting)
        self.seen: set = set()
        self.tentative: List[Tuple[float, np.ndarray, np.ndarray, object]] = []
        self.deferred: List[Tuple[object, float, float]] = []
        self.deferred_once: set = set()
        self.pending: Deque[Pending] = deque()
        self.audit_only = 0
        self._seq = itertools.count(1)
        self.stats = Counter()
        self.calib: List[Tuple[float, int]] = []                # (pre-confirmation mean, A_k)
        self.decisions: List[Tuple[str, int]] = []              # (decision, A_k)

    # ----------------------------------------------------------------- state
    def state_at(self, tr: Track, t: float) -> IMMState:
        if t >= tr.t or not tr.events:
            return predict(tr.state, t - tr.t, self.sa, self.p_sw)
        times = [e.t for e in tr.events]
        k = bisect.bisect_right(times, t)
        if k == 0:
            return predict(tr.base, t - tr.base_t, self.sa, self.p_sw)
        return predict(tr.snaps[k - 1], t - tr.events[k - 1].t, self.sa, self.p_sw)

    def moments_at(self, tr: Track, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Collapsed moments propagated with a constant-velocity model (used by
        the common association front end; the IMM is used for estimation)."""
        if t >= tr.t or not tr.events:
            base, t0 = (tr.state, tr.t) if t >= tr.t else (tr.base, tr.base_t)
        else:
            k = bisect.bisect_right([e.t for e in tr.events], t)
            base, t0 = (tr.base, tr.base_t) if k == 0 else (tr.snaps[k - 1], tr.events[k - 1].t)
        x, P = base.x, base.P
        dt = t - t0
        if abs(dt) > 1e-6:                     # forward prediction or retrodiction
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            a = abs(dt)
            Q = np.array([[a ** 4 / 4, 0, a ** 3 / 2, 0], [0, a ** 4 / 4, 0, a ** 3 / 2],
                          [a ** 3 / 2, 0, a ** 2, 0], [0, a ** 3 / 2, 0, a ** 2]])
            x, P = F @ x, F @ P @ F.T + Q
        return x, P

    def _apply(self, st: IMMState, t_prev: float, ev: Event) -> IMMState:
        st = predict(st, ev.t - t_prev, self.sa, self.p_sw)
        if ev.kind == "point":
            st, _ = joseph_update(st, ev.z, ev.R)
            return st
        x, P = st.x, st.P
        if self.flags.ci:
            xf, Pf, _ = ci_fuse(x, P, ev.z, ev.R, self.cfg.tracker.ci_grid, self.cfg.tracker.ci_tol,
                                self.cfg.tracker.reg)
        else:
            xf, Pf = info_fuse(x, P, ev.z, ev.R, self.cfg.tracker.reg)
        st = st.copy()
        st.set_moments(xf, Pf)
        return st

    def _insert(self, tr: Track, ev: Event) -> None:
        """Insert at physical time and replay every later typed event."""
        if ev.t < tr.base_t:                   # older than the buffer base: retrodict the base
            xb, Pb = self.moments_at(tr, ev.t)
            base = tr.base.copy()
            base.set_moments(xb, Pb)
            tr.base, tr.base_t = base, ev.t
        times = [(e.t, e.seq) for e in tr.events]
        k = bisect.bisect(times, (ev.t, ev.seq))
        tr.events.insert(k, ev)
        st, t_prev = (tr.base, tr.base_t) if k == 0 else (tr.snaps[k - 1], tr.events[k - 1].t)
        snaps = tr.snaps[:k]
        if k < len(tr.events) - 1:
            self.stats["replay"] += 1
        for e in tr.events[k:]:
            st = self._apply(st, t_prev, e)
            t_prev = e.t
            snaps.append(st)
        tr.snaps = snaps
        tr.state, tr.t = st, t_prev

    def _trim(self, tr: Track, now: float) -> None:
        horizon = now - self.cfg.tracker.oosm_buffer_s
        while tr.events and tr.events[0].t < horizon:
            tr.base, tr.base_t = tr.snaps[0], tr.events[0].t
            tr.events.pop(0)
            tr.snaps.pop(0)

    def _new_track(self, st: IMMState, t: float, now: float) -> Optional[Track]:
        if len(self.tracks) >= self.cfg.tracker.max_tracks:
            return None
        tid = next(_tid)
        tr = Track(tid, st, t, st.copy(), t, last_update_t=t, created_t=now)
        self.tracks[tid] = tr
        return tr

    # ----------------------------------------------------------- point reports
    def _r_assoc(self, rep, vel, sig_tau) -> np.ndarray:
        return r_time(psd(rep.R_raw), vel, sig_tau)

    def _score(self, source, ctx):
        if not self.flags.reliability:
            w = FIXED_TYPE_WEIGHTS[source_type(source)]
            return w, w, False
        return self.rel.score(source, ctx)

    def ingest_points(self, batch: List[Tuple[object, float, float, float]], now: float) -> List[Tuple[object, int, str]]:
        """batch: (report, converted generation time, timing sd, arrival time).
        Returns (report, track id, decision) for every processed report."""
        L = self.cfg.packets.fixed_lag_s
        live = []
        for rep, tg, sd, ta in batch + [(r, tg, sd, ta) for r, tg, sd, ta in self.deferred_batch(now)]:
            if self.flags.provenance and (rep.uuid in self.seen or any(rep.uuid in t.ledger for t in self.tracks.values())):
                self.stats["duplicate"] += 1
                continue
            if now - tg > L:
                self.audit_only += 1
                continue
            live.append((rep, tg, sd, ta))
        live.sort(key=lambda r: r[1])
        results = []
        scans: Dict[tuple, list] = {}
        for item in live:
            scans.setdefault((item[0].source, round(item[1])), []).append(item)
        for _, scan in sorted(scans.items(), key=lambda kv: kv[1][0][1]):
            tids = list(self.tracks)
            pairs: List[Tuple[object, Optional[int]]] = []
            if tids:
                C = np.full((len(scan), len(tids)), 1e6)
                for a, (rep, tg, sd, ta) in enumerate(scan):
                    for b, tid in enumerate(tids):
                        xm, Pm = self.moments_at(self.tracks[tid], tg)
                        R = self._r_assoc(rep, xm[2:], sd)
                        S = Pm[:2, :2] + R
                        nu = rep.z - xm[:2]
                        d2 = float(nu @ np.linalg.solve(S, nu))
                        if d2 <= self.cfg.admission.gate_chi2:
                            C[a, b] = 0.5 * d2 + 0.5 * np.log(max(np.linalg.det(2 * np.pi * S), 1e-12))
                rows, cols = linear_sum_assignment(C)
                assigned = {r: c for r, c in zip(rows, cols) if C[r, c] < 1e6}
                for a, item in enumerate(scan):
                    if a in assigned:
                        b = assigned[a]
                        row = np.sort(C[a])
                        if (len(row) > 1 and row[1] < 1e6 and row[1] - row[0] < self.cfg.tracker.ambiguity_margin
                                and item[0].uuid not in self.deferred_once):
                            self.deferred_once.add(item[0].uuid)
                            self.deferred.append((item, now, item[1]))      # near tie: defer
                            self.stats["deferred"] += 1
                            continue
                        pairs.append((item, tids[b]))
                    else:
                        pairs.append((item, None))
            else:
                pairs = [(item, None) for item in scan]
            for item, tid in pairs:
                if tid is None:
                    self._initiate(item, now)
                    continue
                dec = self._admit_and_update(item, self.tracks[tid], now)
                results.append((item[0], tid, dec))
        return results

    def deferred_batch(self, now: float):
        out, keep = [], []
        for item, t_def, tg in self.deferred:
            if now - t_def >= 1.0:
                out.append(item)
            else:
                keep.append((item, t_def, tg))
        self.deferred = keep
        return out

    def _initiate(self, item, now: float) -> None:
        rep, tg, sd, ta = item
        self.seen.add(rep.uuid)
        R = psd(rep.R_raw * 1.5)
        for tr in self.tracks.values():                 # near an existing track: not a new target
            xm, Pm = self.moments_at(tr, tg)
            S = Pm[:2, :2] + R
            nu = rep.z - xm[:2]
            if nu @ np.linalg.solve(S, nu) <= 4 * self.cfg.admission.gate_chi2 or np.linalg.norm(nu) < 25.0:
                return
        win = self.cfg.tracker.init_window_s
        self.tentative = [x for x in self.tentative if now - x[0] <= self.cfg.packets.fixed_lag_s]
        for k, (t0, z0, R0, rep0) in enumerate(self.tentative):
            dt = tg - t0
            if 0.2 <= abs(dt) <= win and np.linalg.norm(rep.z - z0) <= 16.0 * abs(dt) + 3 * np.sqrt(R[0, 0] + R0[0, 0]):
                (ta_, za, Ra), (tb, zb, Rb) = sorted([(t0, z0, R0), (tg, rep.z, R)], key=lambda u: u[0])
                v = (zb - za) / (tb - ta_)
                st = IMMState.init(zb, Rb, v0_sd=6.0)
                st.xs[:, 2:] = v
                tr = self._new_track(st, tb, now)
                if tr is not None:
                    tr.ledger |= {rep.uuid, rep0.uuid}
                    _vote(tr.labels, rep0.truth_target)
                    _vote(tr.labels, rep.truth_target)
                    tr.last_update_t = tb
                    tr.n_accept = 2
                del self.tentative[k]
                return
        self.tentative.append((tg, rep.z.copy(), R, rep))

    def _admit_and_update(self, item, tr: Track, now: float) -> str:
        rep, tg, sd, ta = item
        fl = self.flags
        self.seen.add(rep.uuid)
        r, mean, cold = self._score(rep.source, rep.ctx)
        t_ev = tg if fl.generation_time else now
        st = self.state_at(tr, t_ev)
        Rs = r_src(rep.R_raw, rep.xi, self.cfg.admission.lambda_xi)
        Rt = r_time(Rs, st.x[2:], sd)
        if fl.freshness_tau:
            Rt = Rt * np.exp(max(now - tg, 0) / fl.freshness_tau)
        S = st.P[:2, :2] + Rt
        nu = rep.z - st.x[:2]
        dec, eta, d2 = admit(nu, S, r, self.cfg.admission, cold, fl.consistency)
        self.calib.append((mean, rep.truth_event))
        self.decisions.append((dec, rep.truth_event))
        self.pending.append(Pending(rep.source, rep.ctx, rep.z.copy(), Rt, tg, tr.tid, now))
        if dec in (ACCEPT, DOWNWEIGHT):
            R_eff = Rt / max(eta, self.cfg.admission.eps_r)
            self._insert(tr, Event(t_ev, next(self._seq), "point", rep.z.copy(), R_eff, rep.source))
            tr.ledger.add(rep.uuid)
            _vote(tr.labels, rep.truth_target)
            tr.n_accept += 1
            tr.rel_sum += r
            if tr.status in ("lost", "uncertain"):
                tr.status, tr.reacq_until = "reacquiring", now + 5.0
            tr.last_update_t = max(tr.last_update_t, tg)
            tr.src_updates.append((tg, rep.source))
            if source_type(rep.source) in ("uav", "monitor"):
                self._confirm_from_point(tr, rep, tg, Rt, now)
        self.stats[dec] += 1
        return dec

    # ---------------------------------------------------------------- digests
    def ingest_digest(self, dg, t_ref: float, now: float):
        """Returns (result, matched local track id or None)."""
        if now - t_ref > self.cfg.packets.fixed_lag_s:
            self.audit_only += 1
            return "audit", None
        best, best_d2 = None, self.cfg.admission.gate_chi2
        for tid, tr in self.tracks.items():
            xm, Pm = self.moments_at(tr, t_ref)
            S = Pm[:2, :2] + dg.P[:2, :2]
            nu = dg.x[:2] - xm[:2]
            d2 = float(nu @ np.linalg.solve(S, nu))
            if d2 < best_d2:
                best, best_d2 = tid, d2
        if best is None:
            near = any(np.linalg.norm(self.moments_at(tr, t_ref)[0][:2] - dg.x[:2]) < 40.0 for tr in self.tracks.values())
            if dg.status == "active" and dg.n_accept >= 3 and not near:
                st = IMMState.init(dg.x[:2], dg.P[:2, :2])
                st.set_moments(dg.x, dg.P)
                tr = self._new_track(st, t_ref, now)
                if tr is not None:
                    tr.origin = "digest"
                    tr.ledger |= set(dg.ledger)
                    _vote(tr.labels, dg.label)
                    tr.last_update_t = t_ref
                    tr.n_accept = 1
                    return "init", tr.tid
            return "unmatched", None
        tr = self.tracks[best]
        anc = set(dg.ledger)
        if self.flags.provenance and anc and anc <= tr.ledger:
            self.stats["digest_duplicate"] += 1
            return "duplicate", best
        t_ev = t_ref if self.flags.generation_time else now
        self._insert(tr, Event(t_ev, next(self._seq), "digest", dg.x.copy(), dg.P.copy(), dg.sender))
        tr.ledger |= anc
        _vote(tr.labels, dg.label, 0.3)
        if dg.last_update_t > tr.last_update_t:
            tr.last_update_t = min(dg.last_update_t, now)
        self.stats["digest_fused"] += 1
        return "fused", best

    # ---------------------------------------------------------- confirmations
    def _confirm(self, p: Pending, z_conf, R_conf, vel, Pv, t_conf, w) -> None:
        dt = t_conf - p.t
        z_tilde = p.z + vel * dt
        Sigma = p.R_time + R_conf + Pv * dt * dt + np.eye(2) * (0.5 * dt) ** 2
        e = z_tilde - z_conf
        y = int(e @ np.linalg.solve(Sigma, e) <= self.cfg.reliability.confirm_chi2)
        if self.flags.reliability:
            self.rel.update(p.source, p.ctx, y, w, t_conf)
        self.stats["confirmations"] += 1

    def _confirm_from_point(self, tr: Track, rep, tg, Rt, now) -> None:
        st = tr.state
        keep = deque()
        for p in self.pending:
            if p.tid == tr.tid and p.source != rep.source and 0 < tg - p.t <= 10.0:
                w_ind = 1.0 if source_type(rep.source) == "uav" else 0.8
                w_geom = float(np.exp(-(tg - p.t) / 10.0))
                self._confirm(p, rep.z, Rt, st.x[2:], st.P[2:, 2:], tg, w_ind * w_geom)
            else:
                keep.append(p)
        self.pending = keep

    def maintain(self, now: float) -> None:
        tc, rc = self.cfg.tracker, self.cfg.reliability
        # low-covariance local estimate route + retention bounds
        keep = deque()
        for p in self.pending:
            tr = self.tracks.get(p.tid)
            if tr is None or now - p.t_recv > rc.retention_s:
                continue
            if tr is not None and now - p.t >= 3.0:
                P = tr.state.P
                indep = sum(1 for t, s in tr.src_updates if t > p.t and s != p.source)
                if np.trace(P[:2, :2]) < 30.0 and indep >= 2:
                    dt = tr.t - p.t
                    zb = tr.state.x[:2] - tr.state.x[2:] * dt
                    self._confirm(p, zb, P[:2, :2], np.zeros(2), np.zeros((2, 2)), p.t, 0.6)
                    continue
            keep.append(p)
        while len(keep) > rc.retention_records:
            keep.popleft()
        self.pending = keep
        tids = list(self.tracks)
        for i, a in enumerate(tids):                    # merge tracks that describe one target
            for b in tids[i + 1:]:
                if a not in self.tracks or b not in self.tracks:
                    continue
                ta_, tb_ = self.tracks[a], self.tracks[b]
                xa, Pa = self.belief(a, now)
                xb, Pb = self.belief(b, now)
                nu = xa[:2] - xb[:2]
                if np.linalg.norm(nu) < 15.0 and nu @ np.linalg.solve(Pa[:2, :2] + Pb[:2, :2], nu) < 4.0:
                    keep_, drop = (a, b) if (ta_.n_accept, -a) >= (tb_.n_accept, -b) else (b, a)
                    k, d = self.tracks[keep_], self.tracks[drop]
                    k.ledger |= d.ledger
                    for lab, v in d.labels.items():
                        k.labels[lab] += 0.5 * v
                    k.n_accept += d.n_accept
                    del self.tracks[drop]
                    self.stats["merged"] += 1
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            self._trim(tr, now)
            gap = now - tr.last_update_t
            if gap >= tc.delete_after_s:
                del self.tracks[tid]
                continue
            if tr.status == "reacquiring" and now < tr.reacq_until:
                continue
            tr.status = "active" if gap < tc.uncertain_after_s else ("uncertain" if gap < tc.lost_after_s else "lost")

    # ------------------------------------------------------------------ views
    def belief(self, tid: int, now: float) -> Tuple[np.ndarray, np.ndarray]:
        tr = self.tracks[tid]
        st = predict(tr.state, now - tr.t, self.sa, self.p_sw)
        return st.x, st.P

    def label(self, tid: int) -> int:
        """Evaluator-only: majority hidden target label of the track's evidence."""
        lab = self.tracks[tid].labels
        lab = Counter({k: v for k, v in lab.items() if k >= 0})
        return lab.most_common(1)[0][0] if lab else -1
