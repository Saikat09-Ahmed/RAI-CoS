"""Expected information gain at arrival and virtual-queue scheduling
(Sec. 4.7, 4.8; Eqs. 20-23).

For logical point report m held by sender i and receiver q:

  dI   = 1/2 [ ln det(P_q^-(tau_a) + eps I) - ln det(P_q^+(tau_a; m) + eps I) ]       (20)
  u    = 1 + (2.5 - 1)(0.4 a_age + 0.4 a_loss + 0.2 a_cap),  a_age = min{1, A/12}
  D    = w_D O + (1 - w_D) Pi_time,  O = Jaccard(ancestry, represented ancestry)
  g    = p_del * u * max{0, dI} - lambda_dup D - lambda_E E_tx                       (21)
  Q   <- max{0, Q + B_t - B_bar}                                                     (22)
  x*   = argmax sum x (V g - Q l)   s.t. p_del >= 0.20, dI >= 0.0075 nats, capacity   (23)

P_q^- is predicted only from the latest digest that actually arrived from q,
inflated for digest age; P_q^+ applies a generation-time correction with the
sender's own conservative R_eff. The deployed rule is greedy by positive
density (V g - Q l)/l with deterministic tie-breaks; ``exact_select`` is the
16-byte-quantised exact solver used as a diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .imm import IMMState, joseph_update, predict

BROAD_P = np.diag([150.0 ** 2, 150.0 ** 2, 15.0 ** 2, 15.0 ** 2])


@dataclass
class Candidate:
    report: object
    tg: float              # converted generation time (sender clock)
    receiver: str
    nbytes: int
    p_del: float
    urgency: float
    dI: float
    dup: float
    g: float
    tid: int


def _cv_pred(P: np.ndarray, dt: float, sa: float = 0.8) -> np.ndarray:
    if dt <= 0:
        return P
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = sa ** 2
    Qb = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Qb
    Q[np.ix_([1, 3], [1, 3])] = Qb
    return F @ P @ F.T + Q


def info_gain(P_ref: np.ndarray, t_ref: float, tg: float, t_arr: float, R_eff: np.ndarray, eps: float) -> float:
    """Eq. 20 with constant-velocity covariance propagation."""
    P_minus = _cv_pred(P_ref, t_arr - t_ref)
    P_g = _cv_pred(P_ref, max(tg - t_ref, 0.0))
    S = P_g[:2, :2] + R_eff
    K = P_g[:, :2] @ np.linalg.inv(S)
    A = np.eye(4)
    A[:, :2] -= K
    P_post = A @ P_g @ A.T + K @ R_eff @ K.T
    P_plus = _cv_pred(P_post, t_arr - max(tg, t_ref))
    I = np.eye(4) * eps
    return 0.5 * (np.linalg.slogdet(P_minus + I)[1] - np.linalg.slogdet(P_plus + I)[1])


def urgency(age: float, p_loss: float, p_cap: float, cfg) -> float:
    a_age = min(1.0, age / cfg.age_norm_s)
    return cfg.urgency_min + (cfg.urgency_max - cfg.urgency_min) * (
        cfg.w_age * a_age + cfg.w_loss * p_loss + cfg.w_cap * p_cap)


def duplicate_risk(uuid: int, tg: float, ledger: set, ev_times: List[float], cfg) -> float:
    anc = {uuid}
    O = len(anc & ledger) / max(len(anc | ledger), 1)
    Pi = max((np.exp(-abs(tg - h) / cfg.tau_D_s) for h in ev_times), default=0.0)
    return cfg.w_D * O + (1 - cfg.w_D) * Pi


class VirtualQueue:
    def __init__(self, budget: float):
        self.B = budget
        self.Q = 0.0

    def update(self, bytes_epoch: float) -> None:
        self.Q = max(0.0, self.Q + bytes_epoch - self.B)


def greedy_select(cands: List[Candidate], Q: float, V: float, capacity: Dict[str, int]) -> List[Candidate]:
    scored = []
    for c in cands:
        dens = (V * c.g - Q * c.nbytes) / c.nbytes
        if dens > 0:
            scored.append((-dens, -c.urgency, c.nbytes, c.tg, c.report.uuid, c.receiver, c))
    scored.sort(key=lambda s: s[:6])
    cap = dict(capacity)
    out = []
    for *_, c in scored:
        if cap.get(c.receiver, 0) >= c.nbytes:
            cap[c.receiver] -= c.nbytes
            out.append(c)
    return out


def exact_select(cands: List[Candidate], Q: float, V: float, capacity: Dict[str, int], unit: int = 16):
    """Per-receiver 0/1 knapsack on 16-byte-quantised costs (objective is
    separable across receivers, constraints are per link)."""
    out = []
    by_r: Dict[str, List[Candidate]] = {}
    for c in cands:
        if V * c.g - Q * c.nbytes > 0:
            by_r.setdefault(c.receiver, []).append(c)
    for r, cs in by_r.items():
        W = capacity.get(r, 0) // unit
        best = np.zeros(W + 1)
        keep = np.zeros((len(cs), W + 1), dtype=bool)
        for k, c in enumerate(cs):
            w = int(np.ceil(c.nbytes / unit))
            val = V * c.g - Q * c.nbytes
            for cap in range(W, w - 1, -1):
                if best[cap - w] + val > best[cap]:
                    best[cap] = best[cap - w] + val
                    keep[k, cap] = True
        cap = W
        for k in range(len(cs) - 1, -1, -1):
            if keep[k, cap]:
                out.append(cs[k])
                cap -= int(np.ceil(cs[k].nbytes / unit))
    return out
