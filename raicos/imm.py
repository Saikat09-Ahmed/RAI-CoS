"""Four-mode IMM filter (Eq. 3, Eq. 18).

Modes: constant velocity, slow/stop (velocity decays), acceleration
(high process noise) and coordinated turn. The coordinated-turn mode uses
the turn-rate estimate carried by the track as a known input, which makes
its transition linear in the state; with a 2-D position observation every
mode update is a (Joseph-form) Kalman update, i.e. the EKF reduces to its
exact linear form for this measurement model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
NM = 4


def _q_block(dt: float, s2: float) -> np.ndarray:
    return s2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])


def transition(mode: int, dt: float, omega: float, sigma_a) -> Tuple[np.ndarray, np.ndarray]:
    s2 = sigma_a[mode] ** 2
    F = np.eye(4)
    if mode == 1:                                   # slow/stop: v <- a v
        tau = 2.0
        a = np.exp(-dt / tau)
        F[0, 2] = F[1, 3] = tau * (1 - a)
        F[2, 2] = F[3, 3] = a
    elif mode == 3 and abs(omega) > 1e-4:           # coordinated turn, known rate
        w = omega
        sw, cw = np.sin(w * dt), np.cos(w * dt)
        F = np.array([[1, 0, sw / w, -(1 - cw) / w],
                      [0, 1, (1 - cw) / w, sw / w],
                      [0, 0, cw, -sw],
                      [0, 0, sw, cw]])
    else:
        F[0, 2] = F[1, 3] = dt
    q11, q12, q22 = s2 * dt ** 4 / 4, s2 * dt ** 3 / 2, s2 * dt ** 2
    Q = np.array([[q11, 0, q12, 0], [0, q11, 0, q12], [q12, 0, q22, 0], [0, q12, 0, q22]])
    return F, Q


_MTM: dict = {}


def _mtm(dt: float, p_switch: float) -> np.ndarray:
    k = (dt, p_switch)
    if k not in _MTM:
        _MTM[k] = mode_transition_matrix(dt, p_switch)
    return _MTM[k]


def mode_transition_matrix(dt: float, p_switch: float) -> np.ndarray:
    p = 1 - (1 - p_switch) ** max(dt, 0.0)
    T = np.full((NM, NM), p / (NM - 1))
    np.fill_diagonal(T, 1 - p)
    return T


@dataclass
class IMMState:
    xs: np.ndarray                      # (4 modes, 4)
    Ps: np.ndarray                      # (4, 4, 4)
    mu: np.ndarray                      # (4,)
    omega: float = 0.0

    def copy(self) -> "IMMState":
        return IMMState(self.xs.copy(), self.Ps.copy(), self.mu.copy(), self.omega)

    @property
    def x(self) -> np.ndarray:
        return self.mu @ self.xs

    @property
    def P(self) -> np.ndarray:
        d = self.xs - self.mu @ self.xs
        P = np.einsum("m,mab->ab", self.mu, self.Ps) + np.einsum("m,ma,mb->ab", self.mu, d, d)
        return 0.5 * (P + P.T)

    @staticmethod
    def init(z: np.ndarray, R: np.ndarray, v0_sd: float = 10.0) -> "IMMState":
        x = np.r_[z, 0.0, 0.0]
        P = np.diag([R[0, 0], R[1, 1], v0_sd ** 2, v0_sd ** 2])
        return IMMState(np.tile(x, (NM, 1)), np.tile(P, (NM, 1, 1)), np.full(NM, 1 / NM))

    def set_moments(self, x: np.ndarray, P: np.ndarray) -> None:
        """Replace every mode's moments (after CI or initialisation)."""
        self.xs[:] = x
        self.Ps[:] = P


_FQ_CACHE: dict = {}


def _fq_stack(dt: float, omega: float, sigma_a):
    dt, omega = round(dt, 2), round(omega, 2)          # 10 ms / 0.01 rad/s quantisation for caching
    key = (dt, omega, sigma_a)
    v = _FQ_CACHE.get(key)
    if v is None:
        F = np.stack([transition(m, dt, omega, sigma_a)[0] for m in range(NM)])
        Q = np.stack([transition(m, dt, omega, sigma_a)[1] for m in range(NM)])
        v = (F, Q)
        if len(_FQ_CACHE) > 20000:
            _FQ_CACHE.clear()
        _FQ_CACHE[key] = v
    return v


def predict(st: IMMState, dt: float, sigma_a, p_switch: float) -> IMMState:
    """IMM interaction + mode-conditioned prediction (vectorised over modes)."""
    if dt <= 5e-3:
        return st.copy()
    T = _mtm(round(dt, 2), p_switch)
    c = np.maximum(T.T @ st.mu, 1e-12)
    w = (T * st.mu[:, None]) / c[None, :]                   # w[i, j] = P(i | j)
    x0 = w.T @ st.xs                                        # (j, 4)
    d = st.xs[None, :, :] - x0[:, None, :]                  # (j, i, 4)
    P0 = np.einsum("ij,iab->jab", w, st.Ps) + np.einsum("ij,jia,jib->jab", w, d, d)
    F, Q = _fq_stack(dt, st.omega, sigma_a)
    xs = np.einsum("jab,jb->ja", F, x0)
    Ps = np.einsum("jab,jbc,jdc->jad", F, P0, F) + Q
    return IMMState(xs, Ps, c / c.sum(), st.omega)


def joseph_update(st: IMMState, z: np.ndarray, R: np.ndarray) -> Tuple[IMMState, float]:
    """Mode-conditioned Joseph updates and mode-probability update (Eq. 18)."""
    Ps = st.Ps
    S = Ps[:, :2, :2] + R                                    # H P H^T + R
    Si = np.linalg.inv(S)
    K = np.einsum("mab,mbc->mac", Ps[:, :, :2], Si)          # P H^T S^-1
    nu = z[None, :] - st.xs[:, :2]
    xs = st.xs + np.einsum("mab,mb->ma", K, nu)
    A = np.broadcast_to(np.eye(4), (NM, 4, 4)).copy()
    A[:, :, :2] -= K                                         # I - K H
    Pn = np.einsum("mab,mbc,mdc->mad", A, Ps, A) + np.einsum("mab,bc,mdc->mad", K, R, K)
    Pn = 0.5 * (Pn + np.transpose(Pn, (0, 2, 1)))
    d2 = np.einsum("ma,mab,mb->m", nu, Si, nu)
    detS = np.maximum(np.linalg.det(S), 1e-12)
    lik = np.exp(-0.5 * d2) / (2 * np.pi * np.sqrt(detS))
    mu = st.mu * np.maximum(lik, 1e-300)
    tot = mu.sum()
    mu = mu / tot if tot > 0 and np.isfinite(tot) else st.mu.copy()
    out = IMMState(xs, Pn, mu, st.omega)
    v0, v1 = st.x[2:], out.x[2:]
    if np.linalg.norm(v0) > 2 and np.linalg.norm(v1) > 2:
        dpsi = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
        out.omega = float(np.clip(0.7 * st.omega + 0.3 * dpsi, -0.35, 0.35))
    return out, float(tot)


def innovation(st: IMMState, z: np.ndarray, R: np.ndarray):
    x, P = st.x, st.P
    nu = z - H @ x
    S = H @ P @ H.T + R
    return nu, S, float(nu @ np.linalg.solve(S, nu))


def ci_fuse(x1, P1, x2, P2, grid: int = 21, tol: float = 1e-3, reg: float = 1e-6):
    """Covariance intersection minimising log det P_CI: 21-point coarse search
    followed by golden-section refinement to `tol` (Eq. 19)."""
    I = np.eye(len(x1)) * reg
    A, B = np.linalg.inv(P1 + I), np.linalg.inv(P2 + I)

    def f(w):
        return np.linalg.slogdet(w * A + (1 - w) * B)[1] * -1.0     # log det P_CI

    ws = np.linspace(0, 1, grid)
    vals = [f(w) for w in ws]
    k = int(np.argmin(vals))
    lo, hi = ws[max(k - 1, 0)], ws[min(k + 1, grid - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    while hi - lo > tol:
        if f(c) < f(d):
            hi = d
        else:
            lo = c
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    w = 0.5 * (lo + hi)
    P = np.linalg.inv(w * A + (1 - w) * B + I)
    x = P @ (w * A @ x1 + (1 - w) * B @ x2)
    return x, 0.5 * (P + P.T), w


def info_fuse(x1, P1, x2, P2, reg: float = 1e-6):
    """Naive (independence-assuming) fusion used by the no-CI ablation."""
    I = np.eye(len(x1)) * reg
    A, B = np.linalg.inv(P1 + I), np.linalg.inv(P2 + I)
    P = np.linalg.inv(A + B)
    return P @ (A @ x1 + B @ x2), 0.5 * (P + P.T)
