"""Point-report covariance pipeline and report-consistency admission
(Sec. 3.5, 4.2, 4.4; Eqs. 11, 12, 15).

R_src   = PSD[ R_raw * exp(lambda_xi (1 - xi)) ]              lambda_xi = 1.5
R_time  = PSD[ R_src + J sigma_tau^2 J^T ]                    timing uncertainty
R_assoc = PSD[ R_phys + J sigma_tau^2 J^T ]                   common association front end
eta     = r N(nu;0,S) / ( r N(nu;0,S) + (1 - r) c_out )      inside the 99.5% gate
R_eff   = R_time / max(eta, eps_r)

Decisions: REJECT if eta <= 0.15, DOWNWEIGHT for 0.15 < eta < 0.65,
ACCEPT for eta >= 0.65; inflation beyond 20 R_time is rejected.
"""
from __future__ import annotations

from math import gamma, pi

import numpy as np

ACCEPT, DOWNWEIGHT, REJECT, GATED = "ACCEPT", "DOWNWEIGHT", "REJECT", "GATED"


def psd(M: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    M = 0.5 * (M + M.T)
    if M.shape == (2, 2) and M[0, 0] > floor and M[0, 0] * M[1, 1] - M[0, 1] ** 2 > floor * M[0, 0]:
        return M                                  # already positive definite
    w, V = np.linalg.eigh(M)
    return (V * np.maximum(w, floor)) @ V.T


def r_src(R_raw: np.ndarray, xi: float, lambda_xi: float) -> np.ndarray:
    return psd(R_raw * np.exp(lambda_xi * (1 - xi)))


def r_time(R: np.ndarray, vel: np.ndarray, sigma_tau: float) -> np.ndarray:
    J = vel.reshape(2, 1)
    return psd(R + (J @ J.T) * sigma_tau ** 2)


def unit_ball_volume(d: int) -> float:
    return pi ** (d / 2) / gamma(d / 2 + 1)


def admit(nu: np.ndarray, S: np.ndarray, r: float, cfg, cold_start: bool = False,
          use_consistency: bool = True):
    """Return (decision, eta, d2)."""
    d = len(nu)
    d2 = float(nu @ np.linalg.solve(S, nu))
    if d2 > cfg.gate_chi2:
        return GATED, 0.0, d2
    if not use_consistency:
        eta = r
    else:
        detS = max(np.linalg.det(S), 1e-12)
        g = np.exp(-0.5 * d2) / ((2 * pi) ** (d / 2) * np.sqrt(detS))
        c_out = 1.0 / (unit_ball_volume(d) * cfg.gate_chi2 ** (d / 2) * np.sqrt(detS))
        eta = r * g / (r * g + (1 - r) * c_out)
    accept_thr = cfg.eta_accept + (0.15 if cold_start else 0.0)   # tightened when cold
    if eta <= cfg.eta_reject or 1.0 / max(eta, cfg.eps_r) > cfg.max_inflation:
        return REJECT, eta, d2
    if eta >= accept_thr:
        return ACCEPT, eta, d2
    return DOWNWEIGHT, eta, d2
