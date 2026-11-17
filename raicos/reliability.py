"""Context-conditioned source reliability (Sec. 3.5, 4.3; Eqs. 9, 10, 16, 17).

Each receiver keeps its own forgotten Beta posteriors at three levels:
source-context, source and source type (prior). Sparse source-context keys
back off to the source level and then to the type prior through
excess-count shrinkage (Eq. 16, lambda_shrink = 20). The control score is the
exact 5th percentile of the effective posterior (Eq. 10); calibration uses
the pre-confirmation posterior mean.

Forgetting: alpha_t = alpha0 + rho (alpha_{t-1} - alpha0) + w y (Eq. 9) with
rho = 0.995. A residual-change statistic D (Eq. 17, lambda_D = 0.95) switches
to rho = 0.98 for 30 s after five consecutive confirmations above 0.35.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from scipy.stats import beta as beta_dist


def source_type(source: str) -> str:
    return "uav" if source.startswith("uav") else ("monitor" if source.startswith("mon") else "mobile")


class ReliabilityModel:
    def __init__(self, cfg, use_lower_bound: bool = True, forgetting: bool = True):
        self.cfg = cfg
        self.use_lb = use_lower_bound
        self.forgetting = forgetting
        self.sc: Dict[Tuple[str, tuple], np.ndarray] = {}
        self.s: Dict[str, np.ndarray] = {}
        self.D: Dict[str, float] = defaultdict(float)
        self.D_run: Dict[str, int] = defaultdict(int)
        self.fast_until: Dict[str, float] = defaultdict(lambda: -1.0)
        self._cache: Dict[Tuple[str, tuple], Tuple[float, float]] = {}

    def prior(self, source: str) -> np.ndarray:
        return np.array(self.cfg.type_prior[source_type(source)], float)

    def _rho(self, source: str, t: float) -> float:
        if not self.forgetting:
            return 1.0
        return self.cfg.rho_fast if t < self.fast_until[source] else self.cfg.rho

    def update(self, source: str, ctx: tuple, y: int, w: float, t: float) -> None:
        a0 = self.prior(source)
        rho = self._rho(source, t)
        for table, key in ((self.sc, (source, ctx)), (self.s, source)):
            cur = table.get(key, a0.copy())
            cur = a0 + rho * (cur - a0) + w * np.array([y, 1 - y], float)
            table[key] = cur
        # residual-change statistic (Eq. 17)
        lam = self.cfg.lambda_D
        self.D[source] = lam * self.D[source] + (1 - lam) * (1 - y)
        if self.D[source] > self.cfg.D_threshold:
            self.D_run[source] += 1
            if self.D_run[source] >= self.cfg.D_consecutive:
                self.fast_until[source] = t + self.cfg.rho_fast_hold_s
                self.D_run[source] = 0
        else:
            self.D_run[source] = 0
        self._cache.clear()

    def effective(self, source: str, ctx: tuple) -> Tuple[float, float, bool]:
        """(alpha~, beta~, cold_start) after hierarchical shrinkage (Eq. 16)."""
        a0 = self.prior(source)
        lam = self.cfg.lambda_shrink
        s_cnt = self.s.get(source)
        cold = s_cnt is None
        if cold:
            return float(a0[0]), float(a0[1]), True
        n_s = max(s_cnt.sum() - a0.sum(), 0.0)
        w_s = n_s / (n_s + lam)
        eff_s = a0 + w_s * (s_cnt - a0)                       # source backed off to type prior
        sc_cnt = self.sc.get((source, ctx))
        if sc_cnt is None:
            return float(eff_s[0]), float(eff_s[1]), False
        n_sc = max(sc_cnt.sum() - a0.sum(), 0.0)
        w = n_sc / (n_sc + lam)
        eff = a0 + w * (sc_cnt - a0) + (1 - w) * (eff_s - a0)
        return float(max(eff[0], 1e-3)), float(max(eff[1], 1e-3)), False

    def score(self, source: str, ctx: tuple) -> Tuple[float, float, bool]:
        """(control score r^0.05, posterior mean, cold_start)."""
        key = (source, ctx)
        if key in self._cache:
            return self._cache[key]
        a, b, cold = self.effective(source, ctx)
        mean = a / (a + b)
        r = float(beta_dist.ppf(self.cfg.quantile, a, b)) if self.use_lb else mean
        self._cache[key] = (r, mean, cold)
        return r, mean, cold
