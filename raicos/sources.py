"""Heterogeneous point-report sources (Sec. 3.3, 4.2, 5.2).

Every source emits the 2-D Cartesian position report of Eq. (4) as the
logical record of Eq. (6): UUID, source, context, (hidden) target, type,
observation, raw covariance, source-clock generation time, confidence xi,
ancestry and deadline class.

* UAV: nominal sigma 3.5 m, inflated with range and degraded visibility.
* Fixed monitor: same form plus a persistent-bias state; failing monitors
  either go silent or develop a bias.
* Mobile (simulated crowd) source: Gaussian/Student-t mixture; the Student-t
  component is biased and heavy-tailed (nominal 10%).
Latent quality propensities: Beta(18,2) UAV, Beta(14,3) monitor, Beta(8,4)
mobile, shifting a source-type failure logit. Latent reliability and the
truth event A_k are used only by the evaluator.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .roads import RoadTarget

_uuid = itertools.count(1)
CHI2_2_95 = 5.991


@dataclass
class Report:
    uuid: int
    source: str
    ctx: tuple                     # (type, range band, visibility, motion regime)
    z: np.ndarray
    R_raw: np.ndarray
    tau_g: float                   # generation time in the source clock
    xi: float                      # source confidence
    ancestry: frozenset
    deadline: str = "ordinary"
    # receiver-local / evaluator-only fields (never used by online logic)
    truth_target: int = -1
    truth_event: int = -1          # A_k
    t_true: float = 0.0
    origin_uav: str = ""


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SourceField:
    def __init__(self, cfg, rmap, rng: np.random.Generator):
        self.cfg, self.map, self.rng = cfg, rmap, rng
        ext = rmap.extent
        # occlusion zones (degraded visibility)
        self.occl = [(ext * a, ext * b, ext * (a + 0.22), ext * (b + 0.22))
                     for a, b in rng.uniform(0.1, 0.65, size=(3, 2))]
        nodes = rmap.nodes
        pick = rng.choice(len(nodes), cfg.num_monitors, replace=False)
        self.monitors = nodes[pick].copy()
        qp = cfg.quality_prior
        self.q_mon = rng.beta(*qp["monitor"], size=cfg.num_monitors)
        self.mon_fail = rng.random(cfg.num_monitors) < cfg.monitor_failure
        self.mon_fail_t = rng.uniform(0.1, 0.8, cfg.num_monitors) * cfg.duration_s
        self.mon_fail_kind = rng.choice(["outage", "bias"], cfg.num_monitors)
        ang = rng.uniform(0, 2 * np.pi, cfg.num_monitors)
        self.mon_bias = np.c_[np.cos(ang), np.sin(ang)] * rng.uniform(25, 45, (cfg.num_monitors, 1))
        self.mobiles = [RoadTarget(rmap, rng, -1) for _ in range(cfg.num_mobile_potential)]
        for m in self.mobiles:
            m.cruise = rng.uniform(1.2, 6.0)
        self.q_mob = rng.beta(*qp["mobile"], size=cfg.num_mobile_potential)
        self.mob_active = rng.random(cfg.num_mobile_potential) < cfg.mobile_participation
        self.q_uav = rng.beta(*qp["uav"], size=cfg.num_uavs)
        self.clock = {}
        for i in range(cfg.num_uavs):
            self.clock[f"uav{i}"] = rng.normal(0, cfg.clock_offset_sd_s)
        for i in range(cfg.num_monitors):
            self.clock[f"mon{i}"] = rng.normal(0, cfg.clock_offset_sd_s)
        for i in range(cfg.num_mobile_potential):
            self.clock[f"mob{i}"] = rng.normal(0, 2 * cfg.clock_offset_sd_s)
        self.drifted = False

    # ----------------------------------------------------------------- helpers
    def degraded(self, p) -> bool:
        return any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in self.occl)

    def _ctx(self, stype: str, rng_m: float, radius: float, p, speed: float) -> tuple:
        band = "near" if rng_m < radius / 3 else ("medium" if rng_m < 2 * radius / 3 else "far")
        regime = "slow" if speed < 3.0 else "moving"
        return (stype, band, "degraded" if self.degraded(p) else "clear", regime)

    def _truth_event(self, z, p_true, R_raw) -> int:
        e = z - p_true
        Rs = R_raw + 1e-2 * np.eye(2)
        return int(e @ np.linalg.solve(Rs, e) <= CHI2_2_95)

    def step(self, t: float, dt: float) -> None:
        for m in self.mobiles:
            m.step(dt)
        if self.rng.random() < 0.01:                     # participation churn
            k = self.rng.integers(len(self.mobiles))
            self.mob_active[k] = self.rng.random() < self.cfg.mobile_participation
        if self.cfg.reliability_drift_at_s > 0 and t >= self.cfg.reliability_drift_at_s and not self.drifted:
            self.drifted = True                           # abrupt reliability drift
            k = self.rng.choice(len(self.q_mob), size=len(self.q_mob) // 3, replace=False)
            self.q_mob[k] = 0.25
            self.q_mon[: max(1, len(self.q_mon) // 4)] = 0.3

    # ---------------------------------------------------------------- sensing
    def uav_reports(self, uid: int, pos: np.ndarray, targets, t: float) -> List[Report]:
        c, out = self.cfg, []
        src = f"uav{uid}"
        for tg in targets:
            d = float(np.linalg.norm(tg.pos - pos))
            if d > c.uav_sense_radius_m:
                continue
            deg = self.degraded(tg.pos)
            if self.rng.random() > (0.72 if deg else 0.95):
                continue
            sig = c.uav_sigma_m * (1 + 0.6 * d / c.uav_sense_radius_m) * (1.8 if deg else 1.0)
            fail_p = _sigmoid(_logit(0.02) - 12 * (self.q_uav[uid] - 0.9) + (1.0 if deg else 0))
            z = tg.pos + self.rng.normal(0, sig, 2)
            if self.rng.random() < fail_p:
                z = z + self.rng.standard_t(3, 2) * 3 * sig
            R_raw = np.eye(2) * (1.5 * sig) ** 2   # declared (conservative) incl. range/visibility
            xi = float(np.clip(1 - 0.5 * d / c.uav_sense_radius_m - (0.3 if deg else 0), 0.1, 1))
            uu = next(_uuid)
            out.append(Report(uu, src, self._ctx("uav", d, c.uav_sense_radius_m, tg.pos, tg.speed), z, R_raw,
                              t + self.clock[src], xi, frozenset({uu}), truth_target=tg.tid,
                              truth_event=self._truth_event(z, tg.pos, R_raw), t_true=t, origin_uav=src))
        return out

    def exogenous_reports(self, targets, t: float) -> List[Report]:
        c, out = self.cfg, []
        if round(t) % max(1, int(c.monitor_period_s)) == 0:
            for m, mp in enumerate(self.monitors):
                if self.mon_fail[m] and t >= self.mon_fail_t[m] and self.mon_fail_kind[m] == "outage":
                    continue
                for tg in targets:
                    d = float(np.linalg.norm(tg.pos - mp))
                    if d > c.monitor_radius_m:
                        continue
                    src = f"mon{m}"
                    sig = c.monitor_sigma_m * (1 + 0.5 * d / c.monitor_radius_m)
                    z = tg.pos + self.rng.normal(0, sig, 2)
                    if self.mon_fail[m] and t >= self.mon_fail_t[m]:
                        z = z + self.mon_bias[m]
                    elif self.rng.random() < _sigmoid(_logit(0.04) - 12 * (self.q_mon[m] - 0.82)):
                        z = z + self.rng.normal(0, 4 * sig, 2)
                    R_raw = np.eye(2) * (1.5 * sig) ** 2
                    uu = next(_uuid)
                    out.append(Report(uu, src, self._ctx("monitor", d, c.monitor_radius_m, tg.pos, tg.speed), z,
                                      R_raw, t + self.clock[src], 0.9, frozenset({uu}), truth_target=tg.tid,
                                      truth_event=self._truth_event(z, tg.pos, R_raw), t_true=t))
        if round(t) % max(1, int(c.mobile_period_s)) == 0:
            for k, mob in enumerate(self.mobiles):
                if not self.mob_active[k]:
                    continue
                for tg in targets:
                    d = float(np.linalg.norm(tg.pos - mob.pos))
                    if d > c.mobile_radius_m:
                        continue
                    src = f"mob{k}"
                    sig = c.mobile_error_m
                    base = _logit(c.mobile_outlier)
                    fail = _sigmoid(base - 12 * (self.q_mob[k] - 0.67) + (0.8 if self.degraded(tg.pos) else 0)
                                    + 0.05 * tg.speed)
                    z = tg.pos + self.rng.normal(0, sig, 2)
                    if self.rng.random() < fail:                 # biased Student-t component
                        ang = self.rng.uniform(0, 2 * np.pi)
                        bias = np.array([np.cos(ang), np.sin(ang)]) * self.rng.uniform(25, 60)
                        z = tg.pos + bias + self.rng.standard_t(2, 2) * 1.5 * sig
                    R_raw = np.eye(2) * (1.5 * sig) ** 2
                    xi = float(np.clip(self.rng.normal(0.7, 0.15), 0.1, 1))
                    uu = next(_uuid)
                    out.append(Report(uu, src, self._ctx("mobile", d, c.mobile_radius_m, tg.pos, tg.speed), z,
                                      R_raw, t + self.clock[src], xi, frozenset({uu}), truth_target=tg.tid,
                                      truth_event=self._truth_event(z, tg.pos, R_raw), t_true=t))
        return out

    def position_of(self, source: str) -> Optional[np.ndarray]:
        if source.startswith("mon"):
            return self.monitors[int(source[3:])]
        if source.startswith("mob"):
            return self.mobiles[int(source[3:])].pos
        return None
