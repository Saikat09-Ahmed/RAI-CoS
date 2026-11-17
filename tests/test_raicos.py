import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raicos.admission import ACCEPT, GATED, REJECT, admit  # noqa: E402
from raicos.config import AdmissionConfig, ExperimentConfig, ReliabilityConfig, SchedulerConfig  # noqa: E402
from raicos.env import METHODS, run_episode  # noqa: E402
from raicos.imm import IMMState, ci_fuse, info_fuse  # noqa: E402
from raicos.reliability import ReliabilityModel  # noqa: E402
from raicos.scheduler import Candidate, duplicate_risk, exact_select, info_gain, urgency  # noqa: E402
from raicos.sources import Report  # noqa: E402
from raicos.tracker import BeliefManager, Flags  # noqa: E402


def _bm():
    cfg = ExperimentConfig()
    return BeliefManager(0, cfg, Flags(reliability=False, consistency=False), np.random.default_rng(0))


def _rep(uuid, t, z, src="uav0"):
    return Report(uuid, src, ("uav", "near", "clear", "moving"), np.asarray(z, float), np.eye(2) * 12.25, t, 0.95,
                  frozenset({uuid}), truth_target=0, truth_event=1)


def _run(order, zs):
    bm = _bm()
    for k in order:
        bm.ingest_points([(_rep(k + 1, float(k), zs[k]), float(k), 0.0, float(max(order)))], float(max(order)))
    bm.maintain(float(max(order)))
    return bm.tracks[next(iter(bm.tracks))]


def test_generation_time_replay_matches_in_order_filtering():
    rng = np.random.default_rng(3)
    zs = [np.array([8.0 * k, 3.0 * k]) + rng.normal(0, 3.5, 2) for k in range(12)]
    a = _run(list(range(12)), zs)
    order = list(range(12))
    order[7], order[9] = order[9], order[7]          # two reports arrive out of order after initiation
    b = _run(order, zs)
    assert np.allclose(a.state.x, b.state.x, atol=1e-6) and np.allclose(a.state.P, b.state.P, atol=1e-6)


def test_exact_duplicate_is_suppressed():
    bm = _bm()
    r1, r2 = _rep(1, 0.0, [0, 0]), _rep(2, 1.0, [8, 3])
    bm.ingest_points([(r1, 0.0, 0.0, 0.0)], 0.0)
    bm.ingest_points([(r2, 1.0, 0.0, 1.0)], 1.0)
    bm.ingest_points([(r2, 1.0, 0.0, 2.0)], 2.0)
    assert bm.stats["duplicate"] == 1


def test_ci_minimises_logdet_and_stays_conservative():
    P1 = np.diag([4.0, 25.0, 1.0, 1.0])
    P2 = np.diag([25.0, 4.0, 1.0, 1.0])
    _, Pci, w = ci_fuse(np.zeros(4), P1, np.zeros(4), P2)
    _, Pnaive = info_fuse(np.zeros(4), P1, np.zeros(4), P2)
    assert 0 <= w <= 1
    assert np.linalg.slogdet(Pci)[1] <= min(np.linalg.slogdet(P1)[1], np.linalg.slogdet(P2)[1]) + 1e-9
    assert np.linalg.slogdet(Pci)[1] >= np.linalg.slogdet(Pnaive)[1]


def test_reliability_backoff_lower_bound_and_drift():
    rm = ReliabilityModel(ReliabilityConfig())
    ctx = ("mobile", "near", "clear", "moving")
    r0, m0, cold = rm.score("mob1", ctx)
    assert cold and r0 < m0
    for _ in range(40):
        rm.update("mob1", ("mobile", "far", "clear", "moving"), 1, 1.0, 0.0)
    r1, _, cold = rm.score("mob1", ctx)          # unseen context backs off to the source level
    assert not cold and r1 > r0
    for k in range(30):
        rm.update("mob1", ctx, 0, 1.0, float(k))
    assert rm.fast_until["mob1"] > 0             # residual-change statistic triggered rho = 0.98


def test_admission_gate_and_thresholds():
    cfg = AdmissionConfig()
    S = np.eye(2) * 25.0
    assert admit(np.array([30.0, 0]), S, 0.9, cfg)[0] == GATED
    assert admit(np.array([0.5, 0.5]), S, 0.95, cfg)[0] == ACCEPT
    assert admit(np.array([11.0, 0]), S, 0.2, cfg)[0] == REJECT


def test_information_gain_grows_with_receiver_staleness_and_urgency_bounds():
    P = np.diag([9.0, 9.0, 1.0, 1.0])
    R = np.eye(2) * 12.25
    fresh = info_gain(P, 10.0, 10.0, 11.0, R, 1e-6)
    stale = info_gain(P, 0.0, 10.0, 11.0, R, 1e-6)
    assert stale > fresh > 0
    sc = SchedulerConfig()
    assert urgency(0, 0, 0, sc) == 1.0 and abs(urgency(100, 1, 1, sc) - 2.5) < 1e-9
    assert duplicate_risk(5, 3.0, {5}, [3.0], sc) == pytest.approx(1.0)


def test_exact_solver_respects_capacity():
    cands = [Candidate(_rep(i, 0.0, [0, 0]), 0.0, "uav1", 144, 1.0, 1.0, 1.0, 0.0, 1.0 + i, 1) for i in range(5)]
    out = exact_select(cands, 0.0, 8.0, {"uav1": 300})
    assert len(out) == 2 and {c.report.uuid for c in out} == {3, 4}


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_runs(method):
    cfg = ExperimentConfig(method=method)
    cfg.scenario.duration_s = 12
    r = run_episode(cfg)
    assert 0.0 <= r["continuity_pct"] <= 100.0 and r["tx_mb"] >= 0
