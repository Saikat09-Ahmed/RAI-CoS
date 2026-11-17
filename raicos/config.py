"""Declared operating settings of RAI-CoS.

Values marked [paper] are stated in the main text of Rashid, Ahmed & Siddique,
Computers 2026, 15(10), 711 (Sections 3-5, Table 1). Values marked [repo] are
given in the article's Supplementary Tables (S7, S8, S10, S15), which are not
part of the main text; this repository uses documented defaults for them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ScenarioConfig:
    road_map: str = "beijing"          # beijing (train/val/test) | chicago-grid | normal (transfer)  [paper, synthetic stand-ins]
    duration_s: float = 600.0          # [paper]
    dt: float = 1.0                    # 1 s decisions [paper]
    num_uavs: int = 8                  # [paper]
    num_targets: int = 10              # [paper]
    num_monitors: int = 12             # [paper]
    num_mobile_potential: int = 30     # [paper]
    mobile_participation: float = 0.75  # [paper]
    monitor_failure: float = 0.10      # [paper]
    mobile_outlier: float = 0.10       # nominal Student-t share [paper]
    mobile_error_m: float = 8.0        # [repo] (20 m under severe stress [paper])
    uav_sigma_m: float = 3.5           # R_UAV = (3.5 m)^2 I [paper]
    uav_sense_radius_m: float = 170.0  # [repo]
    uav_speed_mps: float = 16.0        # waypoint increment per epoch [repo]
    monitor_sigma_m: float = 5.0       # [repo]
    monitor_radius_m: float = 170.0     # [repo]
    monitor_period_s: float = 1.0      # [repo]
    mobile_radius_m: float = 160.0      # [repo]
    mobile_period_s: float = 1.0       # [repo]
    clock_offset_sd_s: float = 0.05    # hidden source-clock offsets [repo]
    capture_radius_m: float = 20.0     # [paper]
    capture_hold_s: float = 5.0        # [paper]
    reliability_drift_at_s: float = -1.0   # >0: abrupt source-quality drift at this time
    collision_sep_m: float = 25.0      # safety controller separation [repo]
    # latent quality propensities [paper]
    quality_prior: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "uav": (18.0, 2.0), "monitor": (14.0, 3.0), "mobile": (8.0, 4.0)})


@dataclass
class NetworkConfig:
    loss: float = 0.20                 # [paper]
    delay_mean_s: float = 2.0          # [paper]
    jitter_s: float = 0.5              # [paper]
    burst: bool = False                # Gilbert-Elliott burst loss instead of i.i.d.
    blackouts: List[Tuple[float, float]] = field(default_factory=list)
    radio_range_m: float = 1500.0      # physical reachability mask [repo]
    neighbourhood: int = 7             # nearest 7 reachable peers [paper]
    link_capacity_bytes: int = 1500    # per directed link per epoch [repo]
    trace_file: str = ""               # optional ns-3 style delivery trace to replay


@dataclass
class PacketConfig:
    point_payload: int = 112           # [paper]
    digest_payload: int = 160          # [paper]
    envelope: int = 32                 # hop envelope [paper]
    ack: int = 32                      # counted acknowledgment envelope [paper: envelope-sized]
    sync_idle_s: float = 5.0           # [paper]
    provenance_ledger: int = 16        # full ancestor ids kept [paper]
    lifetime_s: float = 30.0           # ordinary transport lifetime [paper]
    urgent_lifetime_s: float = 5.0     # track-loss alerts [paper]
    fixed_lag_s: float = 15.0          # L [paper]
    budget_bytes_per_s: float = 1800.0  # B_bar = 50% of uncapped AllShare in this simulator (paper rule; 5.17 KB/s in the paper)


@dataclass
class ReliabilityConfig:
    lambda_shrink: float = 20.0        # [paper]
    quantile: float = 0.05             # [paper]
    rho: float = 0.995                 # [paper]
    rho_fast: float = 0.98             # [paper]
    rho_fast_hold_s: float = 30.0      # [paper]
    lambda_D: float = 0.95             # [paper]
    D_threshold: float = 0.35          # [paper]
    D_consecutive: int = 5             # [paper]
    retention_s: float = 120.0         # [paper]
    retention_records: int = 256       # [paper]
    confirm_chi2: float = 5.991        # chi2_{2,0.95} [paper]
    # source-type priors (alpha0, beta0) [repo]
    type_prior: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "uav": (9.0, 1.0), "monitor": (7.0, 1.5), "mobile": (4.0, 2.0)})


@dataclass
class AdmissionConfig:
    gate_chi2: float = 10.597          # chi2_{2,0.995} [paper]
    eta_reject: float = 0.15           # [paper]
    eta_accept: float = 0.65           # [paper]
    eps_r: float = 0.05                # [paper]
    max_inflation: float = 20.0        # [paper]
    lambda_xi: float = 1.5             # [paper]


@dataclass
class TrackerConfig:
    oosm_buffer_s: float = 15.0        # 15 s ring buffer, 16 epoch states [paper]
    init_reports: int = 2              # two associated reports within 5 s [paper]
    init_window_s: float = 5.0
    uncertain_after_s: float = 6.0     # [paper]
    lost_after_s: float = 12.0         # [paper]
    delete_after_s: float = 30.0       # [paper]
    max_tracks: int = 12               # [paper]
    ambiguity_margin: float = 0.5      # NLL margin for deferral [repo]
    ci_grid: int = 21                  # [paper]
    ci_tol: float = 1e-3               # golden-section tolerance [paper]
    reg: float = 1e-6                  # matrix regularisation [paper]
    # IMM [repo]
    mode_sigma_a: Tuple[float, float, float, float] = (0.5, 0.2, 1.5, 1.0)  # CV, slow/stop, accel, CT
    mode_switch: float = 0.04
    ct_max_rate: float = 0.35


@dataclass
class SchedulerConfig:
    V_dpp: float = 8.0                 # [paper]
    p_del_min: float = 0.20            # [paper]
    dI_min_nats: float = 0.0075        # [paper]
    eps_I: float = 1e-6                # [repo]
    age_norm_s: float = 12.0           # a_age = min{1, A/12} [paper]
    urgency_min: float = 1.0           # [paper]
    urgency_max: float = 2.5           # [paper]
    w_age: float = 0.4                 # [paper]
    w_loss: float = 0.4                # [paper]
    w_cap: float = 0.2                 # [paper]
    w_D: float = 0.5                   # duplicate-risk weight [repo]
    tau_D_s: float = 2.0               # [repo]
    lambda_dup: float = 0.6            # [repo]
    lambda_E: float = 0.05             # [repo]
    ack_timeout_s: float = 8.0         # [repo]
    link_ewma: float = 0.1             # [repo]
    quantize_bytes: int = 16           # exact diagnostic solver [paper]


@dataclass
class EnergyConfig:
    P0_w: float = 145.0                # [paper]
    Pv: float = 0.42                   # W/(m/s)^2 [paper]
    Pa: float = 0.20                   # W/(m/s^2)^2 [paper]
    tx_uj_per_byte: float = 2.2        # [paper]
    rx_uj_per_byte: float = 1.3        # [paper]
    compute_w: float = 9.0             # [repo]
    battery_wh: float = 120.0          # [repo]
    reserve: float = 0.20              # [paper]


@dataclass
class ExperimentConfig:
    method: str = "raicos"
    seed: int = 0
    actor: str = "scripted"            # scripted | learned
    actor_weights: str = ""
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    packets: PacketConfig = field(default_factory=PacketConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def severe(cfg: ExperimentConfig) -> ExperimentConfig:
    """Compound-severe stress (Table 3): 40% loss, 5 s delay, 1 s jitter, 25%
    monitor failure, 50% mobile participation, 20 m mobile error, 15 s outage."""
    cfg.network.loss, cfg.network.delay_mean_s, cfg.network.jitter_s = 0.40, 5.0, 1.0
    cfg.scenario.monitor_failure, cfg.scenario.mobile_participation = 0.25, 0.50
    cfg.scenario.mobile_error_m = 20.0
    cfg.network.blackouts = [(300.0, 315.0)]
    return cfg
