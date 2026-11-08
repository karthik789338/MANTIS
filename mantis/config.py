"""Scenario configuration for MANTIS.

All physical quantities are SI (metres, seconds). One simulation step = ``dt`` seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Any


@dataclass
class ScenarioConfig:
    # ---- scale -----------------------------------------------------------------
    n_casualties: int = 1
    n_uavs: int = 3
    n_medics: int = 1
    n_spare_uavs: int = 0            # extra UAVs waiting at base (energy experiments)
    seed: int = 0
    horizon: int = 240               # steps (240 x 5 s = 20 min)
    dt: float = 5.0
    map_size: float = 1000.0
    grid: int = 40                   # 25 m cells

    # ---- condition 3: UAV obstacles ---------------------------------------------
    obstacles: str = "clear"         # clear | static | dynamic | mixed
    # ---- ground hazards (medic safety) ------------------------------------------
    hazards: str = "static"          # none | static | dynamic | mixed
    n_hazards: int = 3
    # ---- medic staging and casualty placement (triage experiments) -------------
    ccp_distance: float = 450.0       # medics' collection point distance from the incident centre (m)
    casualty_layout: str = "random"   # random | critical_near | critical_far | severity_clusters | scattered
    # ---- condition 5: casualty mobility ------------------------------------------
    mobility: str = "stationary"     # stationary | slow | intermittent | abrupt | mixed
    # ---- condition 2: failures / communication disruption ------------------------
    failures: List[Dict[str, Any]] = field(default_factory=list)
    # each failure: {"t": step, "role": "observer"|"relay"|"gateway"|"any"} or {"t": step, "uav": idx}
    disruption: str = "none"         # none | intermittent | complete
    blackout: tuple = (90, 120)      # steps of complete disruption
    link_degradation: float = 0.0    # 0..0.5 shrinks link ranges
    # ---- condition 6: severity -----------------------------------------------------
    severity_mode: str = "delivered"  # known (oracle) | delivered (UAV packets only) | reports (no UAV vitals)
    severity_dynamics: str = "fixed"  # fixed | changing
    # ---- energy (E8a) --------------------------------------------------------------
    energy: bool = False
    battery_hetero: bool = False
    endurance_steps: int = 150        # nominal hover endurance (12.5 min)

    # ---- physical parameters -----------------------------------------------------
    uav_alt: float = 60.0
    uav_speed: float = 10.0
    medic_speed: float = 1.5
    sense_radius: float = 80.0
    medic_sense_radius: float = 50.0
    reach_radius: float = 20.0
    node_capacity: int = 4            # packets forwarded per UAV per step
    queue_max: int = 40
    report_slot: int = 2              # one report per casualty every 2 steps (10 s)
    timely_deadline: int = 6          # packet must arrive within 30 s to be "timely"
    packet_ttl: int = 24
    heartbeat_timeout: int = 3
    risk_max: float = 0.35            # medic hard safety limit on cell risk
    risk_weight: float = 12.0         # risk weight in safe routing cost
    incap_kappa: float = 0.03         # medic incapacitation hazard = kappa * risk^3 per second
    recharge_steps: int = 12

    def copy(self, **kw) -> "ScenarioConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------
# The 82 original base configurations (Condition 6 table)
# ---------------------------------------------------------------------------------
FIVE_UAVS = [3, 5, 7, 9, 11, 13, 15]
TEN_UAVS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
FIVE_MEDICS = [1, 3, 5]
TEN_MEDICS = [1, 2, 4, 6, 8, 10]


def base_configurations() -> List[Dict[str, int]]:
    cfgs = [dict(n_casualties=1, n_medics=1, n_uavs=3)]
    for m in FIVE_MEDICS:
        for u in FIVE_UAVS:
            cfgs.append(dict(n_casualties=5, n_medics=m, n_uavs=u))
    for m in TEN_MEDICS:
        for u in TEN_UAVS:
            cfgs.append(dict(n_casualties=10, n_medics=m, n_uavs=u))
    assert len(cfgs) == 82, len(cfgs)
    return cfgs


@dataclass
class PolicyConfig:
    """Heuristic stack. Every experiment varies one component and fixes the rest."""
    uav_policy: str = "balanced"      # fixed | coverage | connected | balanced
    recovery: str = "coordinated"     # none | adaptive | coordinated
    positioning: str = "adaptive"     # static | reactive | adaptive
    routing: str = "queue"            # shortest | etx | queue
    scheduling: str = "guarded"       # fifo | sampled | deadline | deficit | guarded
    medic_alloc: str = "urgency"      # nearest | severity | urgency | load | gain (survival-gain triage)
    medic_route: str = "safe"         # shortest | safe
    replan: str = "periodic"          # static | periodic | event | selective | deadline
    energy_policy: str = "aware"      # fixed | reactive | aware
    epoch: int = 6                    # decision epoch in steps (30 s)

    def copy(self, **kw) -> "PolicyConfig":
        return replace(self, **kw)


MANTIS_H = PolicyConfig()  # strongest heuristic stack
