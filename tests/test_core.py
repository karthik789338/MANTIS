"""E0 tests: reproducibility, vitals/severity, safety constraint, comm accounting, configs."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mantis.config import ScenarioConfig, MANTIS_H, base_configurations
from mantis.sim import Simulator, M_INCAP
from mantis.heuristics import Combined
from mantis import vitals as V


def run(cfg, p=MANTIS_H):
    sim = Simulator(cfg, p, Combined(p))
    return sim, sim.run()


def test_82_configs():
    assert len(base_configurations()) == 82


def test_matched_scenarios_identical_across_policies():
    cfg = ScenarioConfig(seed=11, n_casualties=5, n_uavs=7, n_medics=3, mobility="mixed", hazards="dynamic")
    a = Simulator(cfg, MANTIS_H, None); b = Simulator(cfg, MANTIS_H.copy(medic_alloc="nearest"), None)
    assert np.allclose(a.cpos, b.cpos) and np.allclose(a.cp, b.cp) and np.allclose(a.world.hz_c, b.world.hz_c)


def test_deterministic():
    cfg = ScenarioConfig(seed=3, n_casualties=5, n_uavs=7, n_medics=3)
    _, m1 = run(cfg); _, m2 = run(cfg)
    assert m1["TIR"] == m2["TIR"] and m1["sev_w_survival"] == m2["sev_w_survival"]


def test_severity_monotone_in_reserve():
    p = np.linspace(0.05, 1, 40)
    sev = V.severity_from_vitals(V.vitals_from_reserve(p))
    assert sev[0] > sev[-1] and V.triage_from_severity(sev[0]) == 2 and V.triage_from_severity(sev[-1]) == 0
    est = V.reserve_from_vitals(V.vitals_from_reserve(p))
    assert np.abs(est - p).max() < 0.05


def test_safe_routes_respect_risk_limit():
    cfg = ScenarioConfig(seed=5, n_casualties=10, n_uavs=12, n_medics=4, hazards="static")
    sim = Simulator(cfg, MANTIS_H, None)
    W = sim.world
    r = W.risk_planning(0)
    for c in range(sim.n):
        path = sim.ground_path(sim.mpos[0], sim.cpos[c], "safe")
        if path is None:
            continue
        i, j = W.cells_of(np.array(path[:-1]))
        assert (r[i, j] <= cfg.risk_max + 1e-9).all()


def test_packet_accounting():
    sim, m = run(ScenarioConfig(seed=2, n_casualties=10, n_uavs=9, n_medics=2))
    assert 0 <= m["TIR"] <= 1 and 0 <= m["PDR"] <= 1
    assert sim.m_delivered <= sim.m_gen
    assert sim.slot_ok.sum() <= sim.slot_needed.sum()


def test_condition1_runs_and_serves():
    sims = [run(ScenarioConfig(seed=s))[1] for s in range(4)]
    assert np.mean([m["completion"] for m in sims]) > 0.5


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
