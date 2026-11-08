# MANTIS — Mission-Aware Network for Triage, Intelligence, and Survivability

A simulator and coordination framework for resilient, casualty-aware multi-UAV / multi-medic
operations: it covers six experimental conditions, the phased roadmap, and the
experiments X1–X10.

## Layout

```
mantis/
  config.py       ScenarioConfig (conditions 1-6), PolicyConfig (heuristic stack), the 82 base configurations
  world.py        map, buildings / restricted airspace, dynamic obstacles, ground hazard field,
                  risk-aware ground routing, flight routing, wireless link model
  vitals.py       physiology -> vitals (HR, RR, SBP, SpO2, GCS); severity from vitals only
                  (Revised Trauma Score + shock index + SpO2); triage T1/T2/T3
  sim.py          discrete-time simulator (dt = 5 s, 20-min missions), packets, multi-hop delivery,
                  knowledge base from delivered packets, medic motion / treatment / hazard exposure, metrics
  planning.py     shared executors: relay/gateway network planner, medic allocation scores
  heuristics.py   UAV policies, failure recovery, energy management, medic allocation, replanning
  marl/           features, candidate-scoring networks, controller, BC / IPPO / MAPPO / VDN / QMIX / MANTIS-MAPPO
experiments/      run_heuristic.py (X1-X7, 82-config grid), run_marl_eval.py (X8-X10), common.py (stats)
analysis/report.py  -> results/RESULTS.md + results/figures/*.png
tests/test_core.py  E0 tests (determinism, matched scenarios, severity model, safety constraint, accounting)
```

## Key modelling decisions

* **Severity is known only from vitals.** Observer UAVs measure noisy vitals; the medics' knowledge base
  holds only what was actually *delivered* through the multi-hop network (or what a medic sees within
  50 m). Severity = f(RTS, shock index, SpO2); the deterioration rate is estimated from the vitals trend.
  `severity_mode="known"` gives the oracle used for the initial ground-planning model; `"reports"` gives
  location-only incident reports with no UAV vitals.
* **Medic safety is a hard constraint.** The ground hazard field (static + moving hazard zones) gives each
  cell a risk in [0, 1]. A medic is incapacitated with hazard rate κ·risk³. The *safe* router forbids any
  cell whose current or 60-s-predicted risk exceeds `risk_max = 0.35` and weights risk in the path cost.
  Routes are re-checked every step, medics retreat if a hazard moves onto them, and treatment is
  aborted if risk at the treatment site rises. Casualties inside unsafe zones are **deferred** (the
  patient may be compromised) until the hazard moves or they move out.
* **Route replanning** happens on location / severity / hazard events (executor level), and the allocation
  replanning strategy is selectable (static, periodic, event, selective, deadline-selective).
* **Matched scenarios.** All exogenous randomness is pre-sampled from seed-derived streams, so every policy
  sees an identical map, casualties, physiology, mobility, failures and hazards for a given seed.
  Training seeds are < 1e6; evaluation seeds are ≥ 1e7.
