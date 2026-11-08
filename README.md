# MANTIS — Mission-Aware Network for Triage, Intelligence, and Survivability

A simulator and coordination framework for resilient, casualty-aware multi-UAV / multi-medic
operations: it covers the six experimental conditions, the phased roadmap E0–E8c, and the
experiments X1–X10 from the research plan.

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

## The MARL formulation

Hierarchical macro-actions on a 30-s epoch, plus event-triggered medic decisions. Each **medic** picks a
casualty or HOLD; each **UAV** picks observe(casualty), NETWORK role (relay/gateway, placed by the shared
planner), or RETURN. Actors are permutation-invariant candidate scorers, so one network serves 1–10
casualties and any fleet size. Decisions within a step are sequential, so agents see their teammates'
choices. A safety shield is applied identically to all learners: unreachable or unsafe casualties are
masked, and UAVs are forced home on low battery. Every learner gets the same observation features,
including an *advisor* feature (the heuristic stack's shadow recommendation), the same team reward
and the same 500-episode budget.

* `ippo` — independent PPO with local critics (independent policy-gradient baseline)
* `mappo` — MAPPO with a centralised team critic, from scratch
* `vdn`, `qmix` — value decomposition (QMIX uses a monotonic mixer that handles a variable number of agents)
* `bc` — behaviour cloning of the MANTIS-H stack
* `mantis_mappo` (proposed) — joint UAV+medic MAPPO, BC warm start for actor and critic, and a decaying
  KL anchor to the BC prior
* ablations: `abl_nokl`, `abl_mediconly`, `abl_uavonly`

## Running on Hellbender (SLURM)

```
cd ~/faithmm/mantis
nano hpc/config.sh          # set PARTITION / ACCOUNT / CPUS / EPISODES
bash hpc/setup_env.sh       # one time: creates .venv and runs the tests
bash hpc/submit_all.sh      # heuristic runs + BC + 8 training jobs (array) + selection/evaluation/report
squeue -u $USER             # monitor; logs in logs/, models in results/models/
```
The pipeline is CPU-bound: the simulator is the bottleneck and the networks are tiny, so it does not need a GPU.
Results from the first local 500-episode run are kept in `results_run1_500ep/` for comparison.

## Running locally

```
python -m pytest tests/                          # E0 tests
cd experiments && python run_heuristic.py X1 X2 X3 X4 X5 X6 X7 X5grid --procs=2
python -m mantis.marl.train A 500                # teacher data, BC, MANTIS-MAPPO, MAPPO, IPPO
python -m mantis.marl.train B 500                # VDN, QMIX
python -m mantis.marl.train C 500                # ablations
cd experiments && python run_marl_eval.py X8 X9 X10 --procs=2
python analysis/report.py                        # results/RESULTS.md, results/figures/
```

Requires Python ≥ 3.10, numpy, scipy, pandas, matplotlib and torch (CPU is enough).

## Status of results and next steps

See `results/RESULTS.md`. The hypothesis scorecard at the top is the honest summary. The learned joint policy
(MANTIS-MAPPO) clearly beats the MARL baselines trained from scratch, but it ties or loses to the tuned heuristic stack.
To give MARL a real chance to win:

1. **Training budget**: 500 episodes per method is about 1% of a typical MAPPO budget. Run `python -m mantis.marl.train A 20000`
   on a multi-core node: episodes are independent, so parallelise rollouts.
2. **Reduce reward variance**: train with a curriculum by configuration size, and use per-scenario baselines,
   e.g. a reward relative to MANTIS-H on the same seed.
3. **Close known gaps**: give learners the heuristic's pre-emptive battery handover, which explains most of the X9 gap.
4. **Make the learning problem matter**: in the current scenarios, safe routing and energy management dominate
   outcomes and allocation choices barely matter (X5 n.s.). Harsher medic-to-casualty ratios or longer missions
   would widen the room for learned coordination.

Component choices for MANTIS-H (urgency-distance allocation, periodic replanning) were informed by preliminary runs on
the X1–X7 seed range. X8–X10 use disjoint seed ranges, so the learned-vs-heuristic comparisons are not affected by that choice.
