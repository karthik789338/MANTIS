# MANTIS — experimental results

## Hypothesis scorecard (read this first)

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| C2 | Failure recovery restores casualty awareness | X1: TIR 0.777→0.817, role restoration 76 s→17 s, AoI 34 s→16 s vs no recovery (p<0.001). Coordinated vs adaptive: restoration 1.7 s faster (p=0.003); TIR/AoI equal | **Supported** (vs none); coordinated≈adaptive |
| C3 | Obstacle-aware positioning keeps connectivity safely | X2: TIR 0.40→0.90, collisions 9.6→0.2 per mission vs straight-line (p<1e-10). Link-aware "adaptive" ≈ "reactive" (n.s.) | **Supported** (vs static); adaptive≈reactive |
| C4 | Fleet size needed for timely delivery | X3: TIR ≥ 0.80 needs 5 UAVs for 5 casualties and 9 UAVs for 10; balanced > connected and fixed overall (p<0.03); coverage-first is best at 3–5 UAVs, fixed is marginally best at ≥12 UAVs | **Answered**; balanced best overall, not at every size |
| E6 | Priority/fairness scheduling improves timeliness | X4: queue-aware routing is the big win (TIR 0.75→0.93). Guarded scheduling **lowers** TIR (0.888 vs 0.931 FIFO) because it discards superseded reports; it has lower latency. FIFO/EDF/deficit are indistinguishable (congestion rarely binds) | **Not supported** for guarded scheduling |
| C5 | Replanning under casualty mobility | X6: periodic replanning beats static (+5.8 pp survival, p=0.007); event/selective/deadline-selective are not significantly different in survival; selective and deadline-selective cut reassignments by ~50% | **Supported** (vs static) |
| C6 | Severity-aware allocation from vitals | X5: under safe routing, urgency-distance ≥ nearest/severity/load/survival-gain, but the differences are small and not significant (best p=0.09). The proposed survival-gain rule did **not** beat urgency-distance | **Not supported** (differences n.s.) |
| Safety | Medics are kept safe | X5: safe routing cuts medic incapacitation 3.48→0.075 per mission (−98%, p≈1e-14) with **no significant loss** in severity-weighted survival (0.632 vs 0.613, p=0.55). X10a: removing safe routing is the most damaging ablation | **Strongly supported** |
| E8a | Energy-aware UAV operation | X7: 0 stranded UAVs vs 12.3 (no management) and 1.2 (threshold return), p<1e-5; TIR +1.9 pp vs fixed | **Supported** |
| H | Joint coordination beats independently managed policies | 82-config grid: MANTIS-H beats the naive independent stack (survival +14.8 pp, TIR +26 pp, medic loss 3.5→0.09; p<1e-13). Against an independent stack that **also** uses safe routing, survival and mission score are n.s. and TIR is 0.9 pp worse. X9 compounded: MANTIS-H mission score 2.34 vs 0.99 (p<1e-10), driven mainly by energy management (stranded 0.2 vs 7.4); survival n.s. | **Partly supported**: gains come from safe routing and energy management, not from joint coupling itself |
| X8 | Joint MARL outperforms heuristics and MARL baselines | MANTIS-MAPPO beats MAPPO (+0.68), IPPO (+0.48), VDN (+0.29) mission score (p≤0.001); vs QMIX n.s. It does **not** beat the heuristics: 2.22 vs MANTIS-H 2.29 (p=0.06), and is significantly worse in the compounded X9 setting (2.01 vs 2.34, p<0.001) | **Supported vs MARL baselines; not supported vs heuristics** |

Why MARL did not beat the heuristics: with 500 training episodes per method on 2 CPU cores, PPO fine-tuning did
not measurably improve on the behaviour-cloned prior. On validation, BC scored 2.55, the best RL checkpoint 2.57
(episode 104), and later checkpoints declined. The partially learned variants (medic-only 2.32, UAV-only 2.34) are
level with MANTIS-H. In X9 the learned agents strand more UAVs (0.98 vs 0.20), because the learners' battery shield is
less conservative than the heuristic's predictive return with pre-emptive replacement. These are budget and design
limits, not proof that MARL cannot win; see "Next steps" in the README.


All numbers are means ± 95% t-confidence half-widths over matched scenarios (same seed ⇒ identical map, casualties, physiology, mobility, failures, hazards for every policy). Evaluation seeds ≥ 10,000,000; training seeds < 1,000,000. Paired comparisons use the Wilcoxon signed-rank test; **supported** = better at p < 0.05, *worse* = significantly worse, n.s. = not significant.


## X1 — UAV role failure and recovery (Condition 2)

Failures injected at t = 150 s (and later for successive/multiple). Configurations 1c/3u/1m, 5c/5u/3m, 10c/9u/4m, 10c/15u/4m; 10 matched seeds per cell. `role_restore_s` = time until every casualty the failed UAV was sensing is sensed again and observer connectivity is back to its pre-failure level.

| recovery | TIR | AoI_s | role_restore_s | coverage | sev_w_survival |
|---|---|---|---|---|---|
| none | 0.777 ± 0.028 | 34.319 ± 8.229 | 76.012 ± 22.639 | 0.779 ± 0.018 | 0.767 ± 0.030 |
| adaptive | 0.817 ± 0.021 | 15.781 ± 2.160 | 18.546 ± 5.113 | 0.795 ± 0.014 | 0.765 ± 0.030 |
| coordinated | 0.817 ± 0.021 | 15.742 ± 2.155 | 16.799 ± 5.043 | 0.793 ± 0.014 | 0.765 ± 0.030 |


By failure scenario:

| scenario | recovery | TIR | role_restore_s |
|---|---|---|---|
| observer | none | 0.760 ± 0.084 | 213.500 ± 93.122 |
| observer | adaptive | 0.865 ± 0.045 | 32.750 ± 6.048 |
| observer | coordinated | 0.871 ± 0.046 | 24.500 ± 4.936 |
| relay | none | 0.838 ± 0.078 | 68.500 ± 75.482 |
| relay | adaptive | 0.884 ± 0.050 | 7.875 ± 2.316 |
| relay | coordinated | 0.884 ± 0.049 | 7.875 ± 2.316 |
| gateway | none | 0.878 ± 0.054 | 12.125 ± 9.253 |
| gateway | adaptive | 0.885 ± 0.049 | 7.875 ± 3.336 |
| gateway | coordinated | 0.883 ± 0.049 | 7.375 ± 2.482 |
| successive | none | 0.792 ± 0.091 | 111.875 ± 61.580 |
| successive | adaptive | 0.829 ± 0.071 | 44.562 ± 31.926 |
| successive | coordinated | 0.828 ± 0.071 | 44.812 ± 32.001 |
| intermittent+relay | none | 0.724 ± 0.069 | 33.125 ± 52.796 |
| intermittent+relay | adaptive | 0.775 ± 0.041 | 8.000 ± 3.111 |
| intermittent+relay | coordinated | 0.772 ± 0.042 | 8.000 ± 3.111 |
| blackout+gateway | none | 0.678 ± 0.060 | 9.625 ± 6.827 |
| blackout+gateway | adaptive | 0.695 ± 0.050 | 9.625 ± 6.827 |
| blackout+gateway | coordinated | 0.695 ± 0.050 | 9.625 ± 6.827 |
| multiple | none | 0.768 ± 0.058 | 85.778 ± 40.366 |
| multiple | adaptive | 0.776 ± 0.056 | 19.333 ± 6.009 |
| multiple | coordinated | 0.772 ± 0.057 | 14.944 ± 4.956 |


By configuration:

| cfg | recovery | TIR | role_restore_s |
|---|---|---|---|
| 1c/3u/1m | none | 0.742 ± 0.114 | 178.625 ± 87.652 |
| 1c/3u/1m | adaptive | 0.907 ± 0.073 | 34.250 ± 21.602 |
| 1c/3u/1m | coordinated | 0.913 ± 0.072 | 30.500 ± 21.555 |
| 5c/5u/3m | none | 0.709 ± 0.030 | 69.036 ± 27.385 |
| 5c/5u/3m | adaptive | 0.696 ± 0.025 | 21.250 ± 5.387 |
| 5c/5u/3m | coordinated | 0.692 ± 0.025 | 19.310 ± 5.254 |
| 10c/9u/4m | none | 0.786 ± 0.018 | 59.357 ± 27.385 |
| 10c/9u/4m | adaptive | 0.809 ± 0.018 | 13.143 ± 3.056 |
| 10c/9u/4m | coordinated | 0.806 ± 0.018 | 12.917 ± 2.961 |
| 10c/15u/4m | none | 0.866 ± 0.019 | 11.690 ± 6.143 |
| 10c/15u/4m | adaptive | 0.869 ± 0.020 | 7.786 ± 2.064 |
| 10c/15u/4m | coordinated | 0.870 ± 0.020 | 6.429 ± 1.033 |


Paired tests (matched seed × scenario × configuration):

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| coordinated vs none | TIR | 270 | 0.817 | 0.777 | +0.039 | 51% | 0.00076 | **supported** |
| coordinated vs none | role_restore_s | 270 | 16.799 | 76.012 | -59.213 | 25% | 3.4e-09 | **supported** |
| coordinated vs none | AoI_s | 270 | 15.742 | 34.319 | -18.577 | 55% | 7.3e-08 | **supported** |
| coordinated vs adaptive | TIR | 270 | 0.817 | 0.817 | -0.000 | 24% | 0.092 | n.s. |
| coordinated vs adaptive | role_restore_s | 270 | 16.799 | 18.546 | -1.747 | 14% | 0.0034 | **supported** |
| coordinated vs adaptive | AoI_s | 270 | 15.742 | 15.781 | -0.039 | 29% | 0.94 | n.s. |


## X2 — Obstacle-aware coordination (Condition 3)

| obstacles | positioning | TIR | coverage | collisions | violations | uav_travel_km |
|---|---|---|---|---|---|---|
| clear | static | 0.913 ± 0.014 | 0.868 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 26.495 ± 8.320 |
| clear | reactive | 0.912 ± 0.015 | 0.867 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 31.989 ± 11.177 |
| clear | adaptive | 0.912 ± 0.015 | 0.867 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 31.989 ± 11.177 |
| static | static | 0.551 ± 0.163 | 0.543 ± 0.148 | 7.700 ± 1.804 | 10.100 ± 4.875 | 13.408 ± 2.071 |
| static | reactive | 0.905 ± 0.020 | 0.850 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 28.481 ± 8.039 |
| static | adaptive | 0.914 ± 0.013 | 0.851 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 29.304 ± 9.836 |
| dynamic | static | 0.412 ± 0.155 | 0.385 ± 0.144 | 10.450 ± 1.793 | 0.000 ± 0.000 | 13.199 ± 4.165 |
| dynamic | reactive | 0.905 ± 0.013 | 0.856 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 40.960 ± 11.806 |
| dynamic | adaptive | 0.905 ± 0.013 | 0.856 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 40.960 ± 11.806 |
| mixed | static | 0.238 ± 0.143 | 0.235 ± 0.129 | 10.550 ± 1.887 | 6.100 ± 3.263 | 9.525 ± 2.997 |
| mixed | reactive | 0.875 ± 0.036 | 0.848 ± 0.020 | 0.600 ± 0.576 | 0.000 ± 0.000 | 34.683 ± 8.255 |
| mixed | adaptive | 0.880 ± 0.030 | 0.848 ± 0.020 | 0.650 ± 0.574 | 0.000 ± 0.000 | 34.324 ± 8.410 |


Paired tests on obstacle maps (static, dynamic, mixed):

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| adaptive vs static | TIR | 60 | 0.900 | 0.400 | +0.500 | 97% | 2e-11 | **supported** |
| adaptive vs static | collisions | 60 | 0.217 | 9.567 | -9.350 | 98% | 1.4e-11 | **supported** |
| adaptive vs reactive | TIR | 60 | 0.900 | 0.895 | +0.005 | 27% | 0.13 | n.s. |
| adaptive vs reactive | collisions | 60 | 0.217 | 0.200 | +0.017 | 0% | 0.8 | n.s. |


## X3 — Five-/ten-casualty scalability (Condition 4)


5 casualties (TIR / coverage / minimum-patient TIR):

| UAVs | fixed | coverage | connected | balanced |
|---|---|---|---|---|
| 3 | 0.419 | 0.637 | 0.523 | 0.564 |
| 5 | 0.735 | 0.868 | 0.828 | 0.829 |
| 7 | 0.905 | 0.886 | 0.888 | 0.907 |
| 9 | 0.926 | 0.899 | 0.899 | 0.924 |
| 11 | 0.930 | 0.902 | 0.902 | 0.921 |
| 13 | 0.924 | 0.903 | 0.903 | 0.927 |
| 15 | 0.929 | 0.907 | 0.907 | 0.923 |


Smallest fleet reaching TIR ≥ 0.80: fixed: 7, coverage: 5, connected: 5, balanced: 5


10 casualties (TIR / coverage / minimum-patient TIR):

| UAVs | fixed | coverage | connected | balanced |
|---|---|---|---|---|
| 3 | 0.323 | 0.459 | 0.434 | 0.422 |
| 6 | 0.651 | 0.764 | 0.755 | 0.751 |
| 9 | 0.807 | 0.860 | 0.858 | 0.864 |
| 12 | 0.915 | 0.896 | 0.896 | 0.911 |
| 15 | 0.913 | 0.896 | 0.896 | 0.906 |
| 18 | 0.920 | 0.899 | 0.899 | 0.911 |
| 21 | 0.917 | 0.901 | 0.901 | 0.914 |
| 24 | 0.920 | 0.905 | 0.905 | 0.915 |
| 27 | 0.920 | 0.903 | 0.903 | 0.912 |
| 30 | 0.922 | 0.902 | 0.902 | 0.914 |


Smallest fleet reaching TIR ≥ 0.80: fixed: 9, coverage: 9, connected: 9, balanced: 9


Paired tests (all fleet sizes):

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| balanced vs fixed | TIR | 136 | 0.848 | 0.822 | +0.026 | 50% | 0.028 | **supported** |
| balanced vs fixed | min_patient_TIR | 136 | 0.721 | 0.680 | +0.041 | 45% | 0.011 | **supported** |
| balanced vs coverage | TIR | 136 | 0.848 | 0.846 | +0.002 | 62% | 0.0071 | **supported** |
| balanced vs coverage | min_patient_TIR | 136 | 0.721 | 0.717 | +0.003 | 51% | 0.079 | n.s. |
| balanced vs connected | TIR | 136 | 0.848 | 0.835 | +0.013 | 70% | 1.1e-08 | **supported** |
| balanced vs connected | min_patient_TIR | 136 | 0.721 | 0.701 | +0.020 | 54% | 0.0019 | **supported** |


## X4 — Communication, priority and fairness (E6a/E6b)

| variant | TIR | TIR_critical | min_patient_TIR | jain | starvation | AoI_s |
|---|---|---|---|---|---|---|
| shortest+guarded | 0.751 ± 0.017 | 0.643 ± 0.025 | 0.592 ± 0.026 | 0.980 ± 0.004 | 0.042 ± 0.003 | 12.953 ± 0.480 |
| etx+guarded | 0.882 ± 0.011 | 0.748 ± 0.023 | 0.757 ± 0.027 | 0.991 ± 0.003 | 0.033 ± 0.003 | 9.331 ± 0.375 |
| queue+fifo | 0.931 ± 0.009 | 0.784 ± 0.023 | 0.809 ± 0.027 | 0.993 ± 0.002 | 0.034 ± 0.003 | 9.416 ± 0.450 |
| queue+sampled | 0.929 ± 0.010 | 0.784 ± 0.023 | 0.809 ± 0.028 | 0.993 ± 0.002 | 0.034 ± 0.003 | 9.366 ± 0.438 |
| queue+deadline | 0.931 ± 0.008 | 0.785 ± 0.023 | 0.810 ± 0.027 | 0.993 ± 0.002 | 0.034 ± 0.003 | 9.404 ± 0.433 |
| queue+deficit | 0.930 ± 0.010 | 0.781 ± 0.024 | 0.804 ± 0.029 | 0.993 ± 0.002 | 0.034 ± 0.003 | 9.379 ± 0.438 |
| queue+guarded | 0.888 ± 0.010 | 0.753 ± 0.023 | 0.765 ± 0.026 | 0.991 ± 0.002 | 0.034 ± 0.003 | 9.240 ± 0.381 |


Degraded links (link_degradation = 0.2):

| variant | TIR | TIR_critical | min_patient_TIR | starvation |
|---|---|---|---|---|
| shortest+guarded | 0.720 ± 0.020 | 0.615 ± 0.032 | 0.568 ± 0.033 | 0.045 ± 0.005 |
| etx+guarded | 0.867 ± 0.016 | 0.742 ± 0.035 | 0.738 ± 0.035 | 0.035 ± 0.004 |
| queue+fifo | 0.928 ± 0.014 | 0.785 ± 0.034 | 0.809 ± 0.036 | 0.036 ± 0.005 |
| queue+sampled | 0.928 ± 0.016 | 0.780 ± 0.033 | 0.806 ± 0.038 | 0.035 ± 0.005 |
| queue+deadline | 0.930 ± 0.013 | 0.786 ± 0.034 | 0.811 ± 0.036 | 0.036 ± 0.005 |
| queue+deficit | 0.927 ± 0.016 | 0.784 ± 0.036 | 0.806 ± 0.038 | 0.035 ± 0.005 |
| queue+guarded | 0.874 ± 0.015 | 0.739 ± 0.032 | 0.758 ± 0.026 | 0.035 ± 0.005 |


Paired tests:

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| queue+guarded vs shortest+guarded | TIR | 60 | 0.888 | 0.751 | +0.138 | 100% | 1.6e-11 | **supported** |
| queue+guarded vs shortest+guarded | TIR_critical | 60 | 0.753 | 0.643 | +0.110 | 100% | 1.6e-11 | **supported** |
| queue+guarded vs shortest+guarded | min_patient_TIR | 60 | 0.765 | 0.592 | +0.173 | 95% | 2e-11 | **supported** |
| queue+guarded vs queue+fifo | TIR | 60 | 0.888 | 0.931 | -0.042 | 0% | 1.6e-11 | worse |
| queue+guarded vs queue+fifo | TIR_critical | 60 | 0.753 | 0.784 | -0.031 | 12% | 2.7e-09 | worse |
| queue+guarded vs queue+fifo | min_patient_TIR | 60 | 0.765 | 0.809 | -0.044 | 2% | 6.5e-10 | worse |
| queue+guarded vs queue+deadline | TIR | 60 | 0.888 | 0.931 | -0.043 | 0% | 1.6e-11 | worse |
| queue+guarded vs queue+deadline | TIR_critical | 60 | 0.753 | 0.785 | -0.032 | 12% | 2.7e-09 | worse |
| queue+guarded vs queue+deadline | min_patient_TIR | 60 | 0.765 | 0.810 | -0.045 | 2% | 6.5e-10 | worse |
| queue+guarded vs queue+deficit | TIR | 60 | 0.888 | 0.930 | -0.042 | 0% | 1.6e-11 | worse |
| queue+guarded vs queue+deficit | TIR_critical | 60 | 0.753 | 0.781 | -0.028 | 15% | 7.9e-08 | worse |
| queue+guarded vs queue+deficit | min_patient_TIR | 60 | 0.765 | 0.804 | -0.039 | 5% | 7.8e-09 | worse |


## X5 — Severity-aware medic allocation with medic-safe routing (Condition 6)

Severity is computed only from vitals (RTS + shock index + SpO2) carried in delivered UAV packets. Mixed static/dynamic hazards, changing severity, 8 matched seeds × 10 configurations.

| route | alloc | sev_w_survival | completion | T1_golden | medic_incap | medic_exposure | treatment_aborts |
|---|---|---|---|---|---|---|---|
| shortest | nearest | 0.632 ± 0.075 | 0.586 ± 0.084 | 0.332 ± 0.067 | 3.212 ± 0.612 | 109.600 ± 14.357 | 0.000 ± 0.000 |
| safe | nearest | 0.617 ± 0.054 | 0.591 ± 0.052 | 0.216 ± 0.057 | 0.087 ± 0.063 | 41.860 ± 6.098 | 0.150 ± 0.087 |
| shortest | severity | 0.574 ± 0.078 | 0.403 ± 0.087 | 0.278 ± 0.061 | 3.562 ± 0.582 | 102.666 ± 13.189 | 0.000 ± 0.000 |
| safe | severity | 0.619 ± 0.054 | 0.552 ± 0.057 | 0.200 ± 0.054 | 0.075 ± 0.059 | 36.873 ± 4.935 | 0.163 ± 0.109 |
| shortest | urgency | 0.613 ± 0.076 | 0.471 ± 0.084 | 0.325 ± 0.067 | 3.475 ± 0.633 | 104.713 ± 12.340 | 0.000 ± 0.000 |
| safe | urgency | 0.632 ± 0.053 | 0.586 ± 0.053 | 0.204 ± 0.053 | 0.075 ± 0.059 | 38.747 ± 5.011 | 0.188 ± 0.107 |
| shortest | load | 0.534 ± 0.081 | 0.345 ± 0.084 | 0.243 ± 0.066 | 3.575 ± 0.614 | 96.034 ± 12.181 | 0.000 ± 0.000 |
| safe | load | 0.622 ± 0.059 | 0.549 ± 0.063 | 0.192 ± 0.054 | 0.100 ± 0.067 | 35.509 ± 4.727 | 0.300 ± 0.333 |
| shortest | gain | 0.673 ± 0.077 | 0.536 ± 0.087 | 0.367 ± 0.071 | 3.275 ± 0.599 | 112.815 ± 11.439 | 0.000 ± 0.000 |
| safe | gain | 0.626 ± 0.052 | 0.571 ± 0.055 | 0.205 ± 0.053 | 0.075 ± 0.059 | 34.869 ± 4.998 | 0.312 ± 0.335 |


Paired tests — allocation (all with safe routing):

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| urgency vs nearest | sev_w_survival | 80 | 0.632 | 0.617 | +0.015 | 15% | 0.094 | n.s. |
| urgency vs nearest | completion | 80 | 0.586 | 0.591 | -0.005 | 14% | 0.93 | n.s. |
| urgency vs severity | sev_w_survival | 80 | 0.632 | 0.619 | +0.013 | 15% | 0.35 | n.s. |
| urgency vs severity | completion | 80 | 0.586 | 0.552 | +0.034 | 25% | 0.0036 | **supported** |
| urgency vs load | sev_w_survival | 80 | 0.632 | 0.622 | +0.010 | 15% | 0.68 | n.s. |
| urgency vs load | completion | 80 | 0.586 | 0.549 | +0.037 | 29% | 0.013 | **supported** |
| urgency vs gain | sev_w_survival | 80 | 0.632 | 0.626 | +0.006 | 9% | 0.57 | n.s. |
| urgency vs gain | completion | 80 | 0.586 | 0.571 | +0.015 | 14% | 0.41 | n.s. |


Paired tests — medic safety (urgency-distance allocation, safe vs shortest route):

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| safe vs shortest | medic_incap | 80 | 0.075 | 3.475 | -3.400 | 92% | 9.5e-15 | **supported** |
| safe vs shortest | medic_exposure | 80 | 38.747 | 104.713 | -65.967 | 92% | 1.4e-13 | **supported** |
| safe vs shortest | sev_w_survival | 80 | 0.632 | 0.613 | +0.019 | 39% | 0.55 | n.s. |


## All 82 base configurations (Condition 6 table)

Each configuration × 2 seeds, mixed hazards, mixed casualty mobility, changing severity.

| group | stack | sev_w_survival | completion | TIR | medic_incap |
|---|---|---|---|---|---|
| 1 casualties | independent | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.107 ± 0.788 | 1.000 ± 0.000 |
| 1 casualties | independent+safe | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.726 ± 1.000 | 0.000 ± 0.000 |
| 1 casualties | MANTIS-H | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.678 ± 0.965 | 0.000 ± 0.000 |
| 5 casualties | independent | 0.447 ± 0.114 | 0.433 ± 0.128 | 0.465 ± 0.085 | 2.333 ± 0.509 |
| 5 casualties | independent+safe | 0.449 ± 0.060 | 0.419 ± 0.092 | 0.750 ± 0.049 | 0.024 ± 0.048 |
| 5 casualties | MANTIS-H | 0.494 ± 0.072 | 0.452 ± 0.104 | 0.760 ± 0.049 | 0.119 ± 0.102 |
| 10 casualties | independent | 0.522 ± 0.042 | 0.395 ± 0.064 | 0.569 ± 0.053 | 4.017 ± 0.511 |
| 10 casualties | independent+safe | 0.705 ± 0.023 | 0.633 ± 0.039 | 0.824 ± 0.028 | 0.092 ± 0.052 |
| 10 casualties | MANTIS-H | 0.708 ± 0.024 | 0.643 ± 0.040 | 0.809 ± 0.030 | 0.083 ± 0.050 |


Paired tests over all 82 configurations:

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| MANTIS-H vs independent | sev_w_survival | 164 | 0.644 | 0.496 | +0.148 | 71% | 3.2e-14 | **supported** |
| MANTIS-H vs independent | TIR | 164 | 0.795 | 0.537 | +0.259 | 87% | 2.3e-23 | **supported** |
| MANTIS-H vs independent | medic_incap | 164 | 0.091 | 3.549 | -3.457 | 91% | 2.7e-28 | **supported** |
| MANTIS-H vs independent | mission_score | 164 | 2.504 | -1.801 | +4.305 | 95% | 6.2e-28 | **supported** |
| MANTIS-H vs independent+safe | sev_w_survival | 164 | 0.644 | 0.631 | +0.014 | 15% | 0.36 | n.s. |
| MANTIS-H vs independent+safe | TIR | 164 | 0.795 | 0.804 | -0.009 | 32% | 0.00023 | worse |
| MANTIS-H vs independent+safe | medic_incap | 164 | 0.091 | 0.073 | +0.018 | 1% | 0.65 | n.s. |
| MANTIS-H vs independent+safe | mission_score | 164 | 2.504 | 2.484 | +0.020 | 39% | 0.18 | n.s. |


## X6 — Casualty mobility and dynamic medic replanning (Condition 5, E7c)

| replan | sev_w_survival | completion | reassignments | notfound | medic_travel_km |
|---|---|---|---|---|---|
| static | 0.604 ± 0.052 | 0.582 ± 0.050 | 0.000 ± 0.000 | 2.875 ± 0.431 | 3.686 ± 0.265 |
| periodic | 0.662 ± 0.048 | 0.642 ± 0.041 | 18.100 ± 2.585 | 0.738 ± 0.391 | 3.289 ± 0.220 |
| event | 0.656 ± 0.046 | 0.639 ± 0.044 | 26.438 ± 4.824 | 0.775 ± 0.326 | 3.268 ± 0.223 |
| selective | 0.642 ± 0.046 | 0.624 ± 0.042 | 7.750 ± 1.169 | 1.075 ± 0.359 | 3.421 ± 0.251 |
| deadline | 0.648 ± 0.046 | 0.628 ± 0.042 | 9.287 ± 1.462 | 1.125 ± 0.370 | 3.425 ± 0.246 |


By mobility pattern (severity-weighted survival):

| mobility | static | periodic | event | selective | deadline |
|---|---|---|---|---|---|
| abrupt | 0.539 | 0.640 | 0.632 | 0.618 | 0.618 |
| intermittent | 0.628 | 0.670 | 0.670 | 0.638 | 0.638 |
| mixed | 0.608 | 0.678 | 0.653 | 0.647 | 0.647 |
| slow | 0.624 | 0.663 | 0.679 | 0.679 | 0.679 |
| stationary | 0.623 | 0.657 | 0.647 | 0.629 | 0.658 |


Paired tests:

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| periodic vs static | sev_w_survival | 80 | 0.662 | 0.604 | +0.058 | 29% | 0.0074 | **supported** |
| periodic vs static | reassignments | 80 | 18.100 | 0.000 | +18.100 | 0% | 8.2e-15 | worse |
| periodic vs event | sev_w_survival | 80 | 0.662 | 0.656 | +0.005 | 9% | 0.44 | n.s. |
| periodic vs event | reassignments | 80 | 18.100 | 26.438 | -8.338 | 65% | 1.2e-06 | **supported** |
| periodic vs selective | sev_w_survival | 80 | 0.662 | 0.642 | +0.020 | 16% | 0.24 | n.s. |
| periodic vs selective | reassignments | 80 | 18.100 | 7.750 | +10.350 | 9% | 7e-13 | worse |
| periodic vs deadline | sev_w_survival | 80 | 0.662 | 0.648 | +0.014 | 15% | 0.44 | n.s. |
| periodic vs deadline | reassignments | 80 | 18.100 | 9.287 | +8.812 | 11% | 8.8e-12 | worse |


## X7 — Energy-aware UAV operation (E8a)

Heterogeneous batteries (0.6–1.25× nominal, nominal endurance 12.5 min < 20 min mission), 3 spare UAVs at base.

| env | energy_policy | TIR | stranded | safe_return | endurance_min | energy_per_timely | coverage |
|---|---|---|---|---|---|---|---|
| clear | fixed | 0.888 ± 0.023 | 12.250 ± 2.267 | 0.000 ± 0.000 | 11.694 ± 0.247 | 0.030 ± 0.003 | 0.838 ± 0.015 |
| clear | reactive | 0.903 ± 0.018 | 1.100 ± 0.642 | 0.936 ± 0.035 | 16.008 ± 0.447 | 0.042 ± 0.004 | 0.859 ± 0.014 |
| clear | aware | 0.905 ± 0.018 | 0.000 ± 0.000 | 1.000 ± 0.000 | 17.761 ± 0.174 | 0.048 ± 0.005 | 0.855 ± 0.014 |
| obstacles+mobility | fixed | 0.817 ± 0.045 | 12.250 ± 2.312 | 0.000 ± 0.000 | 11.499 ± 0.244 | 0.037 ± 0.005 | 0.758 ± 0.045 |
| obstacles+mobility | reactive | 0.828 ± 0.047 | 1.350 ± 0.631 | 0.923 ± 0.035 | 15.875 ± 0.386 | 0.053 ± 0.007 | 0.764 ± 0.045 |
| obstacles+mobility | aware | 0.839 ± 0.041 | 0.000 ± 0.000 | 1.000 ± 0.000 | 17.747 ± 0.198 | 0.059 ± 0.008 | 0.767 ± 0.042 |


Paired tests:

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| aware vs fixed | TIR | 40 | 0.872 | 0.852 | +0.019 | 72% | 0.00042 | **supported** |
| aware vs fixed | stranded | 40 | 0.000 | 12.250 | -12.250 | 100% | 3e-08 | **supported** |
| aware vs fixed | coverage | 40 | 0.811 | 0.798 | +0.012 | 57% | 0.037 | **supported** |
| aware vs reactive | TIR | 40 | 0.872 | 0.865 | +0.006 | 50% | 0.24 | n.s. |
| aware vs reactive | stranded | 40 | 0.000 | 1.225 | -1.225 | 57% | 5.8e-06 | **supported** |
| aware vs reactive | coverage | 40 | 0.811 | 0.812 | -0.001 | 32% | 0.24 | n.s. |


## X8 — Joint multi-agent learning

60 held-out evaluation scenarios drawn from the training distribution (random configuration among the 82 with ≤15 UAVs, random obstacles, hazards, mobility, failures, disruption, energy), evaluation seeds never used in training. Greedy policies. All learners: same features, actions, reward, executors and 500 training episodes.

Checkpoint selection (every learner, same protocol): checkpoints saved every ~100 training episodes, each scored on 16 validation scenarios (seeds 2·10⁷+, disjoint from training and test); the best is evaluated below. Episode 0 of BC-initialised runs = the BC policy.


| method | selected checkpoint | validation mission score by checkpoint |
|---|---|---|
| mantis_mappo | 104 | 0: 2.55, 104: 2.57, 200: 2.52, 304: 2.47, 400: 2.47, 504: 2.49 |
| mappo | 304 | 0: 0.54, 104: 0.52, 200: 1.84, 304: 1.95, 400: 1.88, 504: 1.29 |
| ippo | 200 | 0: 1.43, 104: 1.83, 200: 2.20, 304: 1.72, 400: 1.66, 504: 1.79 |
| abl_mediconly | 504 | 0: 2.51, 104: 2.48, 200: 2.53, 304: 2.53, 400: 2.46, 504: 2.65 |
| vdn | 200 | 100: 2.03, 200: 2.41, 300: 1.49, 400: 1.89, 500: 2.06 |
| abl_uavonly | 0 | 0: 2.67, 104: 2.67, 200: 2.67, 304: 2.67, 400: 2.67, 504: 2.67 |
| qmix | 500 | 100: 1.64, 200: 1.43, 300: 1.93, 400: 2.16, 500: 2.39 |


| method | mission_score | sev_w_survival | completion | TIR | medic_incap | AoI_s |
|---|---|---|---|---|---|---|
| Independent-H | 2.266 ± 0.231 | 0.664 ± 0.060 | 0.545 ± 0.066 | 0.676 ± 0.063 | 0.150 ± 0.093 | 50.805 ± 14.705 |
| MANTIS-H | 2.294 ± 0.203 | 0.675 ± 0.054 | 0.567 ± 0.062 | 0.645 ± 0.062 | 0.150 ± 0.104 | 48.385 ± 15.914 |
| bc | 2.278 ± 0.187 | 0.653 ± 0.055 | 0.553 ± 0.056 | 0.646 ± 0.062 | 0.100 ± 0.078 | 51.924 ± 16.754 |
| ippo | 1.740 ± 0.223 | 0.572 ± 0.060 | 0.397 ± 0.065 | 0.378 ± 0.053 | 0.067 ± 0.065 | 137.293 ± 24.675 |
| mappo | 1.541 ± 0.195 | 0.576 ± 0.058 | 0.447 ± 0.058 | 0.186 ± 0.032 | 0.117 ± 0.096 | 213.783 ± 20.078 |
| vdn | 1.930 ± 0.235 | 0.652 ± 0.060 | 0.517 ± 0.068 | 0.409 ± 0.046 | 0.167 ± 0.108 | 93.675 ± 17.782 |
| qmix | 2.152 ± 0.187 | 0.594 ± 0.060 | 0.517 ± 0.058 | 0.610 ± 0.059 | 0.050 ± 0.057 | 48.217 ± 13.577 |
| mantis_mappo | 2.222 ± 0.190 | 0.652 ± 0.054 | 0.552 ± 0.058 | 0.645 ± 0.061 | 0.133 ± 0.101 | 51.193 ± 15.974 |


Paired tests (matched scenarios):

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| mantis_mappo vs Independent-H | mission_score | 60 | 2.222 | 2.266 | -0.045 | 35% | 0.09 | n.s. |
| mantis_mappo vs Independent-H | sev_w_survival | 60 | 0.652 | 0.664 | -0.012 | 13% | 0.45 | n.s. |
| mantis_mappo vs Independent-H | TIR | 60 | 0.645 | 0.676 | -0.031 | 20% | 3.6e-05 | worse |
| mantis_mappo vs MANTIS-H | mission_score | 60 | 2.222 | 2.294 | -0.072 | 37% | 0.06 | n.s. |
| mantis_mappo vs MANTIS-H | sev_w_survival | 60 | 0.652 | 0.675 | -0.023 | 8% | 0.28 | n.s. |
| mantis_mappo vs MANTIS-H | TIR | 60 | 0.645 | 0.645 | -0.001 | 50% | 0.77 | n.s. |
| mantis_mappo vs bc | mission_score | 60 | 2.222 | 2.278 | -0.056 | 22% | 0.056 | n.s. |
| mantis_mappo vs bc | sev_w_survival | 60 | 0.652 | 0.653 | -0.001 | 5% | 0.83 | n.s. |
| mantis_mappo vs bc | TIR | 60 | 0.645 | 0.646 | -0.002 | 28% | 0.61 | n.s. |
| mantis_mappo vs ippo | mission_score | 60 | 2.222 | 1.740 | +0.482 | 82% | 3.9e-07 | **supported** |
| mantis_mappo vs ippo | sev_w_survival | 60 | 0.652 | 0.572 | +0.080 | 40% | 0.00067 | **supported** |
| mantis_mappo vs ippo | TIR | 60 | 0.645 | 0.378 | +0.266 | 97% | 2.3e-11 | **supported** |
| mantis_mappo vs mappo | mission_score | 60 | 2.222 | 1.541 | +0.680 | 90% | 5.7e-10 | **supported** |
| mantis_mappo vs mappo | sev_w_survival | 60 | 0.652 | 0.576 | +0.076 | 42% | 0.0022 | **supported** |
| mantis_mappo vs mappo | TIR | 60 | 0.645 | 0.186 | +0.458 | 100% | 1.6e-11 | **supported** |
| mantis_mappo vs vdn | mission_score | 60 | 2.222 | 1.930 | +0.292 | 73% | 0.0012 | **supported** |
| mantis_mappo vs vdn | sev_w_survival | 60 | 0.652 | 0.652 | -0.001 | 17% | 0.74 | n.s. |
| mantis_mappo vs vdn | TIR | 60 | 0.645 | 0.409 | +0.236 | 93% | 3.3e-11 | **supported** |
| mantis_mappo vs qmix | mission_score | 60 | 2.222 | 2.152 | +0.069 | 60% | 0.37 | n.s. |
| mantis_mappo vs qmix | sev_w_survival | 60 | 0.652 | 0.594 | +0.058 | 37% | 0.0031 | **supported** |
| mantis_mappo vs qmix | TIR | 60 | 0.645 | 0.610 | +0.035 | 75% | 1e-05 | **supported** |


## X9 — Full six-condition robustness

Compounded: gateway→relay→observer failures (200/400/600 s), mixed static+dynamic obstacles, mixed casualty mobility, 20% link degradation + intermittent disruption, changing severity, mixed hazards, low/heterogeneous batteries (endurance 9 min) with 2 spares. 20 seeds × 3 configurations.

| method | mission_score | sev_w_survival | TIR | medic_incap | stranded | collisions |
|---|---|---|---|---|---|---|
| Independent-H | 0.991 ± 0.334 | 0.683 ± 0.058 | 0.677 ± 0.031 | 0.083 ± 0.072 | 7.350 ± 1.127 | 0.333 ± 0.148 |
| MANTIS-H | 2.335 ± 0.224 | 0.695 ± 0.057 | 0.638 ± 0.029 | 0.117 ± 0.084 | 0.200 ± 0.141 | 0.567 ± 0.262 |
| ippo | 1.464 ± 0.246 | 0.562 ± 0.061 | 0.428 ± 0.038 | 0.117 ± 0.096 | 1.033 ± 0.365 | 0.550 ± 0.220 |
| mappo | 1.288 ± 0.249 | 0.545 ± 0.066 | 0.215 ± 0.029 | 0.100 ± 0.078 | 1.083 ± 0.369 | 0.600 ± 0.298 |
| qmix | 1.866 ± 0.248 | 0.609 ± 0.060 | 0.599 ± 0.030 | 0.117 ± 0.084 | 1.133 ± 0.410 | 0.600 ± 0.224 |
| mantis_mappo | 2.011 ± 0.251 | 0.645 ± 0.062 | 0.649 ± 0.027 | 0.167 ± 0.108 | 0.983 ± 0.345 | 0.467 ± 0.235 |


Paired tests:

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| mantis_mappo vs Independent-H | mission_score | 60 | 2.011 | 0.991 | +1.020 | 88% | 1.9e-08 | **supported** |
| mantis_mappo vs Independent-H | sev_w_survival | 60 | 0.645 | 0.683 | -0.038 | 10% | 0.024 | worse |
| mantis_mappo vs MANTIS-H | mission_score | 60 | 2.011 | 2.335 | -0.325 | 32% | 0.00048 | worse |
| mantis_mappo vs MANTIS-H | sev_w_survival | 60 | 0.645 | 0.695 | -0.050 | 15% | 0.013 | worse |
| mantis_mappo vs ippo | mission_score | 60 | 2.011 | 1.464 | +0.546 | 90% | 6.9e-09 | **supported** |
| mantis_mappo vs ippo | sev_w_survival | 60 | 0.645 | 0.562 | +0.084 | 52% | 0.00013 | **supported** |
| mantis_mappo vs mappo | mission_score | 60 | 2.011 | 1.288 | +0.722 | 90% | 3.4e-09 | **supported** |
| mantis_mappo vs mappo | sev_w_survival | 60 | 0.645 | 0.545 | +0.100 | 52% | 4.6e-05 | **supported** |
| mantis_mappo vs qmix | mission_score | 60 | 2.011 | 1.866 | +0.145 | 65% | 0.053 | n.s. |
| mantis_mappo vs qmix | sev_w_survival | 60 | 0.645 | 0.609 | +0.036 | 37% | 0.029 | **supported** |

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| MANTIS-H vs Independent-H | mission_score | 60 | 2.335 | 0.991 | +1.345 | 93% | 8e-11 | **supported** |
| MANTIS-H vs Independent-H | sev_w_survival | 60 | 0.695 | 0.683 | +0.012 | 17% | 0.39 | n.s. |
| MANTIS-H vs Independent-H | TIR | 60 | 0.638 | 0.677 | -0.039 | 22% | 6e-06 | worse |
| MANTIS-H vs Independent-H | medic_incap | 60 | 0.117 | 0.083 | +0.033 | 5% | 0.65 | n.s. |


## X10 — Ablation and generalisation

(a) MANTIS-H component ablation on compounded scenarios (10c/15u/4m, 20 seeds):

| variant | mission_score | sev_w_survival | TIR | medic_incap | stranded |
|---|---|---|---|---|---|
| MANTIS-H | 2.314 ± 0.381 | 0.680 ± 0.112 | 0.600 ± 0.034 | 0.100 ± 0.144 | 0.050 ± 0.105 |
| -balanced UAV | 1.997 ± 0.445 | 0.682 ± 0.110 | 0.599 ± 0.037 | 0.400 ± 0.280 | 0.000 ± 0.000 |
| -coordinated recovery | 2.221 ± 0.441 | 0.673 ± 0.113 | 0.601 ± 0.035 | 0.100 ± 0.144 | 0.050 ± 0.105 |
| -adaptive positioning | 2.172 ± 0.375 | 0.687 ± 0.103 | 0.598 ± 0.039 | 0.200 ± 0.192 | 0.150 ± 0.171 |
| -queue routing | 2.087 ± 0.456 | 0.680 ± 0.103 | 0.523 ± 0.034 | 0.200 ± 0.245 | 0.100 ± 0.209 |
| -guarded scheduling | 2.370 ± 0.401 | 0.700 ± 0.112 | 0.692 ± 0.056 | 0.200 ± 0.192 | 0.050 ± 0.105 |
| -urgency allocation (→nearest) | 2.265 ± 0.386 | 0.677 ± 0.106 | 0.619 ± 0.035 | 0.150 ± 0.171 | 0.050 ± 0.105 |
| -safe routing | -1.052 ± 0.820 | 0.683 ± 0.121 | 0.539 ± 0.079 | 3.150 ± 0.437 | 0.000 ± 0.000 |
| -periodic replanning (→static) | 1.941 ± 0.421 | 0.562 ± 0.098 | 0.575 ± 0.043 | 0.100 ± 0.209 | 0.100 ± 0.144 |
| -energy-aware | 0.726 ± 0.534 | 0.684 ± 0.107 | 0.546 ± 0.040 | 0.200 ± 0.245 | 7.450 ± 0.978 |

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| MANTIS-H vs -balanced UAV | mission_score | 20 | 2.314 | 1.997 | +0.317 | 65% | 0.044 | **supported** |
| MANTIS-H vs -coordinated recovery | mission_score | 20 | 2.314 | 2.221 | +0.093 | 35% | 0.94 | n.s. |
| MANTIS-H vs -adaptive positioning | mission_score | 20 | 2.314 | 2.172 | +0.142 | 55% | 0.47 | n.s. |
| MANTIS-H vs -queue routing | mission_score | 20 | 2.314 | 2.087 | +0.227 | 75% | 0.044 | **supported** |
| MANTIS-H vs -guarded scheduling | mission_score | 20 | 2.314 | 2.370 | -0.056 | 45% | 0.33 | n.s. |
| MANTIS-H vs -urgency allocation (→nearest) | mission_score | 20 | 2.314 | 2.265 | +0.048 | 45% | 0.99 | n.s. |
| MANTIS-H vs -safe routing | mission_score | 20 | 2.314 | -1.052 | +3.366 | 95% | 3.8e-06 | **supported** |
| MANTIS-H vs -periodic replanning (→static) | mission_score | 20 | 2.314 | 1.941 | +0.373 | 85% | 0.0032 | **supported** |
| MANTIS-H vs -energy-aware | mission_score | 20 | 2.314 | 0.726 | +1.588 | 100% | 1.9e-06 | **supported** |


(b) Learning ablation (X8 suite). `abl_mediconly` = learned medics + heuristic UAVs; `abl_uavonly` = learned UAVs + heuristic medics:

| method | mission_score | sev_w_survival | TIR | medic_incap |
|---|---|---|---|---|
| MANTIS-H | 2.294 ± 0.203 | 0.675 ± 0.054 | 0.645 ± 0.062 | 0.150 ± 0.104 |
| bc | 2.278 ± 0.187 | 0.653 ± 0.055 | 0.646 ± 0.062 | 0.100 ± 0.078 |
| abl_mediconly | 2.324 ± 0.188 | 0.651 ± 0.054 | 0.654 ± 0.060 | 0.083 ± 0.072 |
| abl_uavonly | 2.341 ± 0.183 | 0.677 ± 0.051 | 0.648 ± 0.061 | 0.100 ± 0.091 |
| mantis_mappo | 2.222 ± 0.190 | 0.652 ± 0.054 | 0.645 ± 0.061 | 0.133 ± 0.101 |

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| mantis_mappo vs MANTIS-H | mission_score | 60 | 2.222 | 2.294 | -0.072 | 37% | 0.06 | n.s. |
| mantis_mappo vs bc | mission_score | 60 | 2.222 | 2.278 | -0.056 | 22% | 0.056 | n.s. |
| mantis_mappo vs abl_mediconly | mission_score | 60 | 2.222 | 2.324 | -0.102 | 35% | 0.038 | worse |
| mantis_mappo vs abl_uavonly | mission_score | 60 | 2.222 | 2.341 | -0.119 | 37% | 0.043 | worse |


(c) Generalisation to unseen fleet sizes (18–30 UAVs; training used ≤15) and unseen maps (1.3 km, 5 hazards):

| gen | method | mission_score | sev_w_survival | TIR | medic_incap |
|---|---|---|---|---|---|
| large-fleet | Independent-H | 2.439 ± 0.332 | 0.630 ± 0.087 | 0.865 ± 0.022 | 0.233 ± 0.234 |
| large-fleet | MANTIS-H | 2.569 ± 0.274 | 0.638 ± 0.085 | 0.869 ± 0.019 | 0.133 ± 0.129 |
| large-fleet | mappo | 1.538 ± 0.310 | 0.485 ± 0.088 | 0.311 ± 0.054 | 0.133 ± 0.129 |
| large-fleet | mantis_mappo | 2.534 ± 0.262 | 0.624 ± 0.081 | 0.864 ± 0.018 | 0.133 ± 0.129 |
| unseen-map | Independent-H | 2.588 ± 0.338 | 0.701 ± 0.104 | 0.822 ± 0.053 | 0.100 ± 0.226 |
| unseen-map | MANTIS-H | 2.744 ± 0.299 | 0.722 ± 0.124 | 0.804 ± 0.048 | 0.000 ± 0.000 |
| unseen-map | mappo | 1.573 ± 0.383 | 0.566 ± 0.138 | 0.145 ± 0.063 | 0.000 ± 0.000 |
| unseen-map | mantis_mappo | 2.576 ± 0.277 | 0.668 ± 0.103 | 0.794 ± 0.043 | 0.000 ± 0.000 |

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| mantis_mappo vs MANTIS-H | mission_score | 30 | 2.534 | 2.569 | -0.035 | 40% | 0.57 | n.s. |
| mantis_mappo vs mappo | mission_score | 30 | 2.534 | 1.538 | +0.996 | 93% | 4.7e-08 | **supported** |

| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |
|---|---|---|---|---|---|---|---|---|
| mantis_mappo vs MANTIS-H | mission_score | 10 | 2.576 | 2.744 | -0.168 | 10% | 0.027 | worse |
| mantis_mappo vs mappo | mission_score | 10 | 2.576 | 1.573 | +1.003 | 100% | 0.002 | **supported** |
