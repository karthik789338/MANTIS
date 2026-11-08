"""Builds results/RESULTS.md (tables with 95% CIs and paired Wilcoxon tests) and figures."""
import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import summarize, paired_test, fmt, RES, MODELS

FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

out = []


SCORECARD = """
## Hypothesis scorecard (read this first)

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| Safety | Medics are kept safe | X5: safe routing cuts medic incapacitation 3.48→0.075 per mission (−98%, p≈1e-14) with no significant loss in severity-weighted survival (0.632 vs 0.613, p=0.55). X10a: removing safe routing is the most damaging ablation | **Strongly supported** |
| C6 | Severity-aware allocation from vitals | X5b (2,000 matched episodes): urgency-distance (severity ÷ travel time) beats nearest-first at both staging distances: severity-weighted survival +1.2–1.6 pp, T1 (critical) survival +3.0–3.2 pp (p≤0.003). The effect grows with scarcity: 10 casualties/1 medic +5.9 pp (p<0.001); scattered casualties +9 pp T1 survival. Severity-first that ignores distance does not help. Survival-gain beats nearest at 50 m staging. With generous medic ratios (original X5) the rules are tied | **Supported**: severity must be combined with travel time; benefit is largest when medics are scarce |
| H | Joint coordination beats independently managed policies | 82-config grid: MANTIS-H vs the independently managed stack (comm-only UAV policy, medics without safety-aware routing, reactive batteries, FIFO): survival +14.8 pp, TIR +26 pp, medic incapacitation 3.55→0.09 (p<1e-13). X9 compounded stress: mission score 2.34 vs 0.99 (p<1e-10). X10a ablation: the gain comes from safe routing, energy-aware operation, replanning and queue-aware routing; an independent stack given the same safe routing is statistically tied (see 82-config section) | **Supported** |
| C2 | Failure recovery restores casualty awareness | X1: TIR 0.777→0.817, role restoration 76 s→17 s, AoI 34 s→16 s vs no recovery (p<0.001); coordinated vs adaptive: 1.7 s faster restoration (p=0.003) | **Supported** |
| C3 | Obstacle-aware positioning | X2: TIR 0.40→0.90, collisions 9.6→0.2 per mission vs straight-line flight (p<1e-10) | **Supported** |
| E8a | Energy-aware UAV operation | X7: 0 stranded UAVs vs 12.3 (no management) and 1.2 (threshold return), p<1e-5; TIR +1.9 pp vs fixed | **Supported** |
| C5 | Replanning under casualty mobility | X6: periodic replanning beats static (+5.8 pp survival, p=0.007); selective/deadline-selective cut reassignments ~50% at similar survival | **Supported** |
| C4 | Fleet size for timely delivery | X3: TIR ≥ 0.80 needs 5 UAVs for 5 casualties and 9 for 10; balanced > connected and fixed (p<0.03) | **Answered** |
| E6 | Communication routing and scheduling | X4: queue-aware routing raises TIR 0.75→0.93 (p<1e-10). Guarded scheduling lowers TIR vs FIFO (0.888 vs 0.931); FIFO/EDF/deficit tied | **Supported for routing; not for guarded scheduling** |
| X8 | Joint MARL | 500 episodes/method: MANTIS-MAPPO is on par with the heuristics (2.22 vs 2.29, n.s.) and significantly beats MAPPO (+0.68), IPPO (+0.48) and VDN (+0.29) (p≤0.001); vs QMIX n.s. In the compounded X9 setting it is below MANTIS-H (2.01 vs 2.34) | **Beats MARL baselines; on par with heuristics** — large-scale (12k-episode) training pending |

MARL note: with 500 episodes, RL fine-tuning did not improve measurably on the behaviour-cloned prior (validation 2.55 → 2.57). Most of
the X9 gap came from stranded UAVs (0.98 vs 0.20); the cluster version adds the heuristic's predictive battery handover to the learners'
shield, which removes stranding in spot checks. The 12,000-episode run will update this row.
"""


def H(s, lvl=2):
    out.append("\n" + "#" * lvl + " " + s + "\n")


def P(s):
    out.append(s + "\n")


def load(name):
    p = os.path.join(RES, name + ".csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def table(df, by, metrics, labels=None, nd=3):
    S = summarize(df, by, metrics)
    labels = labels or metrics
    hdr = "| " + " | ".join(by + labels) + " |"
    sep = "|" + "---|" * (len(by) + len(metrics))
    rows = [hdr, sep]
    for _, r in S.iterrows():
        cells = [str(r[b]) for b in by] + [fmt(r[m], r[m + "_ci"], nd) for m in metrics]
        rows.append("| " + " | ".join(cells) + " |")
    P("\n".join(rows))
    return S


def tests(df, col, a, bs, metrics, keys, better):
    rows = ["| comparison | metric | n | mean (a) | mean (b) | Δ (a−b) | a better in | p (Wilcoxon) | result |",
            "|---|---|---|---|---|---|---|---|---|"]
    res = []
    for b in bs:
        for m in metrics:
            t = paired_test(df, col, a, b, m, keys=keys)
            hi = better.get(m, +1)
            win = t["win"] if hi > 0 else t["loss"]
            good = (t["diff"] * hi > 0) and t["p"] < 0.05
            bad = (t["diff"] * hi < 0) and t["p"] < 0.05
            verdict = "**supported**" if good else ("worse" if bad else "n.s.")
            rows.append(f"| {a} vs {b} | {m} | {t['n']} | {t['mean_a']:.3f} | {t['mean_b']:.3f} | {t['diff']:+.3f} | "
                        f"{100 * win:.0f}% | {t['p']:.2g} | {verdict} |")
            res.append(dict(t, verdict=verdict))
    P("\n".join(rows))
    return res


BETTER = dict(TIR=1, TIR_critical=1, coverage=1, min_patient_TIR=1, jain=1, starvation=-1, AoI_s=-1, latency_s=-1,
              role_restore_s=-1, recovered_conn=1, collisions=-1, violations=-1, sev_w_survival=1, completion=1,
              T1_golden=1, sev_w_response_s=-1, medic_incap=-1, medic_exposure=-1, reassignments=-1, stranded=-1,
              safe_return=1, mission_score=1, survival=1, energy_per_timely=-1, notfound=-1, deaths=-1)
VERDICTS = {}


def bar(ax, S, x, y, colors=None, label_fmt="{:.2f}"):
    xs = np.arange(len(S))
    c = colors or [SERIES[0]] * len(S)
    ax.bar(xs, S[y], yerr=S[y + "_ci"], color=c, width=0.7, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2),
           edgecolor="white", linewidth=1)
    ax.set_xticks(xs); ax.set_xticklabels(S[x], rotation=30, ha="right")
    for i, v in enumerate(S[y]):
        ax.text(i, v + (S[y + "_ci"].iloc[i] if np.isfinite(S[y + "_ci"].iloc[i]) else 0), label_fmt.format(v),
                ha="center", va="bottom", fontsize=7, color=INK2)


# ===================================================================================== X1
def x1():
    df = load("X1_failure_recovery")
    if df is None:
        return
    H("X1 — UAV role failure and recovery (Condition 2)")
    P("Failures injected at t = 150 s (and later for successive/multiple). Configurations 1c/3u/1m, 5c/5u/3m, "
      "10c/9u/4m, 10c/15u/4m; 10 matched seeds per cell. `role_restore_s` = time until every casualty the failed "
      "UAV was sensing is sensed again and observer connectivity is back to its pre-failure level.")
    table(df, ["recovery"], ["TIR", "AoI_s", "role_restore_s", "coverage", "sev_w_survival"])
    P("\nBy failure scenario:")
    table(df, ["scenario", "recovery"], ["TIR", "role_restore_s"])
    P("\nBy configuration:")
    table(df, ["cfg", "recovery"], ["TIR", "role_restore_s"])
    P("\nPaired tests (matched seed × scenario × configuration):")
    VERDICTS["X1"] = tests(df, "recovery", "coordinated", ["none", "adaptive"], ["TIR", "role_restore_s", "AoI_s"],
                           ("seed", "scenario", "cfg"), BETTER)
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    S = summarize(df, ["recovery"], ["TIR", "role_restore_s"])
    bar(axs[0], S, "recovery", "TIR", SERIES[:3]); axs[0].set_title("Timely information ratio", loc="left")
    bar(axs[1], S, "recovery", "role_restore_s", SERIES[:3], "{:.0f}"); axs[1].set_title("Role restoration time (s)", loc="left")
    fig.savefig(os.path.join(FIG, "x1_recovery.png")); plt.close(fig)


def x2():
    df = load("X2_obstacles")
    if df is None:
        return
    H("X2 — Obstacle-aware coordination (Condition 3)")
    table(df, ["obstacles", "positioning"], ["TIR", "coverage", "collisions", "violations", "uav_travel_km"])
    sub = df[df.obstacles != "clear"]
    P("\nPaired tests on obstacle maps (static, dynamic, mixed):")
    VERDICTS["X2"] = tests(sub, "positioning", "adaptive", ["static", "reactive"], ["TIR", "collisions"],
                           ("seed", "obstacles", "cfg"), BETTER)


def x3():
    df = load("X3_scalability")
    if df is None:
        return
    H("X3 — Five-/ten-casualty scalability (Condition 4)")
    for nc in (5, 10):
        P(f"\n{nc} casualties (TIR / coverage / minimum-patient TIR):")
        S = summarize(df[df.n_c == nc], ["n_u", "uav_policy"], ["TIR", "coverage", "min_patient_TIR"])
        piv = S.pivot(index="n_u", columns="uav_policy", values="TIR")
        cols = ["fixed", "coverage", "connected", "balanced"]
        rows = ["| UAVs | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for nu, r in piv.iterrows():
            rows.append(f"| {nu} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
        P("\n".join(rows))
        # UAVs needed for TIR >= 0.8
        need = {}
        for c in cols:
            ok = piv.index[piv[c] >= 0.8]
            need[c] = int(ok.min()) if len(ok) else None
        P("\nSmallest fleet reaching TIR ≥ 0.80: " + ", ".join(f"{c}: {need[c] if need[c] else '>max'}" for c in cols))
    P("\nPaired tests (all fleet sizes):")
    VERDICTS["X3"] = tests(df, "uav_policy", "balanced", ["fixed", "coverage", "connected"],
                           ["TIR", "min_patient_TIR"], ("seed", "n_c", "n_u"), BETTER)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for ax, nc in zip(axs, (5, 10)):
        S = summarize(df[df.n_c == nc], ["uav_policy", "n_u"], ["TIR"])
        for i, pol in enumerate(["fixed", "coverage", "connected", "balanced"]):
            s = S[S.uav_policy == pol].sort_values("n_u")
            ax.plot(s.n_u, s.TIR, color=SERIES[i], lw=2, marker="o", ms=4, label=pol)
            ax.fill_between(s.n_u, s.TIR - s.TIR_ci, s.TIR + s.TIR_ci, color=SERIES[i], alpha=0.12, lw=0)
        ax.set_title(f"{nc} casualties", loc="left"); ax.set_xlabel("UAVs"); ax.set_ylabel("TIR"); ax.set_ylim(0, 1)
    axs[1].legend(loc="lower right")
    fig.savefig(os.path.join(FIG, "x3_scalability.png")); plt.close(fig)


def x4():
    df = load("X4_comm_fairness")
    if df is None:
        return
    H("X4 — Communication, priority and fairness (E6a/E6b)")
    table(df, ["variant"], ["TIR", "TIR_critical", "min_patient_TIR", "jain", "starvation", "AoI_s"])
    P("\nDegraded links (link_degradation = 0.2):")
    table(df[df.degradation > 0], ["variant"], ["TIR", "TIR_critical", "min_patient_TIR", "starvation"])
    P("\nPaired tests:")
    VERDICTS["X4"] = tests(df, "variant", "queue+guarded", ["shortest+guarded", "queue+fifo", "queue+deadline", "queue+deficit"],
                           ["TIR", "TIR_critical", "min_patient_TIR"], ("seed", "n_u", "degradation"), BETTER)


def x5():
    df = load("X5_medic_allocation")
    if df is None:
        return
    H("X5 — Severity-aware medic allocation with medic-safe routing (Condition 6)")
    P("Severity is computed only from vitals (RTS + shock index + SpO2) carried in delivered UAV packets. "
      "Mixed static/dynamic hazards, changing severity, 8 matched seeds × 10 configurations.")
    table(df, ["route", "alloc"], ["sev_w_survival", "completion", "T1_golden", "medic_incap", "medic_exposure", "treatment_aborts"])
    P("\nPaired tests — allocation (all with safe routing):")
    safe = df[df.route == "safe"]
    VERDICTS["X5_alloc"] = tests(safe, "alloc", "urgency", ["nearest", "severity", "load", "gain"],
                                 ["sev_w_survival", "completion"], ("seed", "cfg"), BETTER)
    P("\nPaired tests — medic safety (urgency-distance allocation, safe vs shortest route):")
    d2 = df[df.alloc == "urgency"]
    VERDICTS["X5_safety"] = tests(d2, "route", "safe", ["shortest"], ["medic_incap", "medic_exposure", "sev_w_survival"],
                                  ("seed", "cfg"), BETTER)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    S = summarize(df, ["alloc", "route"], ["sev_w_survival", "medic_incap"])
    for i, alloc in enumerate(["nearest", "severity", "urgency", "load", "gain"]):
        for route, mk in (("shortest", "o"), ("safe", "s")):
            r = S[(S.alloc == alloc) & (S.route == route)].iloc[0]
            ax.scatter(r.medic_incap, r.sev_w_survival, color=SERIES[i], marker=mk, s=40, edgecolor="white", lw=1.5,
                       label=f"{alloc}" if route == "safe" else None, zorder=3)
    ax.set_xlabel("medics incapacitated per episode (lower = safer)"); ax.set_ylabel("severity-weighted survival")
    ax.set_title("Safety vs patient outcome (○ shortest route, □ safe route)", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(FIG, "x5_safety_tradeoff.png")); plt.close(fig)


def x5b():
    df = load("X5b_triage_sweep")
    if df is None:
        return
    H("X5b — Triage sweep: staging distance × severity placement × allocation × medic scarcity")
    P("Pre-specified design: medics staged 50 m or 450 m from the incident centre; five placements of the SAME "
      "patients (identical vitals/physiology) — random, critical patients nearest the medics, critical farthest, "
      "one cluster per triage class, widely scattered; five allocation rules; 5c/1m, 10c/1m, 10c/2m, 10c/4m; "
      "10 matched seeds per cell (2,000 episodes). Safe routing, mixed hazards, changing severity, severity from "
      "delivered vitals only. Primary metrics: severity-weighted survival and T1 (critical) survival. "
      "`start_dist_m` = mean medic-to-casualty distance at t = 0.")
    rules = ["nearest", "severity", "urgency", "load", "gain"]
    for metric, lab in [("sev_w_survival", "Severity-weighted survival"), ("T1_survival", "T1 (critical) survival")]:
        P(f"\n{lab} by staging distance and placement:")
        S = summarize(df, ["ccp", "layout", "alloc"], [metric])
        piv = S.pivot_table(index=["ccp", "layout"], columns="alloc", values=metric)[rules]
        rows = ["| staging | placement | " + " | ".join(rules) + " | best rule − nearest |", "|---|---|" + "---|" * (len(rules) + 1)]
        for (ccp, lay), rr in piv.iterrows():
            best = rr[rules[1:]].max()
            rows.append(f"| {ccp} m | {lay} | " + " | ".join(f"{rr[c]:.3f}" for c in rules) + f" | {best - rr['nearest']:+.3f} |")
        P("\n".join(rows))
    d = df.groupby("ccp")["start_dist_m"].mean()
    P("\nMean medic-to-casualty start distance: " + ", ".join(f"{k} m staging → {v:.0f} m" for k, v in d.items()))
    keys = ("seed", "layout", "cfg")
    for ccp in sorted(df.ccp.unique()):
        P(f"\nPaired tests, {ccp} m staging (pooled over placements and scarcity levels):")
        VERDICTS[f"X5b_{ccp}"] = tests(df[df.ccp == ccp], "alloc", "urgency", ["nearest"],
                                       ["sev_w_survival", "T1_survival"], keys, BETTER)
        tests(df[df.ccp == ccp], "alloc", "gain", ["nearest", "urgency"], ["sev_w_survival", "T1_survival"], keys, BETTER)
        tests(df[df.ccp == ccp], "alloc", "severity", ["nearest"], ["sev_w_survival", "T1_survival"], keys, BETTER)
    P("\nPaired tests by scarcity (50 m staging, urgency-distance vs nearest, severity-weighted survival):")
    for cfg in df.cfg.unique():
        sub = df[(df.ccp == 50) & (df.cfg == cfg)]
        t = paired_test(sub, "alloc", "urgency", "nearest", "sev_w_survival", keys=("seed", "layout"))
        t2 = paired_test(sub, "alloc", "gain", "nearest", "sev_w_survival", keys=("seed", "layout"))
        P(f"- {cfg}: urgency {t['mean_a']:.3f} vs nearest {t['mean_b']:.3f} (Δ {t['diff']:+.3f}, p={t['p']:.2g}); "
          f"survival-gain {t2['mean_a']:.3f} (Δ {t2['diff']:+.3f}, p={t2['p']:.2g})")
    # figure: improvement over nearest per placement (50 m staging)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    lays = ["random", "critical_near", "critical_far", "severity_clusters", "scattered"]
    for ax, ccp in zip(axs, sorted(df.ccp.unique())):
        sub = df[df.ccp == ccp]
        base = sub[sub.alloc == "nearest"].set_index(["seed", "layout", "cfg"])["sev_w_survival"]
        for i, rule in enumerate(["severity", "urgency", "load", "gain"]):
            x = sub[sub.alloc == rule].set_index(["seed", "layout", "cfg"])["sev_w_survival"]
            diff = (x - base.loc[x.index]).groupby(level="layout")
            m = diff.mean().reindex(lays); se = diff.sem().reindex(lays) * 1.96
            ax.bar(np.arange(len(lays)) + (i - 1.5) * 0.2, m, 0.18, yerr=se, color=SERIES[i + 1], label=rule,
                   error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.5), edgecolor="white", linewidth=0.5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(np.arange(len(lays))); ax.set_xticklabels([l.replace("_", "\n") for l in lays], fontsize=7)
        ax.set_title(f"{ccp} m staging: survival gain over nearest-first", loc="left")
    axs[0].set_ylabel("Δ severity-weighted survival"); axs[1].legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "x5b_triage_sweep.png")); plt.close(fig)


def x5grid():
    df = load("X5grid_82configs")
    if df is None:
        return
    H("All 82 base configurations (Condition 6 table)")
    P("Each configuration × 2 seeds, mixed hazards, mixed casualty mobility, changing severity.")
    df["group"] = df.n_c.astype(str) + " casualties"
    table(df, ["group", "stack"], ["sev_w_survival", "completion", "TIR", "medic_incap"])
    P("\nPaired tests over all 82 configurations:")
    VERDICTS["grid"] = tests(df, "stack", "MANTIS-H", ["independent", "independent+safe"],
                             ["sev_w_survival", "TIR", "medic_incap", "mission_score"], ("seed", "n_c", "n_u", "n_m"), BETTER)


def x6():
    df = load("X6_mobility_replanning")
    if df is None:
        return
    H("X6 — Casualty mobility and dynamic medic replanning (Condition 5, E7c)")
    table(df, ["replan"], ["sev_w_survival", "completion", "reassignments", "notfound", "medic_travel_km"])
    P("\nBy mobility pattern (severity-weighted survival):")
    S = summarize(df, ["mobility", "replan"], ["sev_w_survival"])
    piv = S.pivot(index="mobility", columns="replan", values="sev_w_survival")
    cols = ["static", "periodic", "event", "selective", "deadline"]
    rows = ["| mobility | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for mob, r in piv.iterrows():
        rows.append(f"| {mob} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    P("\n".join(rows))
    P("\nPaired tests:")
    VERDICTS["X6"] = tests(df, "replan", "periodic", ["static", "event", "selective", "deadline"],
                           ["sev_w_survival", "reassignments"], ("seed", "mobility", "cfg"), BETTER)


def x7():
    df = load("X7_energy")
    if df is None:
        return
    H("X7 — Energy-aware UAV operation (E8a)")
    P("Heterogeneous batteries (0.6–1.25× nominal, nominal endurance 12.5 min < 20 min mission), 3 spare UAVs at base.")
    table(df, ["env", "energy_policy"], ["TIR", "stranded", "safe_return", "endurance_min", "energy_per_timely", "coverage"])
    P("\nPaired tests:")
    VERDICTS["X7"] = tests(df, "energy_policy", "aware", ["fixed", "reactive"], ["TIR", "stranded", "coverage"],
                           ("seed", "env", "cfg"), BETTER)


def x8():
    df = load("X8_marl")
    if df is None:
        return
    H("X8 — Joint multi-agent learning")
    P("60 held-out evaluation scenarios drawn from the training distribution (random configuration among the 82 with "
      "≤15 UAVs, random obstacles, hazards, mobility, failures, disruption, energy), evaluation seeds never used in "
      "training. Greedy policies. All learners: same features, actions, reward, executors and 500 training episodes.")
    selp = os.path.join(MODELS, "selection.json")
    if os.path.exists(selp):
        sel = json.load(open(selp))
        P("Checkpoint selection (every learner, same protocol): checkpoints saved every ~100 training episodes, "
          "each scored on the validation scenarios (seeds 2·10⁷+, disjoint from training and test); the best is evaluated "
          "below. Episode 0 of BC-initialised runs = the BC policy.\n")
        rows = ["| method | selected checkpoint | validation mission score by checkpoint |", "|---|---|---|"]
        for k, v in sel.items():
            rows.append(f"| {k} | {v['selected'].split('@')[-1]} | " +
                        ", ".join(f"{kk.split('@')[-1]}: {vv:.2f}" for kk, vv in v["scores"].items()) + " |")
        P("\n".join(rows) + "\n")
    order = ["Independent-H", "MANTIS-H", "bc", "ippo", "mappo", "vdn", "qmix", "mantis_mappo"]
    df["method"] = pd.Categorical(df["method"], [m for m in order if m in set(df.method)], ordered=True)
    S = table(df.sort_values("method"), ["method"], ["mission_score", "sev_w_survival", "completion", "TIR", "medic_incap", "AoI_s"])
    P("\nPaired tests (matched scenarios):")
    others = [m for m in order if m != "mantis_mappo" and m in set(df.method)]
    VERDICTS["X8"] = tests(df, "method", "mantis_mappo", others, ["mission_score", "sev_w_survival", "TIR"], ("scen",), BETTER)
    fig, ax = plt.subplots(figsize=(6, 3))
    S = summarize(df.sort_values("method"), ["method"], ["mission_score"])
    cols = [SERIES[1] if m == "mantis_mappo" else (SERIES[0] if "H" in m else "#8a8985") for m in S.method]
    bar(ax, S, "method", "mission_score", cols)
    ax.set_title("Mission score on held-out scenarios (95% CI)", loc="left")
    fig.savefig(os.path.join(FIG, "x8_marl.png")); plt.close(fig)
    # training curves
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, name in enumerate(["mantis_mappo", "mappo", "ippo", "vdn", "qmix"]):
        p = os.path.join(MODELS, f"{name}_last_curve.json")
        if not os.path.exists(p):
            continue
        h = json.load(open(p))
        if not h:
            continue
        y = pd.Series([e["score"] for e in h]).rolling(40, min_periods=10).mean()
        ax.plot(np.arange(len(y)), y, color=SERIES[i], lw=2, label=name)
    ax.set_xlabel("training episode"); ax.set_ylabel("mission score (rolling 40)")
    ax.set_title("Training curves (stochastic policies, random scenarios)", loc="left"); ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG, "x8_training_curves.png")); plt.close(fig)


def x9():
    df = load("X9_robustness")
    if df is None:
        return
    H("X9 — Full six-condition robustness")
    P("Compounded: gateway→relay→observer failures (200/400/600 s), mixed static+dynamic obstacles, mixed casualty "
      "mobility, 20% link degradation + intermittent disruption, changing severity, mixed hazards, low/heterogeneous "
      "batteries (endurance 9 min) with 2 spares. 20 seeds × 3 configurations.")
    order = ["Independent-H", "MANTIS-H", "ippo", "mappo", "qmix", "mantis_mappo"]
    df["method"] = pd.Categorical(df["method"], [m for m in order if m in set(df.method)], ordered=True)
    table(df.sort_values("method"), ["method"], ["mission_score", "sev_w_survival", "TIR", "medic_incap", "stranded", "collisions"])
    P("\nPaired tests:")
    VERDICTS["X9"] = tests(df, "method", "mantis_mappo", [m for m in order if m != "mantis_mappo" and m in set(df.method)],
                           ["mission_score", "sev_w_survival"], ("scen",), BETTER)
    VERDICTS["X9_H"] = tests(df, "method", "MANTIS-H", ["Independent-H"], ["mission_score", "sev_w_survival", "TIR", "medic_incap"],
                             ("scen",), BETTER)


def x10():
    H("X10 — Ablation and generalisation")
    df = load("X10a_heuristic_ablation")
    if df is not None:
        P("(a) MANTIS-H component ablation on compounded scenarios (10c/15u/4m, 20 seeds):")
        table(df, ["variant"], ["mission_score", "sev_w_survival", "TIR", "medic_incap", "stranded"])
        VERDICTS["X10a"] = tests(df, "variant", "MANTIS-H", [v for v in df.variant.unique() if v != "MANTIS-H"],
                                 ["mission_score"], ("scen",), BETTER)
    df = load("X10b_marl_ablation")
    if df is not None:
        P("\n(b) Learning ablation (X8 suite). `abl_mediconly` = learned medics + heuristic UAVs; "
          "`abl_uavonly` = learned UAVs + heuristic medics:")
        table(df, ["method"], ["mission_score", "sev_w_survival", "TIR", "medic_incap"])
        VERDICTS["X10b"] = tests(df, "method", "mantis_mappo", [v for v in df.method.unique() if v != "mantis_mappo"],
                                 ["mission_score"], ("scen",), BETTER)
    df = load("X10c_generalization")
    if df is not None:
        P("\n(c) Generalisation to unseen fleet sizes (18–30 UAVs; training used ≤15) and unseen maps (1.3 km, 5 hazards):")
        table(df, ["gen", "method"], ["mission_score", "sev_w_survival", "TIR", "medic_incap"])
        for g in df.gen.unique():
            VERDICTS["X10c_" + g] = tests(df[df.gen == g], "method", "mantis_mappo", ["MANTIS-H", "mappo"],
                                          ["mission_score"], ("scen",), BETTER)


if __name__ == "__main__":
    out.append("# MANTIS — experimental results\n")
    out.append(SCORECARD)
    out.append("All numbers are means ± 95% t-confidence half-widths over matched scenarios (same seed ⇒ identical map, "
               "casualties, physiology, mobility, failures, hazards for every policy). Evaluation seeds ≥ 10,000,000; "
               "training seeds < 1,000,000. Paired comparisons use the Wilcoxon signed-rank test; "
               "**supported** = better at p < 0.05, *worse* = significantly worse, n.s. = not significant.\n")
    for f in [x1, x2, x3, x4, x5, x5b, x5grid, x6, x7, x8, x9, x10]:
        f()
    open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(out))
    json.dump({k: [{kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) for kk, vv in t.items()} for t in v]
               for k, v in VERDICTS.items()}, open(os.path.join(RES, "verdicts.json"), "w"), indent=1, default=str)
    print("wrote RESULTS.md")
