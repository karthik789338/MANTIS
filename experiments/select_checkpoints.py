"""Checkpoint selection on a VALIDATION scenario set (seeds 2e7+, disjoint from training < 1e6 and
test >= 1e7 (+5000..+9999)). The same protocol is applied to every learned method."""
import sys, glob, shutil, json, re
import numpy as np
from common import *
from mantis.marl.algos import sample_train_cfg

P = 2
for a in sys.argv:
    if a.startswith("--procs="):
        P = int(a.split("=")[1])
NVAL = int(os.environ.get("MANTIS_NVAL", "16"))


def val_suite():
    rng = np.random.default_rng(777)
    out = []
    for i in range(NVAL):
        kw = sample_train_cfg(rng).to_dict(); kw["seed"] = 20_000_000 + i
        out.append(kw)
    return out


def main(names):
    ck = os.path.join(MODELS, "ckpt")
    sel = json.load(open(os.path.join(MODELS, "selection.json"))) if os.path.exists(os.path.join(MODELS, "selection.json")) else {}
    scen = val_suite()
    for name in names:
        files = sorted(glob.glob(os.path.join(ck, f"{name}@*.pt")), key=lambda f: int(re.findall(r"@(\d+)", f)[0]))
        if not files:
            continue
        jobs = []
        for f in files:
            tag = "ckpt/" + os.path.basename(f)[:-3]
            for i, kw in enumerate(scen):
                jobs.append((kw, {}, "marl:" + tag, dict(ckpt=tag, scen=i)))
        df = run_jobs(jobs, f"val_{name}", P)
        S = df.groupby("ckpt", sort=False)["mission_score"].mean()
        best = S.idxmax()
        shutil.copy(os.path.join(MODELS, best + ".pt"), os.path.join(MODELS, f"{name}.pt"))
        sel[name] = dict(selected=best, scores={k: float(v) for k, v in S.items()})
        print(name, "->", best, {k: round(v, 3) for k, v in S.items()}, flush=True)
        json.dump(sel, open(os.path.join(MODELS, "selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])
