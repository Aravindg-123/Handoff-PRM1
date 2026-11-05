"""B1: leave-one-corruption-type-out, protocols S (seen) and U (unseen), same code path.
Model: RF on frozen features_v2, task-grouped folds, 20 fold seeds."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.cv_utils import oof_scores
from common.data import load_v2, make_rf, RES
from common.stats_utils import auc, task_bootstrap, clopper_pearson

SEEDS = range(20)
d, feats = load_v2()
types = sorted(set(d.corruption_type) - {"none"})
ct = d.corruption_type.values
S = np.zeros((len(SEEDS), len(d))); U = {t: np.zeros((len(SEEDS), len(d))) for t in types}
for i, sd in enumerate(SEEDS):
    S[i] = oof_scores(d, feats, make_rf, seed=sd)
    for t in types:
        U[t][i] = oof_scores(d, feats, make_rf, seed=sd, drop_from_train=(ct == t))
    print("seed", sd, flush=True)

def per_seed_auc(M, t):
    m = np.isin(ct, ["none", t]); return np.array([auc(d.label[m], M[i][m]) for i in range(len(M))])

rows = []
clean = d[d.corruption_type == "none"].set_index("task_id").index
for t in types:
    m = np.isin(ct, ["none", t]); n = int((ct == t).sum())
    aS, aU = per_seed_auc(S, t), per_seed_auc(U[t], t)
    sm, um = S.mean(0), U[t].mean(0)            # seed-averaged scores for the CI / pairs
    loS, hiS, nvS, _ = task_bootstrap(d, sm, auc, mask=m, n_boot=5000)
    loU, hiU, nvU, _ = task_bootstrap(d, um, auc, mask=m, n_boot=5000)
    # paired win rate: within task, clean row scores higher than the type-X row (protocol U)
    s = pd.Series(um, index=d.index)
    cs = s[ct == "none"].set_axis(d.task_id[ct == "none"])
    xs = s[ct == t].set_axis(d.task_id[ct == t])
    diff = (cs.reindex(xs.index) - xs).dropna()
    w, l, ti = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
    cl, ch = clopper_pearson(w, w + l)
    rows.append(dict(type=t, n_neg=n, auroc_S=aS.mean(), S_lo=loS, S_hi=hiS,
                     auroc_U=aU.mean(), U_lo=loU, U_hi=hiU, U_minus_S=aU.mean() - aS.mean(),
                     seeds_sd_U=aU.std(), wins=w, ties=ti, losses=l,
                     win_rate=w / max(w + l, 1), win_lo=cl, win_hi=ch,
                     valid_boot_S=nvS, valid_boot_U=nvU))
out = pd.DataFrame(rows).sort_values("auroc_S", ascending=False).round(4)
out.to_csv(RES / "b01_loto_table.csv", index=False); print(out.to_string())
# OOF scores for reuse (seed-averaged S)
pd.DataFrame({"row_id": d.index, "task_id": d.task_id, "S_mean": S.mean(0)}).to_csv(RES / "b01_S_scores.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5)); y = np.arange(len(out))
ax.errorbar(out.auroc_S, y - .15, xerr=[out.auroc_S - out.S_lo, out.S_hi - out.auroc_S], fmt="o", label="S (seen)")
ax.errorbar(out.auroc_U, y + .15, xerr=[out.auroc_U - out.U_lo, out.U_hi - out.auroc_U], fmt="s", label="U (unseen)")
ax.set_yticks(y); ax.set_yticklabels(out.type); ax.axvline(.5, c="gray", ls=":"); ax.legend(); ax.invert_yaxis()
ax.set_xlabel("AUROC on {clean, type}"); plt.tight_layout(); plt.savefig(RES / "figures" / "b01_loto.png", dpi=150)
