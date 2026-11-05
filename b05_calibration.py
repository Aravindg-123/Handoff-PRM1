"""B5: ECE/Brier, nested isotonic/Platt calibration, prior shift, per-type score plot."""
import pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
from common.calib import nested_scores, prior_shift
from common.data import load_v2, make_rf, RES
from common.stats_utils import ece_equal_mass

d, feats = load_v2(); y = d.label.values
ns = nested_scores(d, feats, make_rf, seed=0)
pickle.dump(ns, open(RES / "b05_nested.pkl", "wb"))
rows = [dict(scores=k, ece=ece_equal_mass(y, ns[k]), brier=brier_score_loss(y, ns[k]),
             auroc=roc_auc_score(y, ns[k])) for k in ["raw", "iso", "platt"]]
cal = pd.DataFrame(rows).round(4); cal.to_csv(RES / "b05_calibration.csv", index=False); print(cal)
pd.DataFrame({"row_id": d.index, "task_id": d.task_id, "corruption_type": d.corruption_type, "label": y,
              "raw": ns["raw"], "iso": ns["iso"], "platt": ns["platt"]}).to_csv(RES / "b05_calibrated_oof.csv", index=False)

# prior shift: dev P(good)=pi_dev; deployment bad-rate 5% => P(good)=0.95
pi_dev = y.mean()
sh = pd.DataFrame([dict(deploy_bad_rate=b, mean_p_good_dev=ns["platt"].mean(),
                        mean_p_good_shifted=prior_shift(ns["platt"], pi_dev, 1 - b).mean())
                   for b in [0.5, 0.2, 0.05, 0.01]]).round(4)
sh.to_csv(RES / "b05_prior_shift.csv", index=False); print(sh, "pi_dev(good)=", round(pi_dev, 4))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for k, c in [("raw", "C0"), ("iso", "C1"), ("platt", "C2")]:
    o = np.argsort(ns[k]); b = np.array_split(o, 10)
    ax[0].plot([ns[k][i].mean() for i in b], [y[i].mean() for i in b], "o-", c=c, label=k)
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_xlabel("mean predicted P(good)"); ax[0].set_ylabel("observed fraction good"); ax[0].legend()
ax[0].set_title("Reliability (10 equal-mass bins, OOF)")
order = d.groupby("corruption_type").apply(lambda g: ns["raw"][g.index].mean()).sort_values().index
ax[1].violinplot([ns["raw"][d.index[d.corruption_type == t]] for t in order], vert=False, showmedians=True)
ax[1].set_yticks(range(1, len(order) + 1)); ax[1].set_yticklabels(order); ax[1].set_xlabel("raw OOF score"); ax[1].set_title("Score by corruption type")
plt.tight_layout(); plt.savefig(RES / "figures" / "b05_calibration.png", dpi=150)
