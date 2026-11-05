"""B3: threshold protocol.
RULE (fixed before looking): block when score < t; choose the largest t such that the
false-block rate on clean handoffs <= 5%  (=> maximum detection subject to that cap).
Headline t: chosen on pooled Platt-calibrated dev OOF scores (B5), then FROZEN.
Nested estimate: per outer fold, t from INNER OOF raw scores of training tasks only,
applied to the outer test fold, pooled."""
import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from common.data import load_v2, RES
from common.stats_utils import block_metrics, threshold_fbr_cap, task_bootstrap

CAP = 0.05
d, feats = load_v2(); y = d.label.values
ns = pickle.load(open(RES / "b05_nested.pkl", "rb"))
s = ns["platt"]   # Platt: lowest ECE of the calibrators, ties-free (isotonic lowered AUROC)

t_head = threshold_fbr_cap(y, s, CAP)
# raw-scale equivalent (for the gate, which scores with the raw RF)
t_raw = threshold_fbr_cap(y, ns["raw"], CAP)

# nested: threshold from inner OOF only
blk = np.zeros(len(d), bool); t_inner = []
for k, (tr, s_in) in enumerate(ns["inner"]):
    t = threshold_fbr_cap(y[tr], s_in, CAP); t_inner.append(t)
    te = ns["fold"] == k
    blk[te] = ns["raw"][te] < t
nested = block_metrics(y, np.where(blk, 0.0, 1.0), 0.5)

def ci_of(name, scores, t):
    f = lambda yy, ss: block_metrics(yy, ss, t)[name]
    lo, hi, nv, pt = task_bootstrap(d, scores, f, n_boot=5000)
    return pt, lo, hi

rows = []
for tag, sc, t in [("headline (Platt-calibrated OOF)", s, t_head), ("headline on raw OOF", ns["raw"], t_raw)]:
    m = block_metrics(y, sc, t)
    r = dict(protocol=tag, threshold=round(t, 4), **{k: m[k] for k in ["TP", "FN", "FP", "TN"]})
    for nme in ["tpr", "fbr", "precision", "f1"]:
        pt, lo, hi = ci_of(nme, sc, t); r[nme] = pt; r[nme + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    # F1 of the GOOD class (this is the quantity behind the old 0.9464 in Table XI)
    r["f1_good_class"] = 2 * m["TN"] / (2 * m["TN"] + m["FP"] + m["FN"])
    rows.append(r)
r = dict(protocol="nested (inner-fold thresholds)", threshold=f"mean {np.mean(t_inner):.3f} (sd {np.std(t_inner):.3f})",
         **{k: nested[k] for k in ["TP", "FN", "FP", "TN"]})
f = lambda nme: (lambda yy, ss: block_metrics(yy, ss, 0.5)[nme])
for nme in ["tpr", "fbr", "precision", "f1"]:
    lo, hi, nv, pt = task_bootstrap(d, np.where(blk, 0.0, 1.0), f(nme), n_boot=5000)
    r[nme] = pt; r[nme + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
r["f1_good_class"] = 2 * nested["TN"] / (2 * nested["TN"] + nested["FP"] + nested["FN"])
rows.append(r)
out = pd.DataFrame(rows).round(4); out.to_csv(RES / "b03_threshold.csv", index=False); print(out.T)

# the thresholds quoted in the paper, evaluated on raw OOF for context
ctx = pd.DataFrame([dict(threshold=t, **{k: round(v, 3) for k, v in block_metrics(y, ns["raw"], t).items()})
                    for t in [0.330, 0.45, 0.50, 0.695, 0.7972]])
ctx.to_csv(RES / "b03_paper_thresholds_on_raw_oof.csv", index=False); print(ctx)
json.dump(dict(rule=f"largest t with clean false-block rate <= {CAP}", threshold_platt=t_head,
               threshold_raw=t_raw), open(RES / "frozen_threshold.json", "w"), indent=1)

# ROC + PR (detecting BAD handoffs: positive = bad, score = 1 - s) with chosen point
bad = 1 - y; det = 1 - s; m = block_metrics(y, s, t_head)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
fpr, tpr, _ = roc_curve(bad, det); ax[0].plot(fpr, tpr); ax[0].plot(m["fbr"], m["tpr"], "ro", label=f"t={t_head:.3f}")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_xlabel("false-block rate"); ax[0].set_ylabel("detection rate"); ax[0].legend()
p, r_, _ = precision_recall_curve(bad, det); ax[1].plot(r_, p); ax[1].plot(m["tpr"], m["precision"], "ro")
ax[1].set_xlabel("recall (bad)"); ax[1].set_ylabel("precision of blocks"); ax[0].set_title("ROC"); ax[1].set_title("PR (bad = positive)")
plt.tight_layout(); plt.savefig(RES / "figures" / "b03_roc_pr.png", dpi=150)
