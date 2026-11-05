"""B7: is length_ratio carrying signal beyond corruption type? (discard dataset => type determines label,
so label ~ features + C(type) is perfectly separable; use matched pairs + within-type tests.)"""
import pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common.cv_utils import task_folds
from common.data import load_v2, RES
from common.stats_utils import auc, task_bootstrap, paired_bootstrap_diff

d, feats = load_v2()
ns = pickle.load(open(RES / "b05_nested.pkl", "rb")); d["raw"] = ns["raw"]
sd = d[feats].std().values
clean = d[d.corruption_type == "none"].set_index("task_id")
bad = d[d.corruption_type != "none"].reset_index(drop=True)
X = (clean.loc[bad.task_id, feats].values - bad[feats].values) / sd       # clean - corrupted, scaled
rng = np.random.RandomState(0); flip = rng.rand(len(bad)) < .5
Xs = np.where(flip[:, None], -X, X); ysign = (~flip).astype(int)           # label 1 <=> sign = clean-first
P = pd.DataFrame({"task_id": bad.task_id, "label": ysign, "type": bad.corruption_type})
fold = task_folds(P, 5, 0)


def cv_scores(cols):
    ix = [feats.index(c) for c in cols]; out = np.zeros(len(P))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = LogisticRegression(fit_intercept=False, max_iter=2000).fit(Xs[tr][:, ix], ysign[tr])
        out[te] = m.predict_proba(Xs[te][:, ix])[:, 1]
    return out


sets = {"all 11": feats, "all minus length_ratio": [f for f in feats if f != "length_ratio"],
        "length_ratio only": ["length_ratio"],
        "minus length_ratio & sentence_count_ratio": [f for f in feats if f not in ("length_ratio", "sentence_count_ratio")]}
sc = {k: cv_scores(v) for k, v in sets.items()}
rows = []
for k, s in sc.items():
    lo, hi, _, pt = task_bootstrap(P, s, auc, n_boot=5000)
    rows.append(dict(model=k, pair_auroc=pt, lo=lo, hi=hi))
dl, dh, _, dp = paired_bootstrap_diff(P, sc["all 11"], sc["all minus length_ratio"], auc, n_boot=5000)
mp = pd.DataFrame(rows).round(4); mp.to_csv(RES / "b07_matched_pair.csv", index=False); print(mp)
print(f"all11 - noLR: {dp:.4f} [{dl:.4f}, {dh:.4f}]")
coef = LogisticRegression(fit_intercept=False, max_iter=2000).fit(Xs, ysign).coef_[0]
pd.Series(coef, feats).round(3).sort_values().to_csv(RES / "b07_matched_pair_coefs.csv", header=["coef_per_sd"])
pd.DataFrame([dict(delta_auroc_all_minus_noLR=dp, lo=dl, hi=dh)]).round(4).to_csv(RES / "b07_matched_pair_delta.csv", index=False)

# within-type, length-matched controls (greedy 1-NN on length_ratio, no replacement)
rows = []
cl = d[d.corruption_type == "none"]
for t in ["entity_omission", "truncation", "invert_objective", "negate_edge_case", "wrong_algorithm_name",
          "signature_rename", "inject_false_constraint", "fake_completion"]:
    x = d[d.corruption_type == t]; avail = cl.copy(); picks = []
    for i, r in x.sort_values("length_ratio").iterrows():
        j = (avail.length_ratio - r.length_ratio).abs().idxmin(); picks.append(j); avail = avail.drop(j)
    m = np.zeros(len(d), bool); m[x.index] = True; m[picks] = True
    ma = d[m]; both = d.corruption_type.isin(["none", t])
    pair = np.sign(clean.loc[x.task_id, "length_ratio"].values - x.length_ratio.values)
    w, l = int((pair > 0).sum()), int((pair < 0).sum())
    ci = task_bootstrap(d, d.raw.values, auc, mask=m, n_boot=2000)[:2]
    rows.append(dict(type=t, n=len(x),
                     lr_auroc_unmatched=auc(d.label[both], d.length_ratio[both]),
                     lr_auroc_matched=auc(ma.label.values, ma.length_ratio.values),
                     model_auroc_matched=auc(ma.label.values, ma.raw.values),
                     model_auroc_matched_ci="[%.3f, %.3f]" % tuple(ci),
                     sibling_lr_clean_higher=f"{w}/{w + l}"))
wt = pd.DataFrame(rows).round(3); wt.to_csv(RES / "b07_within_type_matched.csv", index=False); print(wt.to_string())
