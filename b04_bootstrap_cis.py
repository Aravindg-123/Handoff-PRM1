"""B4: task-level bootstrap CIs for model comparison, ablation, per-source; paired differences.
Intervals reflect EVALUATION-sample uncertainty (tasks resampled on fixed OOF scores), not training variance."""
import numpy as np, pandas as pd
from common.cv_utils import oof_scores
from common.data import load_v2, make_rf, RES
from common.oof_store import load_oof
from common.stats_utils import auc, task_bootstrap, paired_bootstrap_diff

d, feats = load_v2()
oof = load_oof(); get = lambda m: oof[(oof.model == m) & (oof.cv_seed == 0)].set_index("row_id").score.reindex(d.index).values
assert (oof[oof.model == "RandomForest_11feat"].task_id.values == d.task_id.values).all()

# Table VIII: model comparison with CIs
models = ["LogisticRegression_3feat", "RandomForest_3feat", "XGBoost_3feat",
          "LogisticRegression_11feat", "RandomForest_11feat", "XGBoost_11feat"]
rows = []
for m in models:
    lo, hi, nv, pt = task_bootstrap(d, get(m), auc); rows.append(dict(model=m, auroc=pt, lo=lo, hi=hi, n_valid=nv))
pd.DataFrame(rows).round(4).to_csv(RES / "b04_model_comparison.csv", index=False); print(pd.DataFrame(rows).round(4))

# paired differences
pairs = [("RandomForest_11feat", "XGBoost_11feat"), ("RandomForest_11feat", "LogisticRegression_11feat"),
         ("XGBoost_11feat", "LogisticRegression_11feat"), ("RandomForest_11feat", "RandomForest_3feat"),
         ("XGBoost_11feat", "XGBoost_3feat")]
rows = []
for a, b in pairs:
    lo, hi, nv, pt = paired_bootstrap_diff(d, get(a), get(b), auc); rows.append(dict(a=a, b=b, delta=pt, lo=lo, hi=hi))
pd.DataFrame(rows).round(4).to_csv(RES / "b04_paired_model_diffs.csv", index=False); print(pd.DataFrame(rows).round(4))

# Ablation (RF, seed 0): drop-one and single-feature
full = get("RandomForest_11feat"); rows = []
for f in feats:
    s_drop = oof_scores(d, [c for c in feats if c != f], make_rf, seed=0)
    s_one = oof_scores(d, [f], make_rf, seed=0)
    lo, hi, _, dd = paired_bootstrap_diff(d, s_drop, full, auc)       # drop minus full
    l1, h1, _, a1 = task_bootstrap(d, s_one, auc, n_boot=5000)
    rows.append(dict(feature=f, auroc_drop_one=auc(d.label, s_drop), delta_vs_full=dd, delta_lo=lo, delta_hi=hi,
                     auroc_alone=a1, alone_lo=l1, alone_hi=h1))
ab = pd.DataFrame(rows).round(4).sort_values("delta_vs_full"); ab.to_csv(RES / "b04_ablation.csv", index=False); print(ab)

# per source
rows = []
for s in sorted(d.source.unique()):
    m = (d.source == s).values; lo, hi, nv, pt = task_bootstrap(d, full, auc, mask=m)
    rows.append(dict(source=s, n=int(m.sum()), n_neg=int((d.label[m] == 0).sum()), auroc=pt, lo=lo, hi=hi, n_valid=nv))
pd.DataFrame(rows).round(4).to_csv(RES / "b04_per_source.csv", index=False); print(pd.DataFrame(rows).round(4))
