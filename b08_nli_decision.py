"""B8: NLI feature decision + freeze features_v2.

DECISION RULE (written before looking at results):
  keep nli_entailment_mean ONLY IF (a) within-task paired direction is correct
  (clean entailment > corrupted entailment, sign test p<0.05) AND (b) the paired
  task-bootstrap 95% CI of dAUROC (12-feature minus 11-feature RF) excludes 0 on the
  positive side.  Otherwise drop it (frozen set = 11 gate features).
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import roc_auc_score
from common.data import load_raw, RES, FEAT_11, FEAT_12, make_rf, oof_multi, V2_CSV, V2_JSON
from common.stats_utils import auc, paired_bootstrap_diff

d = load_raw()
nli = "nli_entailment_mean"
d.groupby("corruption_type")[nli].agg(["size", "mean", "std"]).round(4).to_csv(RES / "b08_nli_by_type.csv")
print(d.groupby("label")[nli].mean().round(4))

# 2. within-task paired sign test
clean = d[d.corruption_type == "none"].set_index("task_id")[nli]
bad = d[d.corruption_type != "none"].copy()
bad["clean_nli"] = bad.task_id.map(clean)
bad["diff"] = bad.clean_nli - bad[nli]
rows = []
for name, g in [("ALL", bad)] + list(bad.groupby("corruption_type")):
    w, l = int((g["diff"] > 0).sum()), int((g["diff"] < 0).sum()); t = int((g["diff"] == 0).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else np.nan
    rows.append(dict(type=name, pairs=len(g), clean_higher=w, clean_lower=l, ties=t,
                     frac_clean_higher=w / max(w + l, 1), sign_test_p=p, mean_diff=g["diff"].mean()))
sign = pd.DataFrame(rows).round(4); sign.to_csv(RES / "b08_nli_sign_test.csv", index=False); print(sign)

# 3. aggregation confound: mean over <=16 sentence pairs; correlation with volume
rho_len = spearmanr(d[nli], d.length_ratio)[0]; rho_sc = spearmanr(d[nli], d.sentence_count_ratio)[0]
print(f"spearman(nli, length_ratio)={rho_len:.3f}  (nli, sentence_count_ratio)={rho_sc:.3f}")

# 4. drop-column dAUROC, paired CI (5 fold seeds averaged)
s12 = oof_multi(d, FEAT_12, make_rf); s11 = oof_multi(d, FEAT_11, make_rf)
lo, hi, nv, pt = paired_bootstrap_diff(d, s12, s11, auc)
res = pd.DataFrame([dict(auroc_12=auc(d.label, s12), auroc_11=auc(d.label, s11), delta=pt,
                         ci_lo=lo, ci_hi=hi, n_valid=nv)]).round(4)
res.to_csv(RES / "b08_nli_drop_column.csv", index=False); print(res)

p_all = sign.loc[0, "sign_test_p"]; frac = sign.loc[0, "frac_clean_higher"]
keep = bool(frac > 0.5 and p_all < 0.05 and lo > 0)
feats = FEAT_12 if keep else FEAT_11
print("DECISION:", "KEEP" if keep else "DROP", "->", len(feats), "features")
d[["task_id", "source", "corruption_type", "label", *feats]].to_csv(V2_CSV, index=False)
V2_JSON.write_text(json.dumps({"features": feats, "nli_kept": keep,
                               "rule": "keep iff paired direction correct & dAUROC CI > 0"}, indent=1))

fig, ax = plt.subplots(figsize=(7, 4))
d.boxplot(column=nli, by="corruption_type", ax=ax, rot=60); ax.set_title("B8: NLI entailment by type"); plt.suptitle("")
plt.tight_layout(); plt.savefig(RES / "figures" / "b08_nli_by_type.png", dpi=150)
