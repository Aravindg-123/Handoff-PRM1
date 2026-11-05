"""B2: diagnose function_name_preserved (fnp). Writes results/b02_*.csv, figure."""
import re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from common.data import load_raw, RES
from common.stats_utils import task_bootstrap
from feature_extraction import compute_function_name_preserved as fnp

d = load_raw()
ch = pd.read_csv("corrupted_handoffs.csv")
ep = ch.drop_duplicates("task_id").set_index("task_id")["entry_point"]
orig = ch.set_index(["task_id", "corruption_type"])["original_handoff"]
d["entry_point"] = d.task_id.map(ep)
assert d.entry_point.notna().all()
d["orig_handoff"] = [orig.get((t, c), h) if c != "none" else h
                     for t, c, h in zip(d.task_id, d.corruption_type, d.handoff)]

# 0. stored feature == recomputation on the stored text? (rules out "computed on pre-corruption text")
d["fnp_recomputed"] = [fnp(h, "", e) for h, e in zip(d.handoff, d.entry_point)]
d["fnp_on_original"] = [fnp(h, "", e) for h, e in zip(d.orig_handoff, d.entry_point)]
stored_ok = (d.fnp_recomputed == d.function_name_preserved).mean()

# 1. the 14 signature_rename rows
defn = lambda h: re.findall(r"def\s+(\w+)\s*\(", h)
sig = d[d.corruption_type == "signature_rename"].copy()
rows = pd.DataFrame({
    "task_id": sig.task_id, "expected_name": sig.entry_point,
    "orig_has_name": [bool(re.search(rf"\b{re.escape(e)}\b", h)) for e, h in zip(sig.entry_point, sig.orig_handoff)],
    "orig_def_names": [defn(h) for h in sig.orig_handoff],
    "corrupt_def_names": [defn(h) for h in sig.handoff],
    "n_mentions_orig": [len(re.findall(rf"\b{re.escape(e)}\b", h)) for e, h in zip(sig.entry_point, sig.orig_handoff)],
    "n_mentions_corrupt": [len(re.findall(rf"\b{re.escape(e)}\b", h)) for e, h in zip(sig.entry_point, sig.handoff)],
    "feature_value": sig.function_name_preserved})
rows.to_csv(RES / "b02_signature_rename_rows.csv", index=False)

# 2. mean by type
bytype = d.groupby("corruption_type").agg(n=("label", "size"), fnp_mean=("function_name_preserved", "mean"),
                                          fnp_on_original_mean=("fnp_on_original", "mean")).round(3)
bytype.to_csv(RES / "b02_fnp_by_type.csv")

# 3. single-feature AUROC {clean, signature_rename}
m = d.corruption_type.isin(["none", "signature_rename"]).values
lo, hi, nv, pt = task_bootstrap(d, d.function_name_preserved.values, roc_auc_score, mask=m)
single = [dict(subset="clean vs signature_rename", auroc=pt, ci_lo=lo, ci_hi=hi, n_valid=nv)]
for t in sorted(set(d.corruption_type) - {"none", "signature_rename"}):
    mm = d.corruption_type.isin(["none", t]).values
    lo, hi, nv, pt = task_bootstrap(d, d.function_name_preserved.values, roc_auc_score, mask=mm, n_boot=2000)
    single.append(dict(subset=f"clean vs {t}", auroc=pt, ci_lo=lo, ci_hi=hi, n_valid=nv))
lo, hi, nv, pt = task_bootstrap(d, d.function_name_preserved.values, roc_auc_score)
single.append(dict(subset="ALL (clean vs every corrupted)", auroc=pt, ci_lo=lo, ci_hi=hi, n_valid=nv))
pd.DataFrame(single).round(3).to_csv(RES / "b02_fnp_single_feature_auroc.csv", index=False)

# 4. why do clean rows score 0?  (35% of clean rows)
cl = d[d.corruption_type == "none"].copy()
z = cl[cl.function_name_preserved == 0]
why = pd.Series({
    "clean rows": len(cl), "clean rows with fnp=0": len(z),
    "  ...with NO def line at all": sum(len(defn(h)) == 0 for h in z.handoff),
    "  ...def line uses a different name": sum(len(defn(h)) > 0 for h in z.handoff),
    "  ...name appears only as substring (e.g. name_impl)": sum(bool(re.search(re.escape(e), h)) for e, h in zip(z.entry_point, z.handoff)),
    "stored feature == recomputation": f"{stored_ok:.3f}"})
why.to_csv(RES / "b02_clean_zero_breakdown.csv", header=["value"])
print(rows.to_string()); print(bytype); print(pd.DataFrame(single).round(3)); print(why)
by_src = z.source.value_counts() / cl.source.value_counts()
print("clean fnp=0 rate by source:\n", by_src.round(3))
by_src.round(3).to_csv(RES / "b02_clean_zero_by_source.csv", header=["rate"])

fig, ax = plt.subplots(figsize=(7, 4))
bt = bytype.fnp_mean.sort_values(); bt.plot.barh(ax=ax, color="#4477aa")
ax.set_xlabel("mean function_name_preserved"); ax.set_title("B2: fnp by corruption type")
plt.tight_layout(); plt.savefig(RES / "figures" / "b02_fnp_by_type.png", dpi=150)
