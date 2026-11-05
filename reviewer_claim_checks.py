"""Optional: reproduces the data facts used to verify the reviewer's points.
Outputs: label/corruption crosstab, length_ratio stats, function_name_preserved
stats and AUROC, NLI entailment/contradiction statistics.
Usage: python reviewer_claim_checks.py path/to/final_dataset_full_features.csv
"""
import sys, pandas as pd
from sklearn.metrics import roc_auc_score

f = pd.read_csv(sys.argv[1], encoding_errors="replace")

# 1. Label vs corruption alignment
print("=== label vs corruption present ===")
print(pd.crosstab(f.corruption_type != "none", f.label))

# 2. Length-ratio statistics by corruption type
print("\nlength_ratio mean by type:\n",
      f.groupby("corruption_type").length_ratio.mean().round(2).to_string())

# 3. function_name_preserved mean by corruption type
print("\nfunction_name_preserved mean by type:\n",
      f.groupby("corruption_type").function_name_preserved.mean().round(3).to_string())

# AUROC: signature_rename keeps the original fn name in the handoff body,
# so fn_preserved == 1 for BOTH corrupted and clean -> AUROC ~ 0.50 is expected.
c = f[f.corruption_type.isin(["signature_rename", "none"])]
print("\nfn_preserved AUROC (signature_rename vs clean):",
      round(roc_auc_score(c.label, c.function_name_preserved), 3))
print("  (expected ~0.50: feature cannot discriminate this corruption type)")

# 4. NLI statistics
# nli_entailment_mean: higher -> more agreement -> predicts clean (label=1)
ent = f.groupby("label").nli_entailment_mean.mean().round(3).to_dict()
auroc_ent = round(roc_auc_score(f.label, f.nli_entailment_mean.fillna(0)), 3)
print("\nnli_entailment_mean by label:", ent,
      "| AUROC (higher=clean):", auroc_ent)

# nli_contradiction_max: counter-intuitively higher for label=1 (clean) because
# longer clean handoffs generate more sentence pairs, inflating the max.
con = f.groupby("label").nli_contradiction_max.mean().round(3).to_dict()
print("nli_contradiction_max by label:", con)
print("  (higher for clean because longer handoffs inflate max; use entailment)")
