"""
recompute_fnp.py
================
Targeted recomputation of `function_name_preserved` in features.csv.

Does NOT re-run sentence embeddings or spaCy NLP (expensive).
Reads handoff text from rollouts.csv, entry_point from ALL_TASKS, and
applies the fixed word-boundary extractor to overwrite the column in features.csv.

Also recomputes for final_dataset_full_features.csv (483-row dataset).

Usage:  python recompute_fnp.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tasks.all_tasks import ALL_TASKS          # noqa: E402

# ---------------------------------------------------------------------------
# Entry-point lookup from task metadata
# ---------------------------------------------------------------------------
EP_LOOKUP = {t["id"]: t.get("entry_point", "") for t in ALL_TASKS}
print(f"ALL_TASKS: {len(ALL_TASKS)} tasks, {sum(1 for v in EP_LOOKUP.values() if v)} with entry_point")


def fnp_fixed(handoff: str, entry_point: str) -> float:
    """Word-boundary check: returns 1.0 if entry_point appears as a whole word."""
    if not entry_point:
        return 0.5
    pattern = r"\b" + re.escape(entry_point) + r"\b"
    return 1.0 if re.search(pattern, handoff) else 0.0


# ---------------------------------------------------------------------------
# Recompute for features.csv (2696 rows) using rollouts.csv for text
# ---------------------------------------------------------------------------
print("\n--- features.csv (2696 rows) ---")
fc = pd.read_csv(ROOT / "features.csv")
ro = pd.read_csv(ROOT / "rollouts.csv")

# Align: rollouts.csv and features.csv should be row-identical
assert len(fc) == len(ro), f"Row count mismatch: {len(fc)} vs {len(ro)}"
assert list(fc["task_id"]) == list(ro["task_id"]), "task_id mismatch"
assert list(fc["corruption_type"]) == list(ro["corruption_type"]), "corruption_type mismatch"

old_col = fc["function_name_preserved"].copy()
fc["function_name_preserved"] = [
    fnp_fixed(str(ro.loc[i, "handoff"]), EP_LOOKUP.get(ro.loc[i, "task_id"], ""))
    for i in range(len(ro))
]

print(f"Old value_counts: {old_col.value_counts().to_dict()}")
print(f"New value_counts: {fc['function_name_preserved'].value_counts().to_dict()}")
print()
print("New mean by corruption_type:")
print(fc.groupby("corruption_type")["function_name_preserved"].mean().round(4).to_string())

fc.to_csv(ROOT / "features.csv", index=False)
print("\nSaved -> features.csv")

# Quick AUROC check
from sklearn.metrics import roc_auc_score
clean_mask = fc["corruption_type"] == "none"
for ctype in ["signature_rename", "tool_result_drop", "over_summarization", "fake_completion"]:
    mask = fc["corruption_type"].isin(["none", ctype])
    sub = fc[mask]
    if sub["label"].nunique() == 2:
        auroc = roc_auc_score(sub["label"], sub["function_name_preserved"])
        print(f"  Single-feat AUROC ({ctype}): {auroc:.4f}")

# ---------------------------------------------------------------------------
# Recompute for final_dataset_full_features.csv (483 rows)
# This one was built from final_dataset.csv which doesn't have entry_point
# but final_dataset_full_features.csv does have handoff text
# ---------------------------------------------------------------------------
fd_path = ROOT / "final_dataset_full_features.csv"
if fd_path.exists():
    print("\n--- final_dataset_full_features.csv (483 rows) ---")
    fd = pd.read_csv(fd_path)
    if "handoff" not in fd.columns:
        print("  SKIP: no handoff column in final_dataset_full_features.csv")
    else:
        old_fd = fd["function_name_preserved"].copy()
        fd["function_name_preserved"] = [
            fnp_fixed(str(fd.loc[i, "handoff"]), EP_LOOKUP.get(fd.loc[i, "task_id"], ""))
            for i in range(len(fd))
        ]
        print(f"Old value_counts: {old_fd.value_counts().to_dict()}")
        print(f"New value_counts: {fd['function_name_preserved'].value_counts().to_dict()}")
        print()
        print("New mean by corruption_type:")
        print(fd.groupby("corruption_type")["function_name_preserved"].mean().round(4).to_string())
        fd.to_csv(fd_path, index=False)
        print(f"\nSaved -> {fd_path.name}")

print("\nDone.")
