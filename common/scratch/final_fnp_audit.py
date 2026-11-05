"""
Final per-type table and AUROC on the 483 labeled rows.
10,000 task-bootstrap CIs. Freeze features_v2 after this.
"""
import pandas as pd, numpy as np, re, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.all_tasks import ALL_TASKS
from common.cv_utils import task_bootstrap_ci

ep_map = {t['id']: t.get('entry_point', '') for t in ALL_TASKS}

df = pd.read_csv(ROOT / 'final_dataset_full_features.csv')
df['ep'] = df['task_id'].map(ep_map)

def fnp_v2(handoff, ep):
    if not ep:
        return 0.5
    return 1.0 if re.search(r'\b' + re.escape(ep) + r'\b', str(handoff)) else 0.0

df['fnp_recomputed'] = df.apply(lambda r: fnp_v2(r['handoff'], r['ep']), axis=1)

# Verify alignment with stored column
n_mismatch = (df['function_name_preserved'] != df['fnp_recomputed']).sum()
print(f"Stored vs recomputed mismatches: {n_mismatch}")

# --- Per-type table ---
print("\n=== Per-type fnp mean (483 labeled rows, recomputed) ===")
grp = df.groupby('corruption_type').agg(
    n=('fnp_recomputed', 'count'),
    mean_fnp=('fnp_recomputed', 'mean'),
    std_fnp=('fnp_recomputed', 'std'),
    n_label_1=('label', 'sum'),
).reset_index()
grp['n_label_0'] = grp['n'] - grp['n_label_1']
grp = grp.sort_values('mean_fnp', ascending=False)
print(grp.to_string(index=False))

# --- Overall AUROC on all 483 rows ---
print("\n=== Overall single-feature AUROC (all 483 rows) ===")
auc_all = roc_auc_score(df['label'], df['fnp_recomputed'])
ci_all, n_valid = task_bootstrap_ci(df, df['fnp_recomputed'].values, roc_auc_score, n_boot=10_000, seed=42)
print(f"AUROC={auc_all:.4f}  95% CI=[{ci_all[0]:.4f}, {ci_all[1]:.4f}]  (n_boot_valid={n_valid})")

# --- clean vs signature_rename AUROC ---
print("\n=== Single-feature AUROC: clean vs signature_rename ===")
mask_sr = df['corruption_type'].isin(['none', 'signature_rename'])
sub_sr = df[mask_sr].copy()
auc_sr = roc_auc_score(sub_sr['label'], sub_sr['fnp_recomputed'])
ci_sr, n_valid_sr = task_bootstrap_ci(sub_sr, sub_sr['fnp_recomputed'].values, roc_auc_score, n_boot=10_000, seed=42)
print(f"n_rows={len(sub_sr)} (clean={mask_sr.sum() - len(sub_sr[sub_sr.corruption_type=='signature_rename'])}, sr={len(sub_sr[sub_sr.corruption_type=='signature_rename'])})")
print(f"AUROC={auc_sr:.4f}  95% CI=[{ci_sr[0]:.4f}, {ci_sr[1]:.4f}]  (n_boot_valid={n_valid_sr})")

# --- clean vs each corruption type ---
print("\n=== Per-corruption AUROC (single feature, clean vs type) ===")
rows = []
for ctype in sorted(df['corruption_type'].unique()):
    if ctype == 'none':
        continue
    mask = df['corruption_type'].isin(['none', ctype])
    sub = df[mask].copy()
    if sub['label'].nunique() < 2:
        continue
    if sub['fnp_recomputed'].nunique() < 2:
        auc = float('nan')
        ci = (float('nan'), float('nan'))
        n_v = 0
    else:
        auc = roc_auc_score(sub['label'], sub['fnp_recomputed'])
        ci, n_v = task_bootstrap_ci(sub, sub['fnp_recomputed'].values, roc_auc_score, n_boot=10_000, seed=42)
    rows.append({'corruption_type': ctype, 'n': len(sub), 'AUROC': auc,
                 'CI_lo': ci[0], 'CI_hi': ci[1], 'n_boot_valid': n_v})

per_type_df = pd.DataFrame(rows)
print(per_type_df.to_string(index=False))

# --- old vs new on the 14 sr rows ---
print("\n=== 0.996 figure check: clean label=1 mean (old stored vs recomputed) ===")
clean_old = df[df['corruption_type'] == 'none']['function_name_preserved'].mean()
clean_new = df[df['corruption_type'] == 'none']['fnp_recomputed'].mean()
all_label1_old = df[df['label'] == 1]['function_name_preserved'].mean()
all_label1_new = df[df['label'] == 1]['fnp_recomputed'].mean()
all_label0_old = df[df['label'] == 0]['function_name_preserved'].mean()
all_label0_new = df[df['label'] == 0]['fnp_recomputed'].mean()
print(f"label=1 mean (stored):     {all_label1_old:.4f}  <- this is the 0.996 figure")
print(f"label=1 mean (recomputed): {all_label1_new:.4f}")
print(f"label=0 mean (stored):     {all_label0_old:.4f}  <- this is the 0.434 figure")
print(f"label=0 mean (recomputed): {all_label0_new:.4f}")
print(f"clean-only mean (stored):     {clean_old:.4f}")
print(f"clean-only mean (recomputed): {clean_new:.4f}")
