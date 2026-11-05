import pandas as pd, numpy as np
df = pd.read_csv('features.csv')

feat = 'function_name_preserved'

print('=== function_name_preserved mean by corruption_type x label ===')
t = df.groupby(['corruption_type','label'])[feat].agg(['mean','count','std']).reset_index()
print(t.to_string())
print()

print('=== Value counts (all rows) ===')
print(df[feat].value_counts().to_string())
print()

sig = df[df['corruption_type']=='signature_rename']
clean = df[df['corruption_type']=='none']
print(f'signature_rename n={len(sig)}')
print('  value_counts:', sig[feat].value_counts().to_dict())
print(f'  mean={sig[feat].mean():.4f}')
print()
print(f'clean n={len(clean)}')
print('  value_counts:', clean[feat].value_counts().to_dict())
print(f'  mean={clean[feat].mean():.4f}')
print()

# Single-feature AUROC on {clean, signature_rename}
from sklearn.metrics import roc_auc_score
mask = df['corruption_type'].isin(['none','signature_rename'])
sub = df[mask]
auroc = roc_auc_score(sub['label'], sub[feat])
print(f'Single-feature AUROC (clean vs signature_rename): {auroc:.4f}')
print()

# What is the AUROC for each type separately?
print('=== Single-feature AUROC per corruption type (clean vs X) ===')
clean_rows = df[df['corruption_type']=='none']
for ctype in sorted(df[df['corruption_type']!='none']['corruption_type'].unique()):
    neg = df[df['corruption_type']==ctype]
    sub = pd.concat([clean_rows, neg])
    try:
        a = roc_auc_score(sub['label'], sub[feat])
        print(f'  {ctype:30s}  n_neg={len(neg):4d}  AUROC={a:.4f}  feat_mean_neg={neg[feat].mean():.4f}  feat_mean_clean={clean_rows[feat].mean():.4f}')
    except Exception as e:
        print(f'  {ctype:30s}  ERROR: {e}')
