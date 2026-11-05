"""
Investigation 4: Where did 0.996 come from?
Run the OLD substring check (entry_point in handoff) on 248 clean rows.
Check git history for pre-recompute CSV values.
"""
import pandas as pd, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.all_tasks import ALL_TASKS

ep_map = {t['id']: t.get('entry_point', '') for t in ALL_TASKS}
df = pd.read_csv(ROOT / 'final_dataset_full_features.csv')
df['ep'] = df['task_id'].map(ep_map)

clean = df[df['corruption_type'] == 'none'].copy()

# Old substring check
def fnp_old(handoff, ep):
    if not ep:
        return 0.5
    return 1.0 if ep in str(handoff) else 0.0

clean['fnp_old_substring'] = clean.apply(lambda r: fnp_old(r['handoff'], r['ep']), axis=1)
clean['fnp_current'] = clean['function_name_preserved']

print("=== Old substring check on 248 clean rows ===")
print(f"Old substring mean: {clean['fnp_old_substring'].mean():.4f}")
print(f"Current \\b regex mean: {clean['fnp_current'].mean():.4f}")
print(f"Old value counts: {dict(clean['fnp_old_substring'].value_counts())}")
print(f"Current value counts: {dict(clean['fnp_current'].value_counts())}")

# Could old code have produced 0.996?
all_rows_old = df.copy()
all_rows_old['fnp_old'] = all_rows_old.apply(lambda r: fnp_old(r['handoff'], r['ep']), axis=1)
lab1_old = all_rows_old[all_rows_old['label'] == 1]['fnp_old'].mean()
lab0_old = all_rows_old[all_rows_old['label'] == 0]['fnp_old'].mean()
print(f"\nOld substring on ALL 483 rows:")
print(f"  label=1 mean: {lab1_old:.4f}")
print(f"  label=0 mean: {lab0_old:.4f}")

print()
print("=== Git history for final_dataset_full_features.csv ===")
try:
    result = subprocess.run(
        ['git', 'log', '--oneline', '-10', '--', 'final_dataset_full_features.csv'],
        cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout or "(no git output)")
    print(result.stderr or "")
except Exception as e:
    print(f"git error: {e}")

print()
print("=== Git history for features.csv ===")
try:
    result = subprocess.run(
        ['git', 'log', '--oneline', '-10', '--', 'features.csv'],
        cwd=ROOT, capture_output=True, text=True
    )
    print(result.stdout or "(no git output)")
except Exception as e:
    print(f"git error: {e}")

# Check if there's a backup/original version of the CSV
print()
print("=== Backup/original CSV files ===")
for f in sorted(ROOT.glob('*.csv')):
    print(f"  {f.name}")
for f in sorted((ROOT / 'results').glob('**/*.csv')) if (ROOT / 'results').exists() else []:
    print(f"  results/{f.name}")
