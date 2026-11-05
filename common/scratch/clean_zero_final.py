"""Final clean-zero breakdown by source for the diagnostic note."""
import pandas as pd, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.all_tasks import ALL_TASKS

ep_map = {t['id']: t.get('entry_point', '') for t in ALL_TASKS}
df = pd.read_csv(ROOT / 'final_dataset_full_features.csv')
df['ep'] = df['task_id'].map(ep_map)

clean = df[df['corruption_type'] == 'none'].copy()
clean_zeros = clean[clean['function_name_preserved'] == 0.0].copy()

print('=== Clean-zero summary ===')
print(f'Total clean rows: {len(clean)}')
print(f'fnp=1.0: {(clean.function_name_preserved==1.0).sum()}  '
      f'({100*(clean.function_name_preserved==1.0).mean():.1f}%)')
print(f'fnp=0.0: {(clean.function_name_preserved==0.0).sum()}  '
      f'({100*(clean.function_name_preserved==0.0).mean():.1f}%)')

print()
print('--- By source ---')
src_counts = clean.groupby('source')['function_name_preserved'].agg(
    n='count',
    fnp1=lambda x: (x == 1.0).sum(),
    fnp0=lambda x: (x == 0.0).sum(),
).copy()
src_counts['pct_zero'] = (src_counts['fnp0'] / src_counts['n'] * 100).round(1)
print(src_counts.to_string())

print()
print('--- 5 MBPP zero examples: entry_point vs def line in handoff ---')
mbpp_zeros = clean_zeros[clean_zeros['source'] == 'mbpp']
mbpp_total = len(clean[clean['source'] == 'mbpp'])
print(f'MBPP zeros: {len(mbpp_zeros)} out of {mbpp_total} mbpp clean rows')
for _, row in mbpp_zeros.head(5).iterrows():
    h = str(row['handoff'])
    defs = re.findall(r'def\s+(\w+)', h)
    print(f'  ep={row["ep"]!r:30s}  ->  def: {defs}')

print()
print('--- All hand_easy zeros: entry_point vs def line ---')
he_zeros = clean_zeros[clean_zeros['source'] == 'hand_easy']
for _, row in he_zeros.iterrows():
    h = str(row['handoff'])
    defs = re.findall(r'def\s+(\w+)', h)
    print(f'  {row["task_id"]:12s}  ep={row["ep"]!r:30s}  ->  def: {defs}')

# Conclusion on 0.996 figure
print()
print('=== Where did 0.996 come from? ===')
# Check the original file in git or look at other numeric features
for col in df.select_dtypes('number').columns:
    m1 = df[df['label'] == 1][col].mean()
    m0 = df[df['label'] == 0][col].mean()
    if m1 > 0.9:
        print(f'  {col}: label=1 mean={m1:.4f}, label=0 mean={m0:.4f}')
