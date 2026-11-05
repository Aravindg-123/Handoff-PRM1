"""
Inspect why clean rows have fnp=0:
- Look at a sample of hand_easy and mbpp clean zeros
- Check if function name appears in handoff via different surface forms
  (e.g. 'fibonacci' vs Fibonacci vs 'compute_fibonacci' as an alias)
"""
import pandas as pd, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.all_tasks import ALL_TASKS

ep_map  = {t['id']: t.get('entry_point', '') for t in ALL_TASKS}

df = pd.read_csv(ROOT / 'final_dataset_full_features.csv')
df['ep'] = df['task_id'].map(ep_map)
df['fnp_recomputed'] = df.apply(
    lambda r: (1.0 if re.search(r'\b' + re.escape(r['ep']) + r'\b', str(r['handoff'])) else 0.0)
    if r['ep'] else 0.5, axis=1)

clean_zeros = df[(df['corruption_type'] == 'none') & (df['fnp_recomputed'] == 0.0)]

# Show full handoff + entry_point for 6 hand_easy and 3 mbpp examples
hand_easy = clean_zeros[clean_zeros['source'] == 'hand_easy'].head(6)
mbpp = clean_zeros[clean_zeros['source'] == 'mbpp'].head(3)
sample = pd.concat([hand_easy, mbpp])

for _, row in sample.iterrows():
    ep = row['ep']
    h = str(row['handoff'])
    # search for any def line in the handoff
    defs = re.findall(r'def\s+(\w+)', h)
    # search for ep as case-insensitive
    ci_match = bool(re.search(r'\b' + re.escape(ep) + r'\b', h, re.IGNORECASE))
    print(f"=== {row['task_id']} [{row['source']}] | entry_point: {ep!r} ===")
    print(f"  def lines found: {defs}")
    print(f"  case-insensitive match: {ci_match}")
    # find lines containing ep letters
    for i, line in enumerate(h.split('\n')):
        if ep.replace('_', ' ').split()[0].lower() in line.lower() or ep.lower() in line.lower():
            print(f"  line {i}: {line.strip()[:100]}")
    print()
