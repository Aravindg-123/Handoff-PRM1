"""
Analyze clean rows where function_name_preserved = 0.0 after fix.
Breaks down by source and whether the entry_point appears in the problem text.
"""
import pandas as pd, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.all_tasks import ALL_TASKS

ep_map  = {t['id']: t.get('entry_point', '') for t in ALL_TASKS}
prob_map = {t['id']: t.get('problem', '') for t in ALL_TASKS}

df = pd.read_csv(ROOT / 'final_dataset_full_features.csv')
df['ep'] = df['task_id'].map(ep_map)

def fnp_v2(handoff, ep):
    if not ep:
        return 0.5
    return 1.0 if re.search(r'\b' + re.escape(ep) + r'\b', str(handoff)) else 0.0

df['fnp_recomputed'] = df.apply(lambda r: fnp_v2(r['handoff'], r['ep']), axis=1)

clean = df[df['corruption_type'] == 'none'].copy()
clean_zeros = clean[clean['fnp_recomputed'] == 0.0].copy()

def ep_in_prob(task_id, ep):
    prob = prob_map.get(task_id, '')
    if not ep:
        return 'NO_EP'
    return 'yes' if re.search(r'\b' + re.escape(ep) + r'\b', prob) else 'no'

clean_zeros['ep_in_problem'] = clean_zeros.apply(
    lambda r: ep_in_prob(r['task_id'], r['ep']), axis=1)

src_col = 'source' if 'source' in clean_zeros.columns else 'label_source'

print(f'=== Clean rows: {len(clean)} total, fnp=0 count: {len(clean_zeros)} ===')
print(f'  fnp=1.0: {(clean.fnp_recomputed == 1.0).sum()}')
print(f'  fnp=0.0: {(clean.fnp_recomputed == 0.0).sum()}')
print(f'  fnp=0.5: {(clean.fnp_recomputed == 0.5).sum()}')

print()
print('=== By source x ep_in_problem ===')
summary = clean_zeros.groupby([src_col, 'ep_in_problem']).size().reset_index(name='count')
print(summary.to_string(index=False))

print()
print('=== Full row listing (clean zeros) ===')
cols = ['task_id', src_col, 'ep', 'ep_in_problem']
# also show first 60 chars of handoff to see if function name is implicit
clean_zeros['handoff_snip'] = clean_zeros['handoff'].str[:80].str.replace('\n', ' ')
display_cols = ['task_id', src_col, 'ep', 'ep_in_problem', 'handoff_snip']
print(clean_zeros[display_cols].to_string(index=False))
