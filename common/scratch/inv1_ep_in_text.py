"""
Investigation 1: How often does entry_point appear in the problem text t?
Check per source. If MBPP is near 0%, we should parse from t instead.
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
df['problem_text'] = df['task_id'].map(prob_map)

def ep_in_t(ep, prob):
    if not ep or not prob:
        return False
    return bool(re.search(r'\b' + re.escape(ep) + r'\b', prob))

# Use unique tasks to avoid double-counting
tasks_df = df.drop_duplicates('task_id').copy()
tasks_df['ep_in_problem'] = tasks_df.apply(lambda r: ep_in_t(r['ep'], r['problem_text']), axis=1)

print("=== entry_point in problem text t — per source ===")
grp = tasks_df.groupby('source').agg(
    n_tasks=('task_id', 'count'),
    ep_in_t=('ep_in_problem', 'sum'),
).copy()
grp['pct'] = (grp['ep_in_t'] / grp['n_tasks'] * 100).round(1)
print(grp.to_string())

print()
# For MBPP tasks where ep NOT in problem: what does the problem text look like?
# and can we parse a function name from it?
mbpp_tasks = tasks_df[tasks_df['source'] == 'mbpp'].copy()
mbpp_no_ep = mbpp_tasks[~mbpp_tasks['ep_in_problem']]
print(f"MBPP tasks where ep not in problem text: {len(mbpp_no_ep)} / {len(mbpp_tasks)}")
print()
print("Sample MBPP: ep vs problem text (first 200 chars):")
for _, row in mbpp_no_ep.head(5).iterrows():
    prob = str(row['problem_text'])[:200].replace('\n', ' ')
    # Try to find any def or function mention
    defs_in_prob = re.findall(r'def\s+(\w+)', str(row['problem_text']))
    names_in_prob = re.findall(r'`(\w+)`', str(row['problem_text']))
    print(f"  {row['task_id']}  ep={row['ep']!r}")
    print(f"    problem: {prob}")
    print(f"    def lines in problem: {defs_in_prob}")
    print(f"    backtick names: {names_in_prob[:5]}")
    print()

# Also check: for HumanEval, does the problem always contain the ep?
he_tasks = tasks_df[tasks_df['source'] == 'humaneval']
print(f"HumanEval: ep in problem = {he_tasks['ep_in_problem'].sum()} / {len(he_tasks)}")
he_no = he_tasks[~he_tasks['ep_in_problem']]
for _, row in he_no.iterrows():
    prob = str(row['problem_text'])[:200].replace('\n', ' ')
    defs_in_prob = re.findall(r'def\s+(\w+)', str(row['problem_text']))
    print(f"  {row['task_id']}  ep={row['ep']!r}  defs: {defs_in_prob}")
    print(f"    problem: {prob}")
