"""
New feature computation: parse expected function name from problem text t
rather than from ALL_TASKS metadata.

Strategy:
1. For each row, try to extract the expected function name from problem text t.
   - Look for def lines: def funcname(
   - Look for backtick patterns: `funcname` or `funcname()`
   - Look for quoted patterns: 'funcname' or "funcname" followed by (
   - Look for 'named X' or 'called X' or 'function X' patterns
2. If found, use that as the expected name (source: 'text').
3. If not found, set function_name_preserved = NaN and name_in_task = 0.
4. Add name_in_task flag: 1 if name was parseable from t, 0 if not.

This avoids using out-of-band metadata (ALL_TASKS entry_point).
"""
import pandas as pd, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.all_tasks import ALL_TASKS

ep_map = {t['id']: t.get('entry_point', '') for t in ALL_TASKS}

def extract_name_from_text(text):
    """Try to parse function name from problem text t.
    Returns (name, method) or (None, None) if not found.
    """
    if not text:
        return None, None
    text = str(text)

    # Priority 1: explicit def line
    m = re.search(r'\bdef\s+(\w+)\s*\(', text)
    if m:
        return m.group(1), 'def_line'

    # Priority 2: backtick identifier followed by (
    m = re.search(r'`(\w+)\s*\(', text)
    if m:
        return m.group(1), 'backtick_call'

    # Priority 3: 'named X' or 'called X' patterns
    m = re.search(r'\b(?:named|called)\s+[`\']?(\w+)[`\']?', text, re.IGNORECASE)
    if m:
        return m.group(1), 'named_pattern'

    # Priority 4: 'function X(' pattern
    m = re.search(r'\bfunction\s+[`\']?(\w+)[`\']?\s*\(', text, re.IGNORECASE)
    if m:
        return m.group(1), 'function_paren'

    # Priority 5: single backtick identifier (no parens)
    m = re.search(r'`(\w+)`', text)
    if m:
        return m.group(1), 'backtick_bare'

    return None, None

df = pd.read_csv(ROOT / 'final_dataset_full_features.csv')
df['ep_meta'] = df['task_id'].map(ep_map)

# Parse from problem text
parsed = df['problem'].apply(extract_name_from_text)
df['ep_text'] = [p[0] for p in parsed]
df['ep_text_method'] = [p[1] for p in parsed]

# Compute new fnp from text-parsed name
def fnp_from_text(handoff, ep_text):
    if not ep_text or not isinstance(ep_text, str):
        return float('nan')
    return 1.0 if re.search(r'\b' + re.escape(ep_text) + r'\b', str(handoff)) else 0.0

df['fnp_text'] = df.apply(lambda r: fnp_from_text(r['handoff'], r['ep_text']), axis=1)
df['name_in_task'] = df['ep_text'].notna().astype(int)

print("=== Extraction method breakdown (all 483 rows, unique tasks) ===")
tasks = df.drop_duplicates('task_id')
print(tasks.groupby(['source', 'ep_text_method']).size().reset_index(name='count').to_string(index=False))

print()
print("=== Coverage by source: how often name parseable from t ===")
grp = tasks.groupby('source').agg(
    n=('task_id', 'count'),
    name_found=('ep_text', lambda x: x.notna().sum()),
).copy()
grp['pct_found'] = (grp['name_found'] / grp['n'] * 100).round(1)
print(grp.to_string())

print()
print("=== Agreement: does text-parsed name match metadata entry_point? ===")
agreed = tasks[tasks['ep_text'].notna()].copy()
agreed['match'] = agreed.apply(
    lambda r: r['ep_text'] == r['ep_meta'] or
              r['ep_text'].lower() == r['ep_meta'].lower(),
    axis=1
)
print(f"Rows where name parseable: {len(agreed)} / {len(tasks)}")
print(f"Exact match with metadata ep: {agreed['match'].sum()} ({100*agreed['match'].mean():.1f}%)")

print()
print("=== Mismatches: text-parsed vs metadata ===")
mismatches = agreed[~agreed['match']][['task_id', 'source', 'ep_meta', 'ep_text', 'ep_text_method']]
print(mismatches.to_string(index=False))

print()
print("=== Per-type fnp mean (text-parsed name) on 483 rows ===")
print(df.groupby('corruption_type').agg(
    n=('fnp_text', 'count'),
    n_nan=('fnp_text', lambda x: x.isna().sum()),
    mean_fnp=('fnp_text', 'mean'),
).sort_values('mean_fnp', ascending=False).to_string())

print()
print("=== Clean rows with text-parsed fnp=0 ===")
clean = df[df['corruption_type'] == 'none']
print(f"clean total: {len(clean)}")
print(f"fnp_text=NaN: {clean['fnp_text'].isna().sum()}")
print(f"fnp_text=1.0: {(clean['fnp_text']==1.0).sum()}")
print(f"fnp_text=0.0: {(clean['fnp_text']==0.0).sum()}")
