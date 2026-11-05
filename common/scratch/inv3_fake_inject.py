"""
Investigation 3: fake_completion and inject_false_constraint zeros.
Open 3 rows each, look at clean sibling's fnp value.
If sibling is also 0, the zero comes from alias-writing tasks, not the corruption.
"""
import pandas as pd, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.all_tasks import ALL_TASKS

ep_map = {t['id']: t.get('entry_point', '') for t in ALL_TASKS}
df = pd.read_csv(ROOT / 'final_dataset_full_features.csv')
df['ep'] = df['task_id'].map(ep_map)

clean_fnp = df[df['corruption_type'] == 'none'].set_index('task_id')['function_name_preserved'].to_dict()

for ctype in ['fake_completion', 'inject_false_constraint']:
    rows = df[df['corruption_type'] == ctype].copy()
    rows['clean_sibling_fnp'] = rows['task_id'].map(clean_fnp)
    print(f"=== {ctype} (n={len(rows)}) ===")
    print(f"All fnp values: {sorted(rows['function_name_preserved'].tolist())}")
    print(f"Clean sibling fnp values: {sorted(rows['clean_sibling_fnp'].tolist())}")
    print()
    print("Detailed row view (3 examples):")
    for _, row in rows.head(3).iterrows():
        h = str(row['handoff'])
        defs = re.findall(r'def\s+(\w+)', h)
        clean_h_rows = df[(df['task_id'] == row['task_id']) & (df['corruption_type'] == 'none')]
        clean_h_text = str(clean_h_rows['handoff'].iloc[0]) if len(clean_h_rows) > 0 else 'N/A'
        clean_defs = re.findall(r'def\s+(\w+)', clean_h_text)
        print(f"  task={row['task_id']}  ep={row['ep']!r}")
        print(f"  Corrupted fnp={row['function_name_preserved']}  defs: {defs}")
        print(f"  Clean sibling fnp={row['clean_sibling_fnp']}  defs: {clean_defs}")
        print(f"  Corrupted snip: {h[:120].replace(chr(10), ' ')}")
        print()

    # Paired comparison: same task, clean vs corrupted
    print(f"Paired: clean fnp 1 and corrupt fnp 0: "
          f"{((rows['clean_sibling_fnp'] == 1.0) & (rows['function_name_preserved'] == 0.0)).sum()}")
    print(f"Paired: both 0: "
          f"{((rows['clean_sibling_fnp'] == 0.0) & (rows['function_name_preserved'] == 0.0)).sum()}")
    print(f"Paired: both 1: "
          f"{((rows['clean_sibling_fnp'] == 1.0) & (rows['function_name_preserved'] == 1.0)).sum()}")
    print()
