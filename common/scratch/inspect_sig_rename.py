import pandas as pd

ch = pd.read_csv('corrupted_handoffs.csv')
print('corrupted_handoffs.csv shape:', ch.shape)
print('corruption_type counts:', ch['corruption_type'].value_counts().to_dict())
print('Has entry_point:', 'entry_point' in ch.columns)
print()

sig = ch[ch['corruption_type']=='signature_rename'].head(6)
for _, row in sig.iterrows():
    ep = str(row.get('entry_point', ''))
    h  = str(row.get('handoff', ''))
    wrong_name = ep + '_impl'
    orig_in  = ep in h if ep else False
    wrong_in = wrong_name in h
    print(f"task={row['task_id']}  ep={ep!r}")
    print(f"  orig_name_in_handoff: {orig_in}")
    print(f"  wrong_name_in_handoff ({wrong_name!r}): {wrong_in}")
    print(f"  handoff[:300]: {h[:300]!r}")
    print()
