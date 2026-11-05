"""Search git history for pre-recompute fnp values."""
import subprocess, pandas, io, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

for commit in ['decf452', '00ad67e', 'b7f0d34']:
    for fname in ['final_dataset_full_features.csv', 'features.csv']:
        result = subprocess.run(
            ['git', 'show', f'{commit}:{fname}'],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            cwd=ROOT
        )
        if result.stdout and 'fatal' not in result.stderr:
            try:
                df = pandas.read_csv(io.StringIO(result.stdout))
                if 'function_name_preserved' in df.columns:
                    uniq = sorted(df['function_name_preserved'].unique())[:6]
                    print(f'{commit}:{fname}  n={len(df)}  fnp_unique={uniq}')
                    if 'label' in df.columns:
                        m1 = df[df['label']==1]['function_name_preserved'].mean()
                        m0 = df[df['label']==0]['function_name_preserved'].mean()
                        print(f'  label=1 mean={m1:.4f}  label=0 mean={m0:.4f}')
            except Exception as e:
                print(f'{commit}:{fname}  parse error: {e}')
        else:
            if result.stderr:
                print(f'{commit}:{fname}  -> {result.stderr.strip()[:80]}')
