"""Merge independently executed, matched ablation shards without duplicate cases."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DIRECTORIES=['shared_mr_sweep_v2','shared_mr_sweep_v2_stem','shared_mr_sweep_v2_rate']
NAMES=['mt_control','mt_mr_low','mt_mr_high','kd_control','kd_mr','stem_kd_control','stem_kd_mr','stem_kd_mr_rate']

def main():
    reports=[json.loads((ROOT/'SNN_ResNet10/checkpoints'/d/'summary.json').read_text()) for d in DIRECTORIES]
    baseline=reports[0]['baseline_accuracy']
    rows={}
    for r in reports:
        assert r['baseline_accuracy']==baseline and r['baseline_mr']==reports[0]['baseline_mr']
        for c in r['cases']:
            if c['name'] in rows: continue
            row={k:v for k,v in c.items() if k!='history'}
            row['validation_no_drop']=all(row['accuracy'][t]>=baseline[t] for t in ['16','64'])
            row['fits_9bit_stress']=row['mr']['minimum']>=-256 and row['mr']['maximum']<=255
            rows[c['name']]=row
    result=dict(status='complete' if all(n in rows for n in NAMES) else 'partial',baseline_accuracy=baseline,baseline_mr=reports[0]['baseline_mr'],cases=[rows[n] for n in NAMES if n in rows],validation_indices=[59000,59256],training='same first64 train, one epoch, seed20260914, lr0.0005; all multi-window T16/T64; grad ratio is regularizer/task norm; fixed BN; detached proxy reset')
    path=ROOT/'IMC_ResNet/checkpoints/shared_mr_sweep_results.json';path.write_text(json.dumps(result,indent=2),encoding='utf8')
    for c in result['cases']:print(c['name'],c['accuracy'],c['mr']['minimum'],c['mr']['maximum'],c['mr']['outside_9bit'],c['validation_no_drop'])

if __name__=='__main__': main()
