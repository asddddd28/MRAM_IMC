"""Replay existing QAT checkpoints under independent MR and two first-layer layouts."""
from pathlib import Path
from evaluate_mr_positive_forward import main, ROOT

if __name__ == '__main__':
    for name, checkpoint in [
        ('original', 'hard_reset_finetuned.pt'),
        ('kd_mr', 'shared_mr_sweep_v2/kd_mr.pt'),
        ('mt_mr_high', 'shared_mr_sweep_v2/mt_mr_high.pt'),
    ]:
        for rows in (36, 18):
            output = ROOT/'IMC_ResNet/checkpoints'/f'independent_transfer_{name}_{rows}.json'
            main(b2=True, checkpoint=ROOT/'SNN_ResNet10/checkpoints'/checkpoint,
                 output=output, rows=rows, best_only=True)
