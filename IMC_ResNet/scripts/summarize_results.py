"""Summarize completed runs only; never infer hardware statistics from events."""
import argparse
import json
from pathlib import Path


def summary(data):
    if data.get('status') != 'complete' or data.get('schema') != 'actual_cluster_v1':
        raise ValueError('expected a completed actual_cluster_v1 result')
    observed, peaks = {}, {}
    for stats in data['mr_by_layer'].values():
        for target, source in [(observed, stats['observed_bits_histogram']),
                               (peaks, stats['logical_peak_bits_histogram'])]:
            for bits, count in source.items():
                target[bits] = target.get(bits, 0) + count
    def avg(hist):
        return sum(int(k) * v for k, v in hist.items()) / sum(hist.values())
    return {
        'max_mr': max(s['max_mr'] for s in data['mr_by_layer'].values()),
        'max_required_bits': max(s['max_required_bits'] for s in data['mr_by_layer'].values()),
        'mean_observed_bits': avg(observed),
        'mean_logical_peak_bits': avg(peaks),
        'over_limit_fold_observations': sum(s['over_limit_fold_observations'] for s in data['mr_by_layer'].values()),
        'logical_mrs_over_limit': sum(s['logical_mrs_over_limit'] for s in data['mr_by_layer'].values()),
        'observed_bits_histogram': observed,
        'logical_peak_bits_histogram': peaks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding='utf8'))
    stats = summary(data)
    ts = [str(t) for t in data['steps']]
    rows = ['# 实际 Cluster 网络测试结果', '',
            (f"测试样本：{data['samples']}；同子集 ANN ACC：{data['ann_accuracy']:.2%}。"
             if 'ann_accuracy' in data else
             f"测试样本：{data['samples']}；来源：{data.get('model_source', '未注明')}。"),
            f"MR 统计窗口：T=1..{max(data['steps'])}，每 fold 清零前采样；零为 0 有效位，包含补零 lanes。", '',
            '| 模型 | ' + ' | '.join('T=' + t for t in ts) + ' |',
            '|---|' + '---:|' * len(ts)]
    for name, report in list(data['controls'].items()) + [('actual_cluster', data['cluster'])]:
        rows.append('| ' + name + ' | ' + ' | '.join(f"{report['accuracy_by_steps'][t]:.2%}" for t in ts) + ' |')
    rows += ['', '## MR 位宽', '', '| 层 | 最大 MR | 最大位宽 | 平均观测位宽 | 平均逻辑峰值位宽 | 超限观测次数 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, s in list(data['mr_by_layer'].items()) + [('全局（按 MR 观测数量加权）', stats)]:
        rows.append(f"| {name} | {s['max_mr']} | {s['max_required_bits']} | {s['mean_observed_bits']:.4f} | {s['mean_logical_peak_bits']:.4f} | {s['over_limit_fold_observations']} |")
    rows += ['', f"实际 Cluster 推理耗时：{data['cluster']['seconds']:.2f} s（含 MR 全量统计，不含校准/对照）。",
             f"与 int5 数字对照的预测不一致数：`{data['prediction_disagreements_vs_int5_digital']}`。",
             '', '完整逐输出通道/Macro/bit-plane 峰值表见同名原始 JSON。',
             '这只是固定小子集的调试结果，不是完整 FashionMNIST 测试集成绩，也不是硬件位宽的全输入保证。']
    args.result.with_suffix('.md').write_text('\n'.join(rows) + '\n', encoding='utf8')
    args.result.with_name(args.result.stem + '_summary.json').write_text(json.dumps(stats, indent=2), encoding='utf8')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
