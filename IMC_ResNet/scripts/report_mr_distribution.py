"""Build exact width distributions from completed Cluster runs (no inference).

Zero is assigned zero significant bits; this is not a physical register width.
Observation distributions weight every fold/time point. Logical-peak
histograms count each (sample, channel, position, macro, plane) only once.
"""
import argparse
import json
from pathlib import Path


def distribution(hist):
    hist = {int(k): int(v) for k, v in hist.items()}
    total = sum(hist.values())
    nonzero = total - hist.get(0, 0)
    return {
        'total': total, 'nonzero': nonzero,
        'mean_bits_including_zero': sum(k*v for k, v in hist.items())/max(total, 1),
        'mean_bits_nonzero': sum(k*v for k, v in hist.items())/max(nonzero, 1),
        'rows': [{'bits': k, 'min_value': 0 if k == 0 else 2**(k-1),
                  'max_value': 2**k-1, 'count': hist.get(k, 0),
                  'percent_all': 100*hist.get(k, 0)/max(total, 1),
                  'percent_nonzero': None if k == 0 else 100*hist.get(k, 0)/max(nonzero, 1)}
                 for k in range(max(hist, default=0)+1)]}


def collect(data):
    global_hist = {key: {} for key in ('observed_bits_histogram', 'logical_peak_bits_histogram')}
    layers = {}
    for name, stats in data['mr_by_layer'].items():
        layers[name] = {key: distribution(stats[key]) for key in global_hist}
        for key in global_hist:
            for bits, count in stats[key].items():
                global_hist[key][bits] = global_hist[key].get(bits, 0) + count
    return {'source_status': data['status'], 'samples': data['samples'], 'steps': data['steps'],
            'definition': 'all padded lanes included; 0 = zero significant bits, NOT a zero-bit physical MR',
            'global': {key: distribution(hist) for key, hist in global_hist.items()}, 'by_layer': layers}


def markdown(result, source):
    lines = ['# MR 有效位宽分布', '', f'数据源：`{source}`。', '',
             f'测试样本：{result["samples"]}，统计时间窗：T={max(result["steps"])}。', '',
             '零计为 0 个有效位；所有补零 lanes 均计入。位宽是无符号数的 bit_length，不是寄存器配置位宽。', '',
             '- **逐 fold 观测**：每个时间步、每个 fold 累加后、发放清零前统计，重复观察同一个 MR。',
             '- **逻辑 MR 峰值**：每个 (样本、输出通道、空间位置、Macro、bit-plane) 在整个时间窗的最大值，仅计一次。',
             '- 两者都不等于物理 MR 数量；排除零值也不等于排除所有映射补零位置的静态容量统计。', '']
    for key, label in [('observed_bits_histogram', '逐 fold 观测分布'), ('logical_peak_bits_histogram', '逻辑 MR 峰值分布')]:
        d = result['global'][key]
        lines += [f'## {label}', '', f'总数 {d["total"]:,}；非零 {d["nonzero"]:,}。',
                  f'含零均值 {d["mean_bits_including_zero"]:.6f} bit；非零均值 {d["mean_bits_nonzero"]:.6f} bit。', '',
                  '| 有效位宽 | MR 数值范围 | 数量 | 占全部 | 占非零 |', '|---:|---:|---:|---:|---:|']
        for row in d['rows']:
            nz = '—' if row['percent_nonzero'] is None else f'{row["percent_nonzero"]:.9f}%'
            lines.append(f'| {row["bits"]} | {row["min_value"]}–{row["max_value"]} | {row["count"]:,} | {row["percent_all"]:.9f}% | {nz} |')
        lines += ['']
    lines += ['## 各层峰值分布（占该层全部逻辑 MR，%）', '',
              '| 层 | 0 bit | 1 bit | 2 bit | 3 bit | 4 bit | 5 bit | 6 bit | 7 bit |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, layer in result['by_layer'].items():
        rows = {r['bits']: r for r in layer['logical_peak_bits_histogram']['rows']}
        lines.append('| '+name+' | '+' | '.join(f'{rows.get(k, {}).get("percent_all", 0):.6f}' for k in range(8))+' |')
    lines += ['', '## 解释边界', '',
              '极少数 7-bit 事件仍足以使统一 6-bit MR 超限；低均值不能作为安全位宽依据。',
              '此分布本身不能证明正负大 MR 同时出现，也不能确定抵消发生在同一 Macro 或不同 Macro；应查看同时状态监督报告。',
              '本次 128 张/T=64 的分布不是全测试集或任意时间窗的无溢出保证。', '']
    return '\n'.join(lines)


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ['#2563eb', '#d97706']
    for ax, key, title in zip(axes, ('observed_bits_histogram', 'logical_peak_bits_histogram'), ('Every-fold MR observations', 'Per-logical-MR sequence peaks')):
        rows = result['global'][key]['rows']
        x = np.array([r['bits'] for r in rows])
        ax.bar(x-.18, [r['percent_all'] for r in rows], .36, label='All lanes', color=colors[0])
        ax.bar(x+.18, [r['percent_nonzero'] or 0 for r in rows], .36, label='Nonzero only', color=colors[1])
        ax.set_yscale('log')
        ax.set_ylim(1e-9, 150)
        ax.set_xticks(x)
        ax.set_xlabel('Unsigned significant bits (zero = 0 bits)')
        ax.set_ylabel('Percentage (log scale)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.legend()
    fig.suptitle(f'Unbalanced actual Cluster | T={max(result["steps"])} | FashionMNIST first {result["samples"]} test images')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.source.read_text(encoding='utf8'))
    if data['status'] not in ('complete', 'monitor_complete', 'running_trace'):
        parser.error('requires a completed observation pass')
    result = collect(data)
    result['source'] = str(args.source.resolve())
    prefix = args.output_prefix or args.source.with_name(args.source.stem+'_width_distribution')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    prefix.with_suffix('.md').write_text(markdown(result, args.source.resolve()), encoding='utf8')
    if args.plot:
        plot(result, prefix.with_suffix('.png'))
    print(json.dumps(result['global'], indent=2))


if __name__ == '__main__':
    main()
