"""Turn simultaneous MR monitoring evidence into a reproducible analysis."""
import argparse
import json
from pathlib import Path
import numpy as np


def percent(n, d):
    return 100*n/max(d, 1)


def evidence_checks(data):
    """Prove rare snapshots and macro/plane histograms cover recorded MR stats."""
    checks = {}
    for name, report in data['conditions_by_layer'].items():
        hist = data['mr_by_layer'][name]['logical_peak_bits_histogram']
        for kind in ('peak_bits_by_macro', 'peak_bits_by_plane'):
            merged = {}
            for h in report[kind]:
                for bits, count in h.items():
                    merged[bits] = merged.get(bits, 0) + count
            assert merged == hist, (name, kind)
        lane_events = sum(int((np.asarray(s['mr']) >= 64).sum()) for s in report['events_7bit'])
        expected = sum(n for b, n in data['mr_by_layer'][name]['observed_bits_histogram'].items() if int(b) >= 7)
        assert lane_events == expected, (name, 'rare snapshot coverage')
        identities = {(s['sample'], s['channel'], s['position'], int(m), int(b))
                      for s in report['events_7bit'] for m, b in np.argwhere(np.asarray(s['mr']) >= 64)}
        unique_expected = sum(n for b, n in hist.items() if int(b) >= 7)
        assert len(identities) == unique_expected, (name, 'rare logical counter coverage')
        checks[name] = {'rare_lane_observations': lane_events, 'rare_logical_counters': len(identities),
                        'macro_and_plane_histograms_match_global': True}
    return checks


def make_report(data, source):
    checks = evidence_checks(data)
    events = [s for r in data['conditions_by_layer'].values() for s in r['events_7bit']]
    top = sorted([s for r in data['conditions_by_layer'].values() for s in r['top_cases']], key=lambda s: -s['max_mr'])
    lines = ['# 未做负载均衡的实际 Cluster：MR 大位宽监督分析', '',
             f'原始证据：`{source}`；状态 `{data["status"]}`。', '',
             f'FashionMNIST 测试集前 {data["samples"]} 张，T={max(data["steps"])}, batch={data["batch_size"]}。',
             '未改变权重、阈值、映射或硬复位机制，MR8 / wide_reference，不做截断。',
             f'ACC：{json.dumps(data["cluster"]["accuracy_by_steps"])}；预测相对原实验不一致数：{json.dumps(data["prediction_disagreements_vs_baseline"])}。',
             f'全部 MR 统计与原实验完全相同：`{data.get("all_mr_statistics_identical_to_baseline", "not compared: subset")}`。', '',
             '## 1. 口径和结构', '',
             '- 逻辑 MR 索引：(样本、输出通道、空间位置、Macro、bit-plane)。所有索引均从 0 开始，t 从 1 开始。',
             '- 每个输出神经元有 16×5 个逻辑 MR；本报告不声称有同样多并行物理 Cluster。',
             '- 每个 fold 后取同一时刻完整 MR/SCU 快照，先记录，再由下游 IF 发放清零。',
             '- 条件统计单位为“神经元×fold 观测”，同一个神经元可重复进入；位宽直方图单位为“MR lane×fold 观测”。',
             '- 多 fold 层的 `v_charged` 是本时间步所有 fold 及残差相加完成后的电压，不是中间 fold 的阈值比较。',
             '- 位平面系数是 `[1,2,4,8,-16]`，MR 本身均无符号；低四位不能称为仅对应正权重，负权重的补码低位也在其中。', '',
             '设从最近一次清零后累计的 PMAC 计数为 C，则 `MR=floor(C/16)`，`SCU=C mod 16`。',
             '因此大 MR 直接需要足够多的“输入发放 AND 权重该位为1”事件，以及在这些事件积累期间未被输出 IF 清零。',
             '但输出 IF 比较的是有符号归约后的电压（还含 bias/残差），不是最大的无符号 MR。', '',
             '对每个 Macro：`P_m=sum((16*MR_b+SCU_b)*2^b, b=0..3)`；`N_m=16*(16*MR_4+SCU_4)`。',
             '同 Macro 抵消量 `L=sum(min(P_m,N_m))`；跨 Macro 额外抵消量 `X=min(sum(P_m),sum(N_m))-L`。',
             '抵消量仅计一次；抵消比例为 `2*min(P,N)/(P+N)`，不是两个不同权重集合的比例。', '',
             '## 2. 大位宽出现条件（MR-only 条件汇总）', '',
             '| 层 / MR门槛 | 神经元-fold次数 | MR lane次数 | 平均未复位步数（范围） | 正负位平面均≥门槛 | 同Macro同时高 | 不同Macro同时高 | 平均抵消比例 | 抵消量中同Macro占比 | 当步发放 | 充电电压<0 |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, report in data['conditions_by_layer'].items():
        for threshold, s in report['threshold_conditions'].items():
            n = s['neuron_fold_observations']
            if not n:
                continue
            local = percent(s['local_cancelled_sum'], s['local_cancelled_sum']+s['cross_macro_cancelled_sum'])
            ratios = [percent(s[k], n) for k in ('both_sides_high', 'same_macro_both_high', 'different_macros_both_high')]
            lines.append(f'| {name} / {threshold} | {n:,} | {s["high_lane_observations"]:,} | {s["mean_age"]:.2f} ({s["age_min"]}–{s["age_max"]}) | {ratios[0]:.3f}% | {ratios[1]:.3f}% | {ratios[2]:.3f}% | {100*s["mean_cancellation_ratio"]:.3f}% | {local:.3f}% | {percent(s["fired_same_step"], n):.4f}% | {percent(s["negative_charged_voltage_count"], n):.4f}% |')
    lines += ['', '**“同 Macro 同时高”和“不同 Macro 同时高”可以重叠，不是相加等于100%的分类。**',
              '门槛32对应至少6bit；门槛48是6bit尾部诊断点；门槛64对应至少7bit。',
              '“正负均高”使用同一个原始 MR 数值门槛，不能代替加权后的 P/N 大小判断（符号位系数绝对值16）。', '',
              '全体神经元-step 的平均未复位步数（含低活动/不发放神经元）：', '']
    for name, report in data['conditions_by_layer'].items():
        lines.append(f'- {name}: {report["all_neuron_step_mean_age"]:.3f}')
    lines += ['', '## 3. 所有 7-bit 同时快照', '',
              '| 样本 | 层 | ch | 空间展平坐标 | t/fold | 最大MR / Macro / bit | age | 低4位最大MR | 符号位最大MR | ≥32的Macro数 | P (含SCU) | N (含SCU) | 净值 | 发放 | 充电电压 / 阈值 |',
              '|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for s in events:
        mr = np.array(s['mr'])
        m = s['with_scu']
        lines.append(f'| {s["sample"]} | {s["layer"]} | {s["channel"]} | {s["position"]} | {s["step"]}/{s["fold"]} | {s["max_mr"]} / {s["max_macro"]} / {s["max_bit"]} | {s["age_since_reset"]} | {mr[:, :4].max()} | {mr[:, 4].max()} | {(mr.max(1)>=32).sum()} | {m["positive"]} | {m["negative"]} | {m["net"]} | {s["fired"]} | {s["v_charged"]:.6f} / {s["if_threshold"]:.6f} |')
    lines += ['', f'共 {len(events)} 个神经元-fold 快照；覆盖 {sum(v["rare_lane_observations"] for v in checks.values())} 次高MR lane观测、{sum(v["rare_logical_counters"] for v in checks.values())} 个逻辑 MR，已与完整直方图逐项核对。', '']
    if top:
        s = top[0]
        lines += ['## 4. 全局最大 MR 的完整 16×5 同时状态', '',
                  f'样本 {s["sample"]}，{s["layer"]}，channel={s["channel"]}，position={s["position"]}，t={s["step"]}，fold={s["fold"]}。',
                  f'最大 MR={s["max_mr"]}，距上次清零 {s["age_since_reset"]} 步，上次发放 t={s["last_fire_step"]}。', '',
                  '| Macro | MR0 (+1) | MR1 (+2) | MR2 (+4) | MR3 (+8) | MR4 (-16) | P含SCU | N含SCU | 净值 |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for macro, values in enumerate(s['mr']):
            m = s['with_scu']
            lines.append('| '+str(macro)+' | '+' | '.join(map(str, values))+f' | {m["positive_by_macro"][macro]} | {m["negative_by_macro"][macro]} | {m["net_by_macro"][macro]} |')
        lines += ['', f'最热 lane 的权重1计数（跨fold求和）K={s["hot_lane_weight_ones_across_folds"]}；自复位以来每步平均有效PMAC计数={s["hot_lane_mean_pmac_per_step_since_reset"]:.4f}；对应输入活动比例={100*s["hot_lane_active_fraction_since_reset"]:.2f}%。',
                  f'权重缩放={s["weight_scale"]:.9f}，每步bias={s["bias_per_step"]:.9f}，累计bias={s["bias_since_reset"]:.6f}；实际充电电压={s["v_charged"]:.6f}，IF阈值={s["if_threshold"]:.6f}。',
                  '活动比例分母是 age×K；若快照来自非最后一个 fold，则是该步尚未完成全部 fold 的累计比例。', '']
        target_trace = [r for r in data.get('traces_by_layer', {}).get(s['layer'], []) if (r['sample'], r['channel'], r['position']) == (s['sample'], s['channel'], s['position'])]
        if target_trace:
            lines += ['## 5. 最大值神经元的从 t=1 开始的有符号权重追踪', '',
                      '此处单独按 **qweight>0 / qweight<0** 分解，避免把补码低位抵消误解为正负权重抵消。', '',
                      '| t | age | 最大MR | 真正正权重累计P | 真正负权重累计N | P−N | 累计bias | 充电电压 | 发放 |',
                      '|---:|---:|---:|---:|---:|---:|---:|---:|---|']
            for r in target_trace:
                if r['step'] not in {1, 8, 16, 24, 32, 40, 48, 56, 60, 61, 62, 63, 64, s['step']} and not r['fired']:
                    continue
                p, n = r['true_weight_positive_since_reset'], r['true_weight_negative_since_reset']
                lines.append(f'| {r["step"]} | {r["age_since_reset"]} | {r["max_mr"]} | {p} | {n} | {p-n} | {r["bias_since_reset"]:.6f} | {r["v_charged"]:.6f} | {r["fired"]} |')
            lines += ['', f'全部目标追踪中，`有符号bitplane归约−真正正负权重差` 的最大整数误差：{data["trace_max_integer_decomposition_error"]}。',
                      f'目标样本重跑预测与原实验不一致数：{data["trace_prediction_disagreements"]}。', '']
    lines += ['## 6. 各位平面的逻辑峰值分布', '',
              '聚合所有层，分母为该位平面的全部逻辑 MR（含补零）；列为百分比。', '',
              '| bit-plane / 系数 | 0bit | 1bit | 2bit | 3bit | 4bit | 5bit | 6bit | 7bit | 非零均值bit |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for plane, beta in enumerate((1, 2, 4, 8, -16)):
        hist = {}
        for report in data['conditions_by_layer'].values():
            for bits, count in report['peak_bits_by_plane'][plane].items():
                hist[int(bits)] = hist.get(int(bits), 0) + count
        total = sum(hist.values())
        nz = total-hist.get(0, 0)
        mean = sum(bits*count for bits, count in hist.items())/max(nz, 1)
        lines.append(f'| {plane} / {beta:+d} | '+' | '.join(f'{percent(hist.get(bits, 0), total):.9f}' for bits in range(8))+f' | {mean:.4f} |')
    lines += ['', '## 7. 位宽分布与适用范围', '',
              '全局、逐层、含零及排除零值的精确分布见同目录 `mr_width_distribution_t64_n128.md/.json/.png`。',
              '本监控 JSON 另外给出各层 `peak_bits_by_macro`（16组）和 `peak_bits_by_plane`（5组），均按逻辑峰值统计。',
              '所有 Macro 分组直方图之和、所有 bit-plane 分组直方图之和都已验证等于该层完整逻辑峰值直方图。',
              '只监督原映射，不运行负载均衡、不调整MR位宽/截断策略；本报告不是缩位宽后的ACC验证。',
              '128张/T=64的结果不提供全数据集、其他T或其他权重的最坏情况保证。', '',
              f'总耗时 {data.get("total_seconds", 0):.2f} 秒。', '']
    return '\n'.join(lines), checks


def plot_trace(data, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traces = [r for rows in data.get('traces_by_layer', {}).values() for r in rows]
    if not traces:
        return
    # Use the same tie-breaking as the written report (multiple maxima exist).
    cases = [s for r in data['conditions_by_layer'].values() for s in r['top_cases']]
    peak = max(cases, key=lambda r: r['max_mr'])
    rows = sorted([r for r in traces if (r['layer'], r['sample'], r['channel'], r['position']) == (peak['layer'], peak['sample'], peak['channel'], peak['position'])], key=lambda r: (r['step'], r['fold']))
    x = [r['step'] for r in rows]
    hot_m, hot_b = peak['max_macro'], peak['max_bit']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, [r['mr'][hot_m][hot_b] for r in rows], label=f'Hot MR: macro {hot_m}, bit {hot_b}', color='#2563eb')
    axes[0].plot(x, [max(m[4] for m in r['mr']) for r in rows], label='Maximum sign-plane MR', color='#d97706')
    axes[0].axhline(63, linestyle='--', color='red', label='6-bit limit (63)')
    axes[0].set_ylabel('Unsigned MR count')
    axes[1].plot(x, [r['true_weight_positive_since_reset'] for r in rows], label='True positive-weight total', color='#2563eb')
    axes[1].plot(x, [r['true_weight_negative_since_reset'] for r in rows], label='True negative-weight magnitude', color='#d97706')
    axes[1].set_ylabel('Integer weighted spikes')
    axes[2].plot(x, [r['v_charged'] for r in rows], label='Charged IF voltage', color='#2563eb')
    axes[2].axhline(peak['if_threshold'], linestyle='--', color='red', label='Firing threshold')
    axes[2].set_ylabel('Digital IF voltage')
    axes[2].set_xlabel('SNN time step')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc='best')
    fig.suptitle(f'Largest-MR trace | sample {peak["sample"]}, {peak["layer"]}, channel {peak["channel"]}, position {peak["position"]}')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    data = json.loads(args.source.read_text(encoding='utf8'))
    if data['status'] != 'complete':
        p.error('requires complete monitor and requested trace passes')
    report, checks = make_report(data, args.source.resolve())
    args.source.with_suffix('.md').write_text(report, encoding='utf8')
    args.source.with_name(args.source.stem+'_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf8')
    if args.plot:
        plot_trace(data, args.source.with_suffix('.png'))
    print(report)


if __name__ == '__main__':
    main()
