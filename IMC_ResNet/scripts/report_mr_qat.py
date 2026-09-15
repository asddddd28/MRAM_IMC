"""Summarize bounded MR-QAT experiments without conflating sample sets."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = ['# MR-aware QAT pilot — 2026-09-13', '',
            '联合回归：40 passed；日志 `SNN_ResNet10/checkpoints/mr_qat_tests.log`。', '',
            '## 实现与语义修复', '',
            '- MR 代理使用 16 Macro × 36 输入 × 5 个二补码 bit-plane；SCU 基数 16。',
            '- 每 fold 观察清零前 MR；无符号状态跨时间累积，输出 IF 发放后同步清零。MR 本身不泄漏。',
            '- 前向 bit-plane 和计数精确；反向使用高斯类别分布 STE、floor STE 和 surrogate spike reset。',
            '- 隐藏 Conv-BN 先融合再进行 int5 QAT；BN 固定，仅训练 layer1.conv1.weight。',
            '- 修复 IFNode.eval() 快捷路径绕过泄漏函数的问题；旧 leakage 推理成绩不可作为启用泄漏的成绩。',
            '- 统一加载器恢复 leakage、阈值、融合拓扑，并物化旧 QAT 的实际前向权重。',
            '', '## 数字模型：前 128 张测试样本', '',
            '每组从相同 leakage=0.99 检查点初始化，前 64 张训练样本、1 epoch、完整 T=64。',
            '同组所有 alpha 使用相同随机种子、数据顺序和空间采样序列。测试前缀仅作探索性评估。', '',
            '| 实验 | alpha | T=16 ACC | T=64 ACC | 训练采样最大 MR |',
            '|---|---:|---:|---:|---:|']
    audit = []
    for folder in ['mr_qat_t64', 'mr_qat_t64_lr2e4', 'mr_qat_t64_lr1e3', 'mr_qat_t64_stable', 'mr_qat_t64_balanced']:
        path = ROOT/'SNN_ResNet10/checkpoints'/folder/'summary.json'
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding='utf8'))
        cfg = d['config']
        label = f"lr={cfg['lr']}, positions={cfg['positions']}, beta={cfg['beta']}, gamma={cfg['gamma']}, detach_reset={cfg.get('detach_reset', False)}, balance={cfg.get('balance_mr_gradient', False)}"
        if folder == 'mr_qat_t64':
            a = d['baseline_accuracy']
            rows.append(f"| 部署一致初始化 | — | {a['16']:.2%} | {a['64']:.2%} | — |")
        for c in d['candidates']:
            a = c['accuracy']
            peak = max(h['max_mr'] for h in c['history'])
            rows.append(f"| {label} | {c['alpha']:g} | {a['16']:.2%} | {a['64']:.2%} | {peak} |")
        control_path = path.parent/'alpha_0.pt'
        if control_path.exists():
            control = torch.load(control_path, map_location='cpu', weights_only=False)['model']
            for c in d['candidates']:
                if not c['alpha']:
                    continue
                candidate = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)['model']
                def qweight(state):
                    w = state['layer1.conv1.weight']
                    scale = w.flatten(1).abs().amax(1).clamp_min(1e-12) / 15
                    return (w / scale[:, None, None, None]).round().to(torch.int32)
                q0, q1 = qweight(control), qweight(candidate)
                audit.append({'run': folder, 'alpha': c['alpha'],
                              'all_model_tensors_identical_to_control': all(torch.equal(v, candidate[k]) for k, v in control.items()),
                              'changed_int5_weights': int((q0 != q1).sum()),
                              'changed_bit_planes': int((((q0 ^ q1)[..., None] >> torch.arange(5)) & 1).sum())})
    rows += ['', '训练采样峰值不能与全空间 Cluster 峰值直接比较，也不是位宽安全证明。', '',
             '损失：CE + alpha×mean(softplus(max_lane(MR−56)))/56 + beta×mean(relu(max_all(MR−56)))/56 + gamma×quiet。',
             'quiet 是按未发放年龄加权的 overflow 项；alpha=0 时 beta/gamma 同时为 0。',
             'balance 轮按 CE/MR 梯度范数比对整个 MR 损失乘以 detached 系数（上限 1e5），逐 batch 保存实际系数与两种梯度范数。',
             '', '导出模型与同组零惩罚对照的整数权重变化：', '',
             '| 实验 | alpha | 全模型张量一致 | int5 权重改变数 | bit-plane 改变数 |',
             '|---|---:|---|---:|---:|']
    for item in audit:
        rows.append(f"| {item['run']} | {item['alpha']:g} | {item['all_model_tensors_identical_to_control']} | {item['changed_int5_weights']} | {item['changed_bit_planes']} |")
    (ROOT/'IMC_ResNet/checkpoints/mr_qat_weight_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    rows += [
             '', '## 实际 Cluster：8 张已知风险诊断样本', '',
             '固定索引 [0,1,2,3,4,5,64,115]，覆盖历史 128 张中全部 4 张出现 7-bit 事件的样本。',
             '映射全部 11 个隐藏卷积，T=64，所有空间位置和补零 lanes；6-bit 上限，wide_reference 记录原始溢出而不饱和。',
             '此集合按历史风险选择，其 ACC 不代表无偏测试精度。', '',
             '| 模型 | T=64 ACC | max MR | MR≥48 | MR≥64 | 超限逻辑 MR | int5 预测差异 T16/T64 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for label, suffix in [('原硬复位', 'original'), ('leakage 部署初始化', 'baseline'),
                           ('lr=2e-4 零惩罚对照', 'control'),
                           ('lr=1e-3 零惩罚对照', 'control_lr1e3'),
                           ('detach-reset 零惩罚对照', 'stable_control'),
                           ('梯度平衡组零惩罚对照', 'balanced_control'),
                           ('梯度平衡 MR 惩罚候选', 'balanced_candidate')]:
        path = ROOT/'IMC_ResNet/checkpoints'/f'mr_qat_{suffix}_stress8.json'
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding='utf8'))
        if d['status'] != 'complete':
            rows.append(f'| {label}（未完成） | — | — | — | — | — | — |')
            continue
        s = d['mr_summary']
        ge48 = sum(v['mr_ge48'] for v in d['mr_by_layer'].values())
        ge64 = sum(v['mr_ge64'] for v in d['mr_by_layer'].values())
        diff = d['prediction_disagreements_vs_int5_digital']
        rows.append(f"| {label} | {d['cluster']['accuracy_by_steps']['64']:.2%} | {s['max_mr']} | {ge48:,} | {ge64:,} | {s['logical_mrs_over_limit']} | {diff['16']}/{diff['64']} |")
    rows += ['', '完整平均有效位宽、逻辑峰值位宽直方图、逐层 firing rate 和 zero-firing ratio 见各 stress8 JSON。',
             '', '## 边界与后续', '',
             '- 这是首次 MR-aware QAT 小规模对照，不是全层、全数据训练，也未证明统一 6-bit MR 安全。',
             '- 新候选尚未完成前 128 张或完整测试集的全空间实际 Cluster 回归。',
             '- 历史原始模型 128 张结果（max MR=66、ACC=91.41%）保持原文件，不与上述 8 张 ACC 混用。',
             '- 未断开 IF reset 梯度的前三轮各组内，MR 候选与零惩罚对照导出模型相同。发现分类梯度范数约 1e8，MR 梯度被淹没。',
             '- stable 轮只断开 IF reset 的反向梯度，前向硬复位和 MR 清零保持不变；不把早期结果归因于 MR 惩罚。',
             '- stable 轮 CE 梯度仍约 1e4；balanced 轮显式平衡分类/MR 梯度，额外反向计算用于记录与缩放。',
             '- 下一阶段应在训练集上挖掘高 MR 的空间位置、增加样本和更新步数，并用独立验证集选择候选。',
             '- 继续保留零惩罚对照，先确认实际 bit-plane 与 MR 超限事件改善，再扩大实际 Cluster 回归。']
    final_path = ROOT/'IMC_ResNet/checkpoints/mr_qat_balanced_candidate_stress8.json'
    control_path = ROOT/'IMC_ResNet/checkpoints/mr_qat_balanced_control_stress8.json'
    if final_path.exists() and control_path.exists():
        final = json.loads(final_path.read_text(encoding='utf8'))
        control = json.loads(control_path.read_text(encoding='utf8'))
        if final['status'] == control['status'] == 'complete':
            count = lambda d, key: sum(s[key] for s in d['mr_by_layer'].values())
            training = json.loads((ROOT/'SNN_ResNet10/checkpoints/mr_qat_t64_balanced/summary.json').read_text(encoding='utf8'))
            candidates = {c['alpha']: c for c in training['candidates']}
            trained, baseline = candidates[10.]['accuracy'], candidates[0.]['accuracy']
            changes = next(a for a in audit if a['run'] == 'mr_qat_t64_balanced' and a['alpha'] == 10.)
            cfg = training['config']
            finding = ['## 本轮最终对照结论', '',
                       f"最后一组：detach-reset + 梯度平衡，lr={cfg['lr']}，alpha=10，beta={cfg['beta']}，gamma={cfg['gamma']}。",
                       f"前 {cfg['test_size']} 张数字测试：MR 候选 T=16 为 {trained['16']:.2%}、T=64 为 {trained['64']:.2%}；零惩罚对照为 {baseline['16']:.2%}、{baseline['64']:.2%}。",
                       f"候选相对对照改变 {changes['changed_int5_weights']} 个 int5 权重、{changes['changed_bit_planes']} 个 bit-plane。",
                       f"8 张风险诊断集：max MR {control['mr_summary']['max_mr']} → {final['mr_summary']['max_mr']}；"
                       f"MR≥48 {count(control, 'mr_ge48')} → {count(final, 'mr_ge48')}；"
                       f"MR≥64 {count(control, 'mr_ge64')} → {count(final, 'mr_ge64')}。",
                       f"候选与 int5 数字参考预测差异：{final['prediction_disagreements_vs_int5_digital']}。",
                       '诊断集结果不构成 128 张或全测试集位宽保证，当前只完成小规模 pilot。', '']
            rows[2:2] = finding
    path = ROOT/'IMC_ResNet/checkpoints/mr_qat_pilot_report.md'
    path.write_text('\n'.join(rows) + '\n', encoding='utf8')
    print(path)


if __name__ == '__main__':
    main()
