"""Normalized edge/charge/reference/comparator chain (Reference sections 9-11)."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .config import AnalogConfig
from .digital import ModelError
from .events import Timing


@dataclass
class DeviceInstance:
    """一次芯片实例的固定失配；增益只读，save/load 可复现实验。"""
    chip_id: str
    chip_seed: int
    plane_gains: np.ndarray

    @classmethod
    def create(cls, cfg):
        rng = np.random.default_rng(cfg.chip_seed)
        gains = 1 + cfg.static_gain_sigma * rng.normal(size=(16, 5))
        if np.any(gains <= 0):
            raise ValueError("sampled nonpositive delay gain: choose a valid mismatch profile")
        return cls(cfg.chip_id, cfg.chip_seed, gains)

    def save(self, path):
        data = asdict(self)
        data["plane_gains"] = self.plane_gains.tolist()
        Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        data["plane_gains"] = np.asarray(data["plane_gains"], dtype=float)
        return cls(**data)

    def __post_init__(self):
        self.plane_gains = np.asarray(self.plane_gains, dtype=float).copy()
        if self.plane_gains.shape != (16, 5) or not np.all(np.isfinite(self.plane_gains)) or np.any(self.plane_gains <= 0):
            raise ValueError("device gains must be positive, finite, shape (16, 5)")
        self.plane_gains.setflags(write=False)


def event_normal(cfg, device, context, step, operation, size):
    """按事件标识生成动态噪声；预览/重试不消耗全局随机数状态。"""
    # Stable event IDs, independent of executor, loop order and Python hash seed.
    key = json.dumps([cfg.event_seed, device.chip_id, context, step, operation], separators=(",", ":"))
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little")
    return np.random.default_rng(seed).normal(size=size)


def advance_voltage(v, current, duration, capacitance, resistance=None):
    """Exact zero-baseline constant-current RC solution (including R=infinity)."""
    if not all(math.isfinite(x) for x in (v, current, duration, capacitance)) or duration < 0 or capacitance <= 0:
        raise ValueError("invalid duration/capacitance")
    if resistance is None:
        return v + current * duration / capacitance
    if not math.isfinite(resistance) or resistance <= 0:
        raise ValueError("resistance must be positive")
    # expm1 避免 duration/(R*C) 很小时 1-exp(-x) 的有效数字损失。
    fraction = -math.expm1(-duration / (resistance * capacitance))
    return v * (1 - fraction) + current * resistance * fraction


def share_charges(charges, capacitances, parasitic=0.0, *, wait=0.0, time_constant=0.0, initial_voltage=0.0):
    """所有接入电容共同分担电荷，零贡献 Macro 的电容也必须计入。"""
    q, c = np.asarray(charges, float), np.asarray(capacitances, float)
    if (q.shape != c.shape or q.size == 0 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(c))
            or not all(math.isfinite(x) for x in (parasitic, wait, time_constant, initial_voltage))
            or np.any(c <= 0) or parasitic < 0 or wait < 0 or time_constant < 0):
        raise ValueError("invalid sharing network")
    target = float(q.sum() / (c.sum() + parasitic))
    if time_constant == 0:
        return target
    return initial_voltage + (target - initial_voltage) * (-math.expm1(-wait / time_constant))


def convert(scu, theta_eff, cfg: AnalogConfig, device, *, context="default", step=0,
            previous_reference=0.0, timing=None, margin_injection=0.0):
    """No side effects: caller commits analog/digital state only after success.

    Scope: common-launch persistent edges and synchronized ideal sharing, plus
    a user-supplied sharing/comparison time. Arbitrary narrow-pulse XOR/switch
    topologies and calibrated metastability are deliberately not synthesized.
    """
    timing = timing or Timing()
    if not math.isfinite(margin_injection):
        raise ValueError("margin injection must be finite")
    # 1. TDP 双分支：只转换 SCU 余数；MR 已通过 theta_eff 进入参考端。
    a = np.asarray(scu, dtype=float)
    gains = np.asarray(cfg.plane_gains) * device.plane_gains
    p = (a[:, :4] * gains[:, :4]) @ np.array([1., 2., 4., 8.])
    n = 16 * a[:, 4] * gains[:, 4]
    common = cfg.common_delay + cfg.common_jitter_sigma * event_normal(cfg, device, context, step, "common_edge", 16)
    branch = cfg.branch_jitter_sigma * event_normal(cfg, device, context, step, "branch_edges", (16, 2))
    plus = cfg.base_delay + cfg.tau * p + cfg.positive_offset + common + branch[:, 0]
    minus = cfg.base_delay + cfg.tau * n + cfg.negative_offset + common + branch[:, 1]
    if np.any(plus < 0) or np.any(minus < 0) or not np.all(np.isfinite(plus + minus)):
        raise ModelError("invalid_edge_time", step=step, context=context)
    # 公共边沿项在相减时抵消，差分抖动与逐位固定失配不会自动抵消。
    delta = plus - minus
    tw, sign = np.abs(delta), np.sign(delta).astype(int)
    ambiguous = (tw > 0) & (tw <= cfg.sign_deadzone)
    if np.any(ambiguous):
        raise ModelError("ambiguous_sign", macros=np.where(ambiguous)[0].tolist(), step=step)
    pulse = np.where(tw < cfg.min_pulse_width, 0.0, tw)
    start, done = np.minimum(plus, minus), np.maximum(plus, minus)
    sign_valid = start + np.where(pulse > 0, cfg.sign_delay, 0.0)
    if np.any((pulse > 0) & (sign_valid > start)):
        raise ModelError("sign_unstable", macros=np.where((pulse > 0) & (sign_valid > start))[0].tolist(),
                         step=step, sign_valid=sign_valid.tolist(), charge_start=start.tolist())
    # 2. 时序检查：显式给定过早的共享/比较时刻时直接报错，不自动推迟。
    share_at = float(done.max()) if timing.share_at is None else timing.share_at
    if share_at < float(done.max()):
        raise ModelError("charge_not_ready", share_at=share_at, last_charge_done=float(done.max()), step=step)
    compare_at = share_at + cfg.share_wait if timing.compare_at is None else timing.compare_at
    if compare_at < share_at:
        raise ModelError("sharing_not_ready", compare_at=compare_at, share_at=share_at, step=step)
    hold = share_at - done
    # 3. 局部充电→保持→电荷共享。电容每步预充电，数字历史不能积分两次。
    local_v, q = np.zeros(16), np.zeros(16)
    for m, cap in enumerate(cfg.capacitances):
        # Precharge to zero every conversion; all history already lives in a/h.
        v = advance_voltage(0.0, sign[m] * cfg.current, pulse[m], cap, cfg.hold_resistance)
        v += cfg.charge_injection / cap
        if cfg.local_voltage_limit is not None and abs(v) > cfg.local_voltage_limit:
            raise ModelError("charge_voltage_out_of_range", macro=m, voltage=float(v),
                             limit=cfg.local_voltage_limit, step=step)
        local_v[m] = advance_voltage(v, 0.0, float(hold[m]), cap, cfg.hold_resistance)
        q[m] = local_v[m] * cap
    vs = share_charges(q, cfg.capacitances, cfg.parasitic_capacitance,
                       wait=compare_at-share_at, time_constant=cfg.share_time_constant)
    # 4. MR 修正参考：量化前后都检查范围，负码合法，不采用饱和裁剪。
    # Float conversion is confined to the analog model, never the integer oracle.
    try:
        code = float(theta_eff)
    except OverflowError as exc:
        raise ModelError("reference_out_of_range", theta_eff=theta_eff) from exc
    if not math.isfinite(code):
        raise ModelError("reference_out_of_range", theta_eff=theta_eff)
    def in_range(value, low, high):
        return low is None or low <= value <= high
    if not in_range(code, cfg.reference_code_min, cfg.reference_code_max):
        raise ModelError("reference_out_of_range", theta_eff=theta_eff, step=step)
    if cfg.reference_resolution:
        # Explicit round-to-nearest, ties-to-even code quantization; no clipping.
        code = round(code / cfg.reference_resolution) * cfg.reference_resolution
    if not in_range(code, cfg.reference_code_min, cfg.reference_code_max):
        raise ModelError("reference_out_of_range", quantized_code=code, step=step)
    gain = cfg.alpha if cfg.reference_gain is None else cfg.reference_gain
    target = gain * code + cfg.reference_offset
    if not math.isfinite(target) or not in_range(target, cfg.reference_voltage_min, cfg.reference_voltage_max):
        raise ModelError("reference_out_of_range", target_voltage=target, step=step)
    vr = target if cfg.reference_time_constant == 0 else previous_reference + (target-previous_reference) * (-math.expm1(-compare_at/cfg.reference_time_constant))
    if cfg.comparator_input_limit is not None and max(abs(vs), abs(vr)) > cfg.comparator_input_limit:
        raise ModelError("comparator_input_out_of_range", v_signal=vs, v_reference=vr, step=step)
    if cfg.comparator_timeout is not None and cfg.comparator_delay > cfg.comparator_timeout:
        raise ModelError("comparator_timeout", delay=cfg.comparator_delay, timeout=cfg.comparator_timeout, step=step)
    # 5. 比较：正 offset 降低 margin，即抬高门槛；margin=0 仍发放。
    noise = cfg.comparator_noise_sigma * float(event_normal(cfg, device, context, step, "comparator", 1)[0])
    margin = vs - vr - cfg.comparator_offset + noise + margin_injection
    if not math.isfinite(margin):
        raise ModelError("analog_numeric_range", step=step)
    diagnostics = []
    if np.any(pulse != tw):
        diagnostics.append({"code": "pulse_suppressed", "macros": np.where(pulse != tw)[0].tolist()})
    if cfg.share_time_constant and compare_at-share_at < 5*cfg.share_time_constant:
        diagnostics.append({"code": "sharing_unsettled", "wait": compare_at-share_at})
    if cfg.reference_time_constant and compare_at < 5*cfg.reference_time_constant:
        diagnostics.append({"code": "reference_unsettled", "wait": compare_at})
    # 参考网络在比较器延迟内继续建立；调用方仅在整个步骤成功后保存此值。
    reference_end = target if cfg.reference_time_constant == 0 else previous_reference + (target-previous_reference) * (-math.expm1(-(compare_at+cfg.comparator_delay)/cfg.reference_time_constant))
    return dict(t_plus=plus, t_minus=minus, tw=tw, effective_tw=pulse, sign=sign,
                sign_valid=sign_valid, charge_done=done, hold=hold, q_cap=q, local_voltage=local_v,
                v_signal=vs, theta_eff=theta_eff, reference_code=code, reference_target=target,
                v_reference=vr, reference_end=reference_end, margin=margin, fire=bool(margin >= 0), share_at=share_at,
                compare_at=compare_at, done_at=compare_at+cfg.comparator_delay,
                diagnostics=diagnostics)
