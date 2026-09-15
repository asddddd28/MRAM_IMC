"""配置与边界校验：结构固定，实验位宽和归一化非理想参数显式配置。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ClusterConfig:
    """Reference v1.0 normalized IF profile; MR width is an experiment setting."""

    # 这三个字段用于记录规范结构，不意味着当前实现支持任意阵列尺寸。
    num_macros: int = 16
    inputs_per_macro: int = 36
    weight_bits: int = 5
    scu_bits: int = 4
    # MR 位宽是实验容量；wide_reference 可保留超过此容量的值并报告越界。
    mr_bits: int = 8
    overflow_policy: str = "error"
    mode: str = "ideal_behavioral"
    backend: str = "numpy"
    theta_count: int = 560
    leak_mode: str = "none"
    bias: int = 0
    reset_mode: str = "hard_zero"
    units: str = "normalized"

    def __post_init__(self):
        for name in ("num_macros", "inputs_per_macro", "weight_bits", "scu_bits", "mr_bits", "theta_count"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if (self.num_macros, self.inputs_per_macro, self.weight_bits) != (16, 36, 5):
            raise ValueError("Reference v1.0 fixes the structure at 16 x 36 x 5")
        if self.scu_bits not in (3, 4) or self.mr_bits < 1:
            raise ValueError("scu_bits must be 3/4 and mr_bits must be positive")
        if self.overflow_policy not in {"wide_reference", "error", "wrap", "saturate"}:
            raise ValueError("invalid overflow_policy")
        if self.mode not in {"integer_reference", "finite_digital", "ideal_behavioral", "nonideal_behavioral"}:
            raise ValueError("invalid mode; calibrated behavior is not implemented")
        if self.backend not in {"scalar", "numpy"}:
            raise ValueError("backend must be scalar or numpy")
        if self.leak_mode != "none" or self.bias != 0 or self.reset_mode != "hard_zero":
            raise ValueError("only no-leakage, zero-bias, hard-zero IF is specified")
        if self.units != "normalized":
            raise ValueError("only explicitly normalized units are supported")

    @property
    def L(self):
        """SCU 模数：低位余数 a 与高位 MR h 共同表示 a + L*h。"""
        return 1 << self.scu_bits

    @property
    def mr_max(self):
        return (1 << self.mr_bits) - 1

    @property
    def state_bits_per_context(self):
        """仅计算 16×5 路 SCU/MR 的逻辑位数，不是物理存储面积。"""
        return 80 * (self.scu_bits + self.mr_bits)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))


@dataclass(frozen=True)
class AnalogConfig:
    """Uncalibrated normalized parameters. None ranges mean ideal unlimited range.

    A first-order lumped sharing/hold model, NOT a switch-level circuit solver.
    Positive comparator offset increases the firing threshold.
    """

    # 时间/电流/电容均为归一化单位；不能据此推导真实芯片 PPA。
    tau: float = 1.0
    base_delay: float = 0.0
    current: float = 1.0
    capacitances: tuple[float, ...] = (1.0,) * 16
    parasitic_capacitance: float = 0.0
    plane_gains: tuple[float, ...] = (1.0,) * 5
    positive_offset: float = 0.0
    negative_offset: float = 0.0
    common_delay: float = 0.0
    # 固定芯片失配在设备创建时抽样；动态抖动按稳定事件键抽样。
    static_gain_sigma: float = 0.0
    common_jitter_sigma: float = 0.0
    branch_jitter_sigma: float = 0.0
    min_pulse_width: float = 0.0
    sign_deadzone: float = 0.0
    sign_delay: float = 0.0
    hold_resistance: float | None = None
    charge_injection: float = 0.0
    local_voltage_limit: float | None = None
    share_time_constant: float = 0.0
    share_wait: float = 0.0
    # None 增益取理想 alpha；None 上下限表示未限制量程，而不是自动裁剪。
    reference_gain: float | None = None
    reference_code_min: float | None = None
    reference_code_max: float | None = None
    reference_voltage_min: float | None = None
    reference_voltage_max: float | None = None
    reference_resolution: float = 0.0
    reference_offset: float = 0.0
    reference_time_constant: float = 0.0
    comparator_offset: float = 0.0
    comparator_noise_sigma: float = 0.0
    comparator_input_limit: float | None = None
    comparator_delay: float = 0.0
    comparator_timeout: float | None = None
    chip_seed: int = 0
    event_seed: int = 1
    chip_id: str = "normalized-chip"

    def __post_init__(self):
        for name, size in (("capacitances", 16), ("plane_gains", 5)):
            value = tuple(getattr(self, name))
            if len(value) != size or any(not math.isfinite(v) or v <= 0 for v in value):
                raise ValueError(f"{name} must have {size} positive finite values")
            object.__setattr__(self, name, value)
        positive = {"tau", "current", "hold_resistance", "local_voltage_limit", "reference_gain", "comparator_input_limit"}
        signed = {"positive_offset", "negative_offset", "charge_injection", "reference_offset", "comparator_offset",
                  "reference_code_min", "reference_code_max", "reference_voltage_min", "reference_voltage_max"}
        for f in fields(self):
            name, value = f.name, getattr(self, f.name)
            if name in {"chip_id", "chip_seed", "event_seed", "capacitances", "plane_gains"}:
                continue
            if value is None:
                continue
            if not math.isfinite(value) or (name in positive and value <= 0) or (name not in positive | signed and value < 0):
                raise ValueError(f"invalid {name}")
        for prefix in ("reference_code", "reference_voltage"):
            lo, hi = getattr(self, prefix + "_min"), getattr(self, prefix + "_max")
            if (lo is None) != (hi is None) or (lo is not None and lo > hi):
                raise ValueError(f"{prefix} requires a valid min/max pair")
        if not isinstance(self.chip_id, str) or not self.chip_id:
            raise ValueError("chip_id must be a nonempty stable string")
        if not math.isfinite(self.total_capacitance) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("derived analog capacitance/scale exceeds float64 range")
        if type(self.chip_seed) is not int or type(self.event_seed) is not int or min(self.chip_seed, self.event_seed) < 0:
            raise ValueError("seeds must be nonnegative integers")

    @property
    def total_capacitance(self):
        return sum(self.capacitances) + self.parasitic_capacitance

    @property
    def alpha(self):
        """计数到共享电压的理想比例 I*tau/C_total，参考端使用同一尺度。"""
        return self.current * self.tau / self.total_capacitance

    def is_ideal(self):
        # Unit scales, capacitances and common launch delay can vary ideally.
        allowed = {"tau", "current", "capacitances", "parasitic_capacitance", "base_delay", "common_delay",
                   "chip_seed", "event_seed", "chip_id"}
        default = AnalogConfig()
        return all(getattr(self, f.name) == getattr(default, f.name) for f in fields(self) if f.name not in allowed)
