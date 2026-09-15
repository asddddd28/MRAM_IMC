"""固定步长 TDP 波形；只负责时序观察，不参与 SimpleCluster 的整数判决。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PulseEvent:
    slot: int
    polarity: int
    amplitude: int = 1
    source_bit: int | None = None

@dataclass(frozen=True)
class TDPWaveform:
    samples: np.ndarray
    step: float = 1.0
    polarity: int = 1
    events: tuple[PulseEvent, ...] = ()
    @property
    def duration(self): return self.samples.size * self.step
    @property
    def pulse_count(self): return int(np.abs(self.samples).sum())

def _check_step(step):
    if not isinstance(step, (int, float)) or not np.isfinite(step) or step <= 0: raise ValueError("step must be positive and finite")

def eigen_train(value: int, *, bits=5, step=1.0):
    """按论文 eigen-train 规则生成 2**bits 个 slot 的多电平波形。"""
    if type(value) is not int or type(bits) is not int or bits < 1 or not 0 <= value < (1 << bits): raise ValueError("invalid unsigned value/bits")
    _check_step(step); slots = 1 << bits; samples = np.zeros(slots, dtype=np.int16); events=[]
    for bit in range(bits):
        if value & (1 << bit):
            period = slots >> bit
            for slot in range(0, slots, period):
                samples[slot] += 1; events.append(PulseEvent(slot, 1, 1, bit))
    return TDPWaveform(samples, float(step), 1, tuple(events))

def signed_eigen_train(value: int, *, bits=5, step=1.0):
    if type(value) is not int or not -(1 << (bits-1)) <= value <= (1 << (bits-1))-1: raise ValueError("invalid signed value/bits")
    sign = 0 if value == 0 else (1 if value > 0 else -1); wave=eigen_train(abs(value), bits=bits, step=step)
    return TDPWaveform(wave.samples * sign, wave.step, sign, tuple(PulseEvent(e.slot, sign, e.amplitude, e.source_bit) for e in wave.events))

def scu_to_tdp(scu, *, bits=5, step=1.0):
    """把每一路 SCU 余数展开成逐 bit-plane、逐 slot 的 TDP 波形。

    返回 (16,5) 的波形表；MSB 仍作为独立的负分支，便于调试 TW/SG。
    """
    a=np.asarray(scu, dtype=object)
    if a.shape != (16,5): raise ValueError("scu must have shape (16,5)")
    if any(type(v) is not int or v < 0 or v >= (1 << bits) for v in a.flat): raise ValueError("invalid SCU")
    out=[]
    for macro in range(16):
        row=[]
        for bit in range(5):
            count=int(a[macro,bit]); polarity=-1 if bit == 4 else 1
            wave=eigen_train(count, bits=bits, step=step)
            row.append(TDPWaveform(wave.samples*polarity, wave.step, polarity, tuple(PulseEvent(e.slot,polarity,e.amplitude,bit) for e in wave.events)))
        out.append(row)
    return tuple(tuple(row) for row in out)
