"""Validation deliberately precedes dtype conversion: no float truncation."""
from numbers import Integral
import numpy as np

# 位平面按 LSB→MSB 排列；最高位是补码负权，不是独立正幅值位。
BETA = (1, 2, 4, 8, -16)


def integer_array(value, *, shape=None, low=None, high=None, name="array"):
    """先逐元素验证整数和范围，再复制转换；禁止悄悄截断浮点输入。"""
    a = np.asarray(value, dtype=object)
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    for v in a.flat:
        if not isinstance(v, Integral):
            raise ValueError(f"{name}: values must be integers, not {type(v).__name__}")
        if (low is not None and v < low) or (high is not None and v > high):
            raise ValueError(f"{name}: value {v} outside [{low}, {high}]")
    return np.array([int(v) for v in a.flat], dtype=object).reshape(a.shape)


def encode_weights(weights):
    """把任意形状的 int5 权重转为末轴长度为 5 的二值位平面。"""
    # &31 保留补码低五位；此处转 int64 安全，因为已约束到 [-16,15]。
    w = integer_array(weights, low=-16, high=15, name="weights").astype(np.int64)
    return ((w[..., None] & 31) >> np.arange(5)) & 1


def encode_weight(weight):
    return encode_weights(weight)


def decode_weight(bits):
    """按 (1,2,4,8,-16) 重建一个有符号五位权重。"""
    bits = integer_array(bits, shape=(5,), low=0, high=1, name="bits")
    return sum(int(bits[b]) * BETA[b] for b in range(5))


encode_int5 = encode_weight
decode_int5 = decode_weight
