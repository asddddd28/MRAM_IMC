"""数字参考内核：二值 PMAC、逐折 SCU/MR 更新与精确有符号归约。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import ClusterConfig
from .encoding import BETA, integer_array


class ModelError(RuntimeError):
    """携带稳定错误码及定位信息，由执行器补充上下文，CLI 写入报告。"""
    def __init__(self, code, **diagnostics):
        self.code = code
        self.diagnostics = {"code": code, **diagnostics}
        super().__init__(f"{code}: {diagnostics}")


class MROverflowError(ModelError, OverflowError):
    """严格 MR 容量错误；即使本步最终可能发放，也不能掩盖逐折溢出。"""
    pass


@dataclass
class ClusterState:
    """一个逻辑神经元的独立五路历史，scu/mr 均为 (16,5)。

    object 数组保存 Python 整数；不能用全局带符号膜电位替代这些本地计数。
    外部读写使用副本，避免结果、候选状态和已提交状态共享可变存储。
    """
    # Python integer objects preserve the L0 unlimited-precision contract.
    scu: np.ndarray
    mr: np.ndarray

    @classmethod
    def zeros(cls, config=None):
        return cls(np.zeros((16, 5), dtype=object), np.zeros((16, 5), dtype=object))

    def copy(self):
        return ClusterState(self.scu.copy(), self.mr.copy())

    def validated(self, cfg):
        """返回经过校验的新状态，不在原数组上修正或裁剪越界值。"""
        scu = integer_array(self.scu, shape=(16, 5), low=0, high=cfg.L - 1, name="scu")
        mr = integer_array(self.mr, shape=(16, 5), low=0, name="mr")
        if cfg.overflow_policy != "wide_reference" and any(v > cfg.mr_max for v in mr.flat):
            raise ValueError("loaded MR exceeds configured capacity")
        return ClusterState(scu, mr)


def pmac(inputs, bits, backend="numpy"):
    """(16,36) 输入与 (16,36,5) 位权门控，生成 (16,5) 个 0..36 计数。"""
    x = integer_array(inputs, shape=(16, 36), low=0, high=1, name="inputs")
    b = integer_array(bits, shape=(16, 36, 5), low=0, high=1, name="weight_bits")
    if backend == "scalar":
        return np.array([[sum(int(x[m, i]) * int(b[m, i, p]) for i in range(36))
                          for p in range(5)] for m in range(16)], dtype=object)
    # The maximum is exactly 36; this reduction cannot overflow int64.
    return np.einsum("mi,mib->mb", x.astype(np.int64), b.astype(np.int64)).astype(object)


def accumulate_tile(state, counts, cfg, *, fold=0, step=0, context="default"):
    """只生成一折候选状态，不提交；溢出位置包含 macro/bit/fold/step。

    z=a+k，carry=z//L，a_new=z%L，h_new=h+carry。
    非回卷/饱和情况下逐路满足 a_new+L*h_new = a+L*h+k。
    """
    state = state.validated(cfg)
    k = integer_array(counts, shape=(16, 5), low=0, high=36, name="counts")
    if cfg.backend == "scalar":
        a, h, carry = (np.empty((16, 5), dtype=object) for _ in range(3))
        for m in range(16):
            for b in range(5):
                z = int(state.scu[m, b]) + int(k[m, b])
                carry[m, b], a[m, b] = divmod(z, cfg.L)
                h[m, b] = int(state.mr[m, b]) + carry[m, b]
    else:
        # Explicit preflight before any fixed-width arithmetic. No silent fallback.
        limit = np.iinfo(np.int64).max
        if any(int(h) + (int(a) + int(c)) // cfg.L > limit
               for a, h, c in zip(state.scu.flat, state.mr.flat, k.flat)):
            raise ModelError("int64_capacity_exceeded", step=step, fold=fold,
                             hint="use backend='scalar' for arbitrary precision")
        z = state.scu.astype(np.int64) + k.astype(np.int64)
        carry, a = (z // cfg.L).astype(object), (z % cfg.L).astype(object)
        h = (state.mr.astype(np.int64) + carry.astype(np.int64)).astype(object)
    # 保存策略处理前的候选值；wrap/saturate 是显式变体，不能隐藏原始越界。
    raw = ClusterState(a.copy(), h.copy())
    overflows = []
    for m, b in np.ndindex(16, 5):
        if h[m, b] > cfg.mr_max:
            overflows.append(dict(macro=m, bit=b, old=int(state.mr[m, b]),
                                  carry=int(carry[m, b]), value=int(h[m, b]),
                                  maximum=cfg.mr_max, step=step, fold=fold, context=context))
    if overflows and cfg.overflow_policy == "error":
        raise MROverflowError("mr_overflow", locations=overflows,
                              candidate_scu=raw.scu.tolist(), candidate_mr=raw.mr.tolist())
    if cfg.overflow_policy == "wrap":
        h = h % (cfg.mr_max + 1)
    elif cfg.overflow_policy == "saturate":
        h = np.minimum(h, cfg.mr_max)
    return ClusterState(a, h), carry, overflows


def reduce_state(state, cfg):
    """先逐 Macro 按补码位权归约，再求 S、G；精确总计数 U=S+L*G。

    该归约用于诊断和门槛修正，不替换或合并原来的 80 路数字历史。
    """
    # All weighted sums and MR corrections use Python integers, including in the
    # NumPy backend. int64 is only used for bounded PMAC and counter updates.
    s = np.array([sum(int(state.scu[m, b]) * BETA[b] for b in range(5)) for m in range(16)], dtype=object)
    c = np.array([sum(int(state.mr[m, b]) * BETA[b] for b in range(5)) for m in range(16)], dtype=object)
    u = s + cfg.L * c
    S, G = sum(s), sum(c)
    return dict(local_s=s, local_c=c, local_u=u, total_s=S, total_g=G, total_u=S + cfg.L * G)
