"""Transactional Cluster executor: full input folds, one decision, one commit."""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from .config import ClusterConfig, AnalogConfig
from .encoding import encode_weights
from .digital import ClusterState, ModelError, pmac, accumulate_tile, reduce_state
from .analog import DeviceInstance, convert
from .events import EventSchedule, Timing


@dataclass
class StepResult:
    """一次完整逻辑步：candidate 是判决前值，state 是发放/保留后的值。

    即使 commit=False 也返回拟提交结果，但不会安装到 Cluster 中。
    analog=None 表示数字判决；events 只代表当前声明的功能时序。
    """
    spike: bool
    state: ClusterState
    candidate: ClusterState
    diagnostics: dict
    analog: dict | None
    events: list


class Cluster:
    """一个物理 Cluster 工作实例，负责跨折累积、唯一判决和原子提交。

    多逻辑神经元的状态隔离交给 LogicalMemory；本类仅持有当前工作状态。
    """
    def __init__(self, config=None, analog=None, device=None):
        self.config = config or ClusterConfig()
        self.analog = analog or AnalogConfig()
        self.device = device or DeviceInstance.create(self.analog)
        if self.config.mode != "nonideal_behavioral" and (
            not self.analog.is_ideal() or not np.all(self.device.plane_gains == 1)
        ):
            raise ValueError("nonideal parameters require mode='nonideal_behavioral'")
        self.state = ClusterState.zeros(self.config)
        self.logical_step = 0
        self.reference_state = 0.0
        self.physical_time = 0.0
        self.last_result = None
        self.last_error = None

    def reset(self):
        """Start a new simulation, unlike per-spike digital hard reset."""
        self.state = ClusterState.zeros(self.config)
        self.logical_step = 0
        self.reference_state = 0.0
        self.physical_time = 0.0
        self.last_result = None
        self.last_error = None

    def snapshot(self):
        """复制数字历史并保存逻辑步/物理时间/参考电压，不复制诊断日志。"""
        return dict(state=self.state.copy(), logical_step=self.logical_step,
                    reference_state=self.reference_state, physical_time=self.physical_time)

    def restore(self, snapshot):
        """校验后恢复模拟工作状态；last_result/last_error 不属于快照。"""
        state = snapshot["state"].validated(self.config)
        step = snapshot["logical_step"]
        ref, time = snapshot["reference_state"], snapshot["physical_time"]
        if not isinstance(step, Integral) or step < 0 or not np.isfinite(ref) or not np.isfinite(time) or time < 0:
            raise ValueError("invalid simulation snapshot")
        self.state, self.logical_step = state, int(step)
        self.reference_state, self.physical_time = float(ref), float(time)

    def step(self, inputs, weights=None, *, weight_bits=None, theta=None, context="default",
             margin_injection=0.0, timing=None, commit=True):
        """执行单折；输入形状 (16,36)，weights 与 weight_bits 必须二选一。"""
        if (weights is None) == (weight_bits is None):
            raise ValueError("provide exactly one of weights or weight_bits")
        bits = weight_bits if weight_bits is not None else encode_weights(weights)
        return self._execute([(inputs, bits)], theta=theta, context=context,
                             margin_injection=margin_injection, timing=timing, commit=commit)

    def step_tiles(self, tiles, *, theta=None, context="default", margin_injection=0.0,
                   timing=None, commit=True):
        """Each pair is (binary [16,36] input, int5 [16,36] weights).

        All actual per-fold MR updates are checked, including before a possible
        final reset. Iteration or conversion failure leaves committed state intact.
        """
        encoded = ((x, encode_weights(w)) for x, w in tiles)
        return self._execute(encoded, theta=theta, context=context,
                             margin_injection=margin_injection, timing=timing, commit=commit)

    def _execute(self, tiles, *, theta, context, margin_injection, timing, commit):
        if not isinstance(context, str):
            raise ValueError("context must be a stable string event ID")
        theta = self.config.theta_count if theta is None else theta
        if not isinstance(theta, Integral):
            raise ValueError("theta must be an integer count")
        theta = int(theta)
        if not np.isfinite(margin_injection):
            raise ValueError("margin_injection must be finite")
        if margin_injection and self.config.mode != "nonideal_behavioral":
            raise ValueError("controlled comparator fault requires nonideal_behavioral")
        # 事务开始：只更新独立候选。迭代器/数字/模拟阶段失败都不提交部分结果。
        candidate = self.state.validated(self.config)
        total = np.zeros((16, 5), dtype=object)
        folds, overflows = [], []
        schedule = EventSchedule(context)
        # 数字阶段缺少标定延迟，同一 now 上以依赖边表达先后，不虚构周期数。
        now = self.physical_time
        prev = schedule.schedule("context_load", now)
        prev = schedule.schedule("state_read_done", now, resource="state_backend", dependencies=(prev,))
        try:
            for fold, (inputs, bits) in enumerate(tiles):
                rd = schedule.schedule("weight_read_start", now, resource="weight_bank", dependencies=(prev,), fold=fold)
                rd = schedule.schedule("weight_read_done", now, resource="weight_bank", dependencies=(rd,), fold=fold)
                counts = pmac(inputs, bits, self.config.backend)
                prev = schedule.schedule("pmac_done", now, resource="pmac", dependencies=(rd,), fold=fold)
                candidate, carry, overflow = accumulate_tile(
                    candidate, counts, self.config, fold=fold, step=self.logical_step, context=context)
                prev = schedule.schedule("scu_update", now, resource="scu", dependencies=(prev,), fold=fold)
                prev = schedule.schedule("mr_update", now, resource="mr", dependencies=(prev,), fold=fold)
                total += counts
                overflows.extend(overflow)
                folds.append(dict(fold=fold, counts=counts.copy(), carry=carry.copy(),
                                  candidate_scu=candidate.scu.copy(), candidate_mr=candidate.mr.copy()))
            if not folds:
                raise ValueError("a logical step requires at least one input tile")
            # 所有折结束才归约和比较；禁止每折提前发放，否则正负抵消会失真。
            ready = schedule.schedule("candidate_ready", now, dependencies=(prev,))
            reduced = reduce_state(candidate, self.config)
            # U=S+L*G >= theta 等价于 S>=theta-L*G；负参考有效，不能截到零。
            theta_eff = theta - self.config.L * reduced["total_g"]
            analog = None
            if self.config.mode in {"integer_reference", "finite_digital"}:
                if timing is not None:
                    raise ValueError("digital mode does not consume physical timing")
                fire = reduced["total_u"] >= theta
                done = schedule.schedule("compare_done", now, resource="comparator", dependencies=(ready,))
                end_time = now
            else:
                analog = convert(candidate.scu, theta_eff, self.analog, self.device, context=context,
                                 step=self.logical_step, previous_reference=self.reference_state,
                                 margin_injection=margin_injection, timing=timing)
                fire = analog["fire"]
                launch = schedule.schedule("tdp_launch", now, resource="tdp", dependencies=(ready,))
                charge_ids = []
                for m in range(16):
                    plus = schedule.schedule("positive_edge", now + analog["t_plus"][m], resource=f"tdp+:{m}", dependencies=(launch,), macro=m)
                    minus = schedule.schedule("negative_edge", now + analog["t_minus"][m], resource=f"tdp-:{m}", dependencies=(launch,), macro=m)
                    sign = schedule.schedule("sign_valid", now + analog["sign_valid"][m], resource=f"sg:{m}", dependencies=(launch,), macro=m)
                    charge_ids.append(schedule.schedule("charge_done", now + analog["charge_done"][m], resource=f"charge:{m}", dependencies=(plus, minus, sign), macro=m))
                shared = schedule.schedule("switch_change", now + analog["share_at"], resource="sharing", dependencies=charge_ids,
                                           assumption="synchronized_ideal_assumption" if timing is None else "explicit_timing")
                ref = schedule.schedule("reference_evaluated", now + analog["compare_at"], resource="reference", dependencies=(ready,))
                cmp = schedule.schedule("compare_start", now + analog["compare_at"], resource="comparator", dependencies=(shared, ref))
                end_time = now + analog["done_at"]
                done = schedule.schedule("compare_done", end_time, resource="comparator", dependencies=(cmp,))
            if fire:
                done = schedule.schedule("hard_reset", end_time, dependencies=(done,))
            # 必须由实际比较器结果复位，非理想误发放也影响后续数字历史。
            committed = ClusterState.zeros(self.config) if fire else candidate.copy()
            done = schedule.schedule("state_commit", end_time, dependencies=(done,))
            done = schedule.schedule("state_write_start", end_time, resource="state_backend", dependencies=(done,))
            schedule.schedule("state_write_done", end_time, resource="state_backend", dependencies=(done,))
            diagnostics = dict(step=self.logical_step, context=context, counts=total, folds=folds,
                               overflow=overflows, theta=theta, theta_eff=theta_eff, **reduced,
                               reset=bool(fire), units="normalized", calibrated=False,
                               timing="functional_order_with_assumed_analog_delays; digital latencies unknown",
                               operation_counts=dict(weight_tile_reads=len(folds), pmac_bitplane_counts=80*len(folds),
                                                     scu_updates=80*len(folds), mr_carry_total=int(sum(sum(f["carry"].flat) for f in folds)),
                                                     comparisons=1, state_loads=1, state_commits=1, hard_resets=int(fire)))
            result = StepResult(bool(fire), committed.copy(), candidate.copy(), diagnostics, analog, schedule.events)
        except ModelError as exc:
            exc.diagnostics.update(context=context, logical_step=self.logical_step,
                                   candidate_scu=exc.diagnostics.get("candidate_scu", candidate.scu.tolist()),
                                   candidate_mr=exc.diagnostics.get("candidate_mr", candidate.mr.tolist()))
            self.last_error = exc.diagnostics
            raise
        # 唯一提交点：至此所有校验及事件构建成功；预览不会前进逻辑/物理时间。
        if commit:
            self.state = committed.copy()
            self.logical_step += 1
            self.physical_time = end_time
            if analog is not None:
                # 比较器延迟期间参考仍继续建立，保存结束值而非比较开始值。
                self.reference_state = analog["reference_end"]
            self.last_result = result
            self.last_error = None
        return result

    def run(self, inputs, weights, *, theta=None, context="default"):
        """对时间序列逐步提交；原子性限于每一步，不覆盖整个序列。"""
        results = [self.step(x, weights, theta=theta, context=context) for x in inputs]
        return np.asarray([r.spike for r in results], dtype=np.int64), results
