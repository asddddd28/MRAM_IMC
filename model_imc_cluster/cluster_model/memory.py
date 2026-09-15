"""Logical backing store only: no claim about physical Mem. Array organization."""
from dataclasses import dataclass
from .digital import ClusterState, ModelError
from .encoding import integer_array


@dataclass
class ContextRecord:
    """神经元私有数字状态与逻辑步；物理时间/参考电压属于复用的 Cluster。"""
    state: ClusterState
    logical_step: int = 0


class LogicalMemory:
    """分离权重行与状态上下文的逻辑后备存储，不假定真实 MRAM 地址布局。"""
    def __init__(self, config, *, state_capacity_bits=None):
        if state_capacity_bits is not None and (type(state_capacity_bits) is not int or state_capacity_bits < 0):
            raise ValueError("state_capacity_bits must be a nonnegative integer or None (unknown)")
        self.config = config
        self.state_capacity_bits = state_capacity_bits
        self.weights = {}
        self.states = {}

    def write_weights(self, line_id, values):
        self.weights[line_id] = integer_array(values, shape=(16, 36), low=-16, high=15, name="weight row")

    def read_weights(self, line_id):
        if line_id not in self.weights:
            raise KeyError(f"missing logical weight line: {line_id}")
        return self.weights[line_id].copy()

    def ensure_capacity(self, state_id):
        if state_id not in self.states and self.state_capacity_bits is not None:
            required = (len(self.states)+1) * self.config.state_bits_per_context
            if required > self.state_capacity_bits:
                raise ModelError("state_capacity_exceeded", context=state_id, required_bits=required,
                                 capacity_bits=self.state_capacity_bits)

    def read_state(self, state_id):
        """读取副本；未分配上下文返回全零状态，但不占用新状态槽。"""
        self.ensure_capacity(state_id)
        record = self.states.get(state_id)
        if record is None:
            return ContextRecord(ClusterState.zeros(self.config))
        return ContextRecord(record.state.copy(), record.logical_step)

    def write_state(self, state_id, state, logical_step):
        self.ensure_capacity(state_id)
        if type(logical_step) is not int or logical_step < 0:
            raise ValueError("logical_step must be a nonnegative integer")
        self.states[state_id] = ContextRecord(state.validated(self.config), logical_step)

    def reset_context(self, state_id):
        # Weight rows and all other logical contexts remain untouched.
        record = self.read_state(state_id)
        self.write_state(state_id, ClusterState.zeros(self.config), record.logical_step)

    def run_context(self, cluster, state_id, input_tiles, weight_line_ids, **kwargs):
        """换入一个逻辑上下文完成全部输入折，再写回；预览和失败恢复原工作状态。"""
        if cluster.config != self.config:
            raise ValueError("memory and Cluster configurations differ")
        if not isinstance(state_id, str):
            raise ValueError("state_id must be a stable string")
        inputs, lines = list(input_tiles), list(weight_line_ids)
        if len(inputs) != len(lines) or not inputs:
            raise ValueError("one weight line is required per nonempty input tile")
        weights = [self.read_weights(line) for line in lines]
        record = self.read_state(state_id)
        # 仅切换神经元私有状态/逻辑步，参考电压与物理时间沿同一物理设备连续演化。
        snapshot = cluster.snapshot()
        cluster.state, cluster.logical_step = record.state, record.logical_step
        try:
            result = cluster.step_tiles(zip(inputs, weights), context=state_id, **kwargs)
            if kwargs.get("commit", True):
                self.write_state(state_id, cluster.state, cluster.logical_step)
            else:
                # 无提交预览既不分配上下文，也不能把预览对象留在物理工作槽中。
                cluster.restore(snapshot)
        except Exception:
            cluster.restore(snapshot)
            raise
        return result
