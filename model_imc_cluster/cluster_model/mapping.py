"""Fixed reversible FC/Conv2D mapping; batching is software, not hardware lanes."""
from numbers import Integral
import numpy as np
from .cluster import Cluster
from .config import ClusterConfig
from .encoding import integer_array
from .memory import LogicalMemory

# 单折 fan-in；超出部分在时间上折叠，不增加硬件并行输出数。
CAPACITY = 16 * 36


def flat_address(index):
    """展平索引→(fold, macro, local)，Macro 内的 36 个输入连续排列。"""
    if not isinstance(index, Integral) or index < 0:
        raise ValueError("index must be a nonnegative integer")
    fold, rem = divmod(int(index), CAPACITY)
    macro, local = divmod(rem, 36)
    return fold, macro, local


def unflatten_address(fold, macro, local):
    """flat_address 的逆映射，先检查 Macro 和局部输入边界。"""
    if not all(isinstance(i, Integral) for i in (fold, macro, local)) or fold < 0 or not 0 <= macro < 16 or not 0 <= local < 36:
        raise ValueError("invalid mapped address")
    return int(fold)*CAPACITY + int(macro)*36 + int(local)


def split_vector(values, *, weights=False):
    """将非空向量零填充后分为 (folds,16,36)，补零不增加点积贡献。"""
    vector = integer_array(values, low=-16 if weights else 0, high=15 if weights else 1,
                           name="weights" if weights else "spikes")
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("mapping needs a nonempty 1D vector")
    folds = (vector.size + CAPACITY - 1) // CAPACITY
    padded = np.zeros(folds * CAPACITY, dtype=object)
    padded[:vector.size] = vector
    return padded.reshape(folds, 16, 36)


def map_fc(inputs, weights):
    """为单个 FC 输出生成所有输入/权重折；调用方必须整组执行再判决。"""
    x, w = split_vector(inputs), split_vector(weights, weights=True)
    if np.asarray(inputs).size != np.asarray(weights).size:
        raise ValueError("FC input and weight row lengths differ")
    return list(zip(x, w))


def iter_conv_tiles(inputs, kernel, *, stride=1, padding=0):
    """CHW input; OIHW kernel; flatten order is channel, kernel_y, kernel_x.

    Yields ((cout, y, x), [(input tile, weight tile), ...]). No bias/leakage.
    """
    x = integer_array(inputs, low=0, high=1, name="conv spikes")
    w = integer_array(kernel, low=-16, high=15, name="conv weights")
    if x.ndim != 3 or w.ndim != 4 or x.shape[0] != w.shape[1] or min(*x.shape, *w.shape) <= 0:
        raise ValueError("expected CHW input and compatible OIHW kernel")
    if type(stride) is not int or type(padding) is not int or stride < 1 or padding < 0:
        raise ValueError("invalid stride/padding")
    padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    kh, kw = w.shape[2:]
    oh, ow = (padded.shape[1]-kh)//stride+1, (padded.shape[2]-kw)//stride+1
    if min(oh, ow) <= 0:
        raise ValueError("kernel does not fit padded input")
    for channel in range(w.shape[0]):
        for y in range(oh):
            for x_pos in range(ow):
                patch = padded[:, y*stride:y*stride+kh, x_pos*stride:x_pos*stride+kw]
                yield (channel, y, x_pos), map_fc(patch.reshape(-1), w[channel].reshape(-1))


class FCLayer:
    """One physical Cluster model time-multiplexed across logical outputs."""
    def __init__(self, weights, config=None, *, analog=None, device=None, layer_id="fc", state_capacity_bits=None):
        self.weights = integer_array(weights, low=-16, high=15, name="FC weights")
        if self.weights.ndim != 2 or min(self.weights.shape) < 1:
            raise ValueError("FC weights must have shape (outputs, inputs)")
        self.cluster = Cluster(config or ClusterConfig(), analog, device)
        self.memory = LogicalMemory(self.cluster.config, state_capacity_bits=state_capacity_bits)
        self.layer_id = layer_id
        for out, row in enumerate(self.weights):
            for fold, tile in enumerate(split_vector(row, weights=True)):
                self.memory.write_weights((layer_id, out, fold), tile)

    def forward(self, inputs, *, sample_ids=None):
        """处理一个时间步；连续调用时用稳定 sample_ids 保持样本的历史归属。

        batch/output 顺序复用同一个 Cluster；每次 run_context 独立提交，
        本方法不承诺整层回滚。省略 sample_ids 时以 batch 位置作为样本身份。
        """
        x = integer_array(inputs, low=0, high=1, name="FC spikes")
        if x.ndim != 2 or x.shape[1] != self.weights.shape[1] or x.shape[0] < 1:
            raise ValueError("FC input shape must be (batch, inputs)")
        ids = list(range(x.shape[0])) if sample_ids is None else list(sample_ids)
        if len(ids) != x.shape[0] or len({str(i) for i in ids}) != len(ids):
            raise ValueError("sample_ids must be unique and match batch size")
        spikes = np.empty((len(x), len(self.weights)), dtype=np.int64)
        results = {}
        for batch, vector in enumerate(x):
            tiles = split_vector(vector)
            for out in range(len(self.weights)):
                context = f"{self.layer_id}/sample={ids[batch]}/out={out}"
                lines = [(self.layer_id, out, f) for f in range(len(tiles))]
                result = self.memory.run_context(self.cluster, context, tiles, lines)
                spikes[batch, out] = result.spike
                results[batch, out] = result
        return spikes, results

    def reset(self):
        self.memory.states.clear()
        self.cluster.reset()
