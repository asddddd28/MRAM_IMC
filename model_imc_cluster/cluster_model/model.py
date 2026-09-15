"""Compatibility imports; authoritative implementations live in the modules below."""
# 仅兼容旧导入，不允许在此复制或分叉计算实现。
from .config import ClusterConfig, AnalogConfig
from .digital import ClusterState
from .cluster import Cluster, StepResult
from .encoding import BETA, encode_weight, encode_weights, decode_weight
