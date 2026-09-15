"""简化版 MRAM-IMC Cluster：纯整数推理与可观测调试内核。"""
from .model import SimpleCluster, SimpleConfig, SimpleResult, SimpleState
from .tdp import PulseEvent, TDPWaveform, eigen_train, signed_eigen_train, scu_to_tdp
__all__ = ["SimpleCluster", "SimpleConfig", "SimpleResult", "SimpleState", "PulseEvent", "TDPWaveform", "eigen_train", "signed_eigen_train", "scu_to_tdp"]
