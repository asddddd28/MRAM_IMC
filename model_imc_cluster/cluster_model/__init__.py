"""稳定公共 API；目录整理不改变 from cluster_model import ... 的用法。"""
from .config import ClusterConfig, AnalogConfig
from .encoding import BETA, encode_weight, decode_weight, encode_weights, encode_int5, decode_int5
from .digital import ClusterState, ModelError, MROverflowError, pmac, accumulate_tile, reduce_state
from .analog import DeviceInstance, advance_voltage, share_charges, convert
from .events import Event, EventSchedule, Timing
from .memory import LogicalMemory, ContextRecord
from .mapping import FCLayer, flat_address, unflatten_address, split_vector, map_fc, iter_conv_tiles
from .trace import json_safe, write_json, write_jsonl, first_divergence
from .tdp import PulseSequence, eigen_train, signed_eigen_train, tdp_sequence_from_count
from .cluster import Cluster, StepResult

__all__ = ["ClusterConfig", "AnalogConfig", "Cluster", "StepResult", "ClusterState", "DeviceInstance", "BETA",
           "encode_weight", "decode_weight", "encode_weights", "encode_int5", "decode_int5", "ModelError",
           "MROverflowError", "PulseSequence", "eigen_train", "signed_eigen_train", "tdp_sequence_from_count", "pmac", "accumulate_tile", "reduce_state", "advance_voltage", "share_charges",
           "convert", "Event", "EventSchedule", "Timing", "LogicalMemory", "ContextRecord", "FCLayer",
           "flat_address", "unflatten_address", "split_vector", "map_fc", "iter_conv_tiles", "json_safe",
           "write_json", "write_jsonl", "first_divergence"]
