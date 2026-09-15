"""Reproducible CLI demonstrations; unimplemented train/calibration stays closed."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from .analog import DeviceInstance
from .cluster import Cluster
from .config import AnalogConfig, ClusterConfig
from .digital import ModelError
from .mapping import FCLayer
from .trace import write_json, write_jsonl


# 默认报告相对项目根目录，和调用者当前工作目录无关；显式路径仍按用户输入。
ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    """仅暴露已实现的参考/模拟/推理命令，不伪装开放训练或标定能力。"""
    parser = argparse.ArgumentParser(description="Five-bit MRAM-IMC Cluster normalized behavioral model")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reference", "simulate", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--steps", type=int, default=8)
        p.add_argument("--seed", type=int, default=20260909, help="data seed, distinct from chip/event seeds")
        p.add_argument("--config", type=Path, help="ClusterConfig JSON profile")
        p.add_argument("--analog-config", type=Path, help="AnalogConfig JSON (simulate only)")
        p.add_argument("--output", type=Path, help="output directory; defaults to reports/<command>")
        p.add_argument("--scu-bits", type=int, choices=[3, 4])
        p.add_argument("--backend", choices=["scalar", "numpy"])
        p.add_argument("--theta", type=int)
        p.add_argument("--overflow-policy", choices=["wide_reference", "error", "wrap", "saturate"])
    return parser


def demo_cluster(cfg, analog, args):
    """固定数据种子执行单 Cluster，对照标量整数参考并输出全状态 trace。"""
    rng = np.random.default_rng(args.seed)
    weights = np.zeros((16, 36), dtype=int)
    # Deliberately small active fan-in for a readable trace, signed weights.
    weights[:, :4] = rng.integers(-3, 8, (16, 4))
    inputs = np.zeros((args.steps, 16, 36), dtype=int)
    inputs[:, :, :4] = rng.integers(0, 2, (args.steps, 16, 4))
    device = DeviceInstance.create(analog)
    cluster = Cluster(cfg, analog, device)
    oracle = Cluster(replace(cfg, mode="integer_reference", backend="scalar"))
    results, expected = [], []
    for x in inputs:
        results.append(cluster.step(x, weights))
        expected.append(oracle.step(x, weights))
    if cfg.mode == "ideal_behavioral":
        for ref, hw in zip(expected, results):
            if ref.spike != hw.spike or not np.array_equal(ref.state.scu, hw.state.scu) or not np.array_equal(ref.state.mr, hw.state.mr):
                raise AssertionError("ideal behavior diverged from integer reference")
    write_json(args.output / "inputs.json", dict(weights=weights, inputs=inputs))
    write_jsonl(args.output / "trace.jsonl", results)
    device.save(args.output / "device.json")
    from .trace import first_divergence
    return dict(spikes=[int(r.spike) for r in results],
                candidate_u=[r.diagnostics["total_u"] for r in results],
                integer_spikes=[int(r.spike) for r in expected],
                first_divergence=first_divergence(expected, results),
                overflow_updates=sum(len(r.diagnostics["overflow"]) for r in results),
                max_candidate_mr=max(int(max(r.candidate.mr.flat)) for r in results),
                physical_time_normalized=cluster.physical_time,
                latency_scope="functional digital order; uncalibrated analog delays only")


def demo_infer(cfg, args):
    """8→4→2 IF 小网络，以独立矩阵 oracle 检查跨步发放与硬清零。"""
    rng = np.random.default_rng(args.seed)
    weights = [rng.integers(-2, 5, (4, 8)), rng.integers(-2, 5, (2, 4))]
    thresholds = [6, 4] if args.theta is None else [args.theta]*2
    configs = [replace(cfg, theta_count=t) for t in thresholds]
    layers = [FCLayer(w, c, layer_id=name) for w, c, name in zip(weights, configs, ["hidden", "output"])]
    inputs = rng.integers(0, 2, (args.steps, 2, 8))
    u = [np.zeros((2, 4), dtype=np.int64), np.zeros((2, 2), dtype=np.int64)]
    outputs, reference, selected_traces = [], [], []
    for n, data in enumerate(inputs):
        actual, expected = data, data
        for i, layer in enumerate(layers):
            u[i] += expected @ weights[i].T
            expected = (u[i] >= thresholds[i]).astype(int)
            u[i][expected.astype(bool)] = 0
            actual, traces = layer.forward(actual)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"FC inference differs from independent IF oracle at step {n}, layer {i}")
            # One selected context per layer avoids unnecessarily huge full traces.
            selected_traces.append(traces[0, 0])
        outputs.append(actual)
        reference.append(expected)
    write_json(args.output / "inputs.json", dict(inputs=inputs, weights=weights, thresholds=thresholds))
    write_jsonl(args.output / "trace.jsonl", selected_traces)
    return dict(network="8 -> 4 -> 2, two recurrent IF FC layers, batch=2", spikes=outputs,
                oracle_spikes=reference, oracle_match=True, first_divergence=None,
                logical_contexts=[len(layer.memory.states) for layer in layers],
                state_bits_per_context=cfg.state_bits_per_context,
                hardware_parallelism="software batching; no extra hardware lanes assumed")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 1 or args.seed < 0:
        parser.error("steps must be positive and seed must be nonnegative")
    cfg = ClusterConfig.load(args.config) if args.config else ClusterConfig()
    changes = {k: getattr(args, k) for k in ("scu_bits", "backend", "theta", "overflow_policy") if getattr(args, k) is not None}
    if "theta" in changes:
        changes["theta_count"] = changes.pop("theta")
    if args.command == "reference":
        changes["mode"] = "integer_reference"
        if args.backend is None:
            changes["backend"] = "scalar"
    analog = AnalogConfig()
    if args.analog_config:
        if args.command != "simulate":
            parser.error("--analog-config is only supported for simulate")
        analog = AnalogConfig(**json.loads(args.analog_config.read_text(encoding="utf-8-sig")))
        changes["mode"] = "nonideal_behavioral"
    cfg = replace(cfg, **changes)
    if args.command == "infer" and (cfg.mode not in {"integer_reference", "finite_digital", "ideal_behavioral"} or cfg.overflow_policy in {"wrap", "saturate"}):
        parser.error("the oracle-checked inference demo requires nonfaulting digital/ideal mode")
    args.output = args.output or ROOT / "reports" / args.command
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(reference_version="v1.0 (2026-09-09)", command=args.command, data_seed=args.seed,
                    steps=args.steps, digital=asdict(cfg), analog=asdict(analog),
                    calibrated=False, units="normalized", python=sys.version, numpy=np.__version__,
                    unknowns=["physical MR bit width/encoding", "Mem.Array row format and capacity", "real resource sharing",
                              "actual TW/SG switch waveforms/topology", "calibrated range/delay/noise/energy", "LIF leakage"])
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "summary.json", {"status": "running", "command": args.command})
    try:
        summary = demo_infer(cfg, args) if args.command == "infer" else demo_cluster(cfg, analog, args)
    except ModelError as exc:
        write_json(args.output / "error.json", exc.diagnostics)
        write_json(args.output / "summary.json", {"status": "failed", "code": exc.code})
        print(f"Simulation stopped: {exc.code}; diagnostic: {args.output / 'error.json'}", file=sys.stderr)
        return 2
    summary["status"] = "complete"
    write_json(args.output / "summary.json", summary)
    print(f"{args.command}: {args.steps} logical steps completed")
    print(f"summary: {args.output / 'summary.json'}")
    print("units=normalized; calibrated=false; no real-chip PPA claim")
    return 0
