"""Configuration, atomicity, precision, mapping and end-to-end inference tests."""
from dataclasses import replace
import math
import numpy as np
import pytest
from cluster_model import *


def single(w=1):
    x, weights = np.zeros((16, 36), int), np.zeros((16, 36), int)
    x[0, 0], weights[0, 0] = 1, w
    return x, weights


@pytest.mark.parametrize("kwargs", [
    {"num_macros": 8}, {"inputs_per_macro": 32}, {"weight_bits": 4}, {"scu_bits": 2},
    {"mr_bits": 0}, {"mr_bits": 3.5}, {"theta_count": 2.5}, {"overflow_policy": "clip"},
    {"mode": "calibrated_behavioral"}, {"backend": "gpu"}, {"units": "SI"},
    {"leak_mode": "LIF"}, {"bias": 1}, {"reset_mode": "subtract_theta"},
])
def test_configuration_rejects_undefined_semantics(kwargs):
    with pytest.raises(ValueError):
        ClusterConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"tau": 0}, {"current": -1}, {"capacitances": [1]*15}, {"plane_gains": [1]*4},
    {"hold_resistance": 0}, {"share_time_constant": -1}, {"comparator_offset": math.nan},
    {"reference_code_min": 0}, {"reference_voltage_min": 2, "reference_voltage_max": 1},
    {"chip_seed": -1}, {"event_seed": 1.5},
])
def test_analog_config_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        AnalogConfig(**kwargs)


@pytest.mark.parametrize("bad", [-17, 16, 1.2, math.nan, "3"])
def test_encoding_rejects_invalid_weights(bad):
    with pytest.raises(ValueError):
        encode_weight(bad)


def test_invalid_input_bits_weights_and_state_are_not_silently_truncated():
    c = Cluster()
    x, w = single()
    for values in (x.astype(float)+.25, x+2, x[:, :35]):
        with pytest.raises(ValueError):
            c.step(values, w)
    with pytest.raises(ValueError):
        c.step(x, w.astype(float))
    with pytest.raises(ValueError):
        c.step(x, w, weight_bits=encode_weights(w))
    with pytest.raises(ValueError):
        c.step(x)
    with pytest.raises(ValueError):
        c.step(x, w, theta=1.5)
    with pytest.raises(ValueError):
        c.step_tiles([])
    c.state.scu[0, 0] = 16
    with pytest.raises(ValueError):
        c.step(x, w)


def test_wide_reference_true_python_integer_precision_and_numpy_capacity_guard():
    huge = 1 << 100
    cfg = ClusterConfig(mode="integer_reference", backend="scalar", overflow_policy="wide_reference", theta_count=huge*100)
    c = Cluster(cfg)
    c.state.mr[0, 0] = huge
    r = c.step(*single(1))
    assert r.diagnostics["total_u"] == huge*16+1
    c2 = Cluster(replace(cfg, backend="numpy"))
    c2.state.mr[0, 0] = huge
    with pytest.raises(ModelError, match="int64_capacity_exceeded"):
        c2.step(*single(1))
    assert c2.state.mr[0, 0] == huge and c2.logical_step == 0
    # Counter fits int64, but its signed weighted reduction does not. Reduction
    # still uses Python integers; no int64 dot/reduction wraparound is possible.
    c3 = Cluster(replace(cfg, backend="numpy"))
    c3.state.mr[0, 4] = (1 << 63) - 1
    r3 = c3.step(*single(1))
    assert r3.diagnostics["total_u"] == 1 - 256*((1 << 63)-1)


def test_preview_is_repeatable_and_returned_arrays_do_not_alias_state():
    cfg = ClusterConfig(mode="nonideal_behavioral", theta_count=100)
    ac = AnalogConfig(base_delay=5, branch_jitter_sigma=.01)
    c = Cluster(cfg, ac)
    r1 = c.step(*single(3), commit=False)
    r2 = c.step(*single(3), commit=False)
    assert c.logical_step == 0 and not np.any(c.state.scu)
    assert c.physical_time == c.reference_state == 0
    np.testing.assert_array_equal(r1.analog["t_plus"], r2.analog["t_plus"])
    actual = c.step(*single(3))
    np.testing.assert_array_equal(r1.state.scu, actual.state.scu)
    np.testing.assert_array_equal(r1.analog["t_plus"], actual.analog["t_plus"])
    actual.state.scu[:] = 0
    actual.candidate.scu[:] = 0
    assert c.state.scu[0, 0] == 1
    assert c.state.scu[0, 1] == 1


def test_generator_failure_rolls_back_entire_fold_transaction():
    c = Cluster(ClusterConfig(theta_count=100))
    def tiles():
        yield single(3)
        raise RuntimeError("input producer failed")
    with pytest.raises(RuntimeError, match="producer failed"):
        c.step_tiles(tiles())
    assert c.logical_step == 0 and not np.any(c.state.scu)
    assert c.reference_state == 0


def test_ideal_mode_does_not_silently_allow_nonideal_or_fault_parameters():
    with pytest.raises(ValueError, match="nonideal"):
        Cluster(ClusterConfig(), AnalogConfig(comparator_offset=.1))
    with pytest.raises(ValueError, match="fault"):
        Cluster().step(*single(3), margin_injection=.1)


def test_digital_finite_mode_is_exact_and_does_not_use_float_comparator():
    cfg = ClusterConfig(mode="finite_digital", backend="scalar", mr_bits=100, theta_count=1 << 90)
    c = Cluster(cfg)
    c.state.mr[0, 0] = (1 << 86)-1
    c.state.scu[0, 0] = 15
    r = c.step(*single(1))
    assert r.diagnostics["total_u"] == 1 << 90 and r.spike
    assert r.analog is None


def test_reference_quantization_ranges_and_stateful_settling():
    cfg = ClusterConfig(mode="nonideal_behavioral", theta_count=3)
    ac = AnalogConfig(reference_resolution=2, reference_time_constant=2)
    c = Cluster(cfg, ac)
    # Integer code 3 rounds to 4 (ties-to-even), from previous voltage 0.
    r1 = c.step(*single(), timing=Timing(share_at=1, compare_at=2))
    vr1 = .25*(1-math.exp(-1))
    assert r1.analog["reference_code"] == 4
    assert r1.analog["v_reference"] == pytest.approx(vr1)
    r2 = c.step(*single(), timing=Timing(share_at=2, compare_at=2))
    assert r2.analog["v_reference"] == pytest.approx(vr1+(.25-vr1)*(1-math.exp(-1)))
    assert "reference_unsettled" in [d["code"] for d in r1.analog["diagnostics"]]
    limited = replace(ac, reference_code_min=-3, reference_code_max=3)
    c = Cluster(cfg, limited)
    with pytest.raises(ModelError, match="reference_out_of_range"):
        c.step(*single())  # rounded code 4 is not silently clipped back to 3


def test_comparator_offset_sign_and_pulse_suppression():
    cfg = ClusterConfig(mode="nonideal_behavioral", theta_count=1)
    assert Cluster(cfg).step(*single()).spike
    assert not Cluster(cfg, AnalogConfig(comparator_offset=.01)).step(*single()).spike
    r = Cluster(cfg, AnalogConfig(min_pulse_width=2)).step(*single())
    assert r.analog["tw"][0] == 1 and r.analog["effective_tw"][0] == 0
    assert r.analog["q_cap"][0] == 0 and not r.spike
    assert r.analog["diagnostics"][0]["code"] == "pulse_suppressed"


def test_logical_memory_capacity_explicit_and_alias_protection():
    cfg = ClusterConfig(theta_count=100)
    memory = LogicalMemory(cfg, state_capacity_bits=cfg.state_bits_per_context)
    c = Cluster(cfg)
    x, w = single(3)
    memory.write_weights("W", w)
    w[:] = 0
    assert memory.read_weights("W")[0, 0] == 3
    memory.run_context(c, "A", [x], ["W"])
    before = c.snapshot()
    with pytest.raises(ModelError, match="state_capacity_exceeded"):
        memory.run_context(c, "B", [x], ["W"])
    assert c.logical_step == before["logical_step"]
    assert list(memory.states) == ["A"]
    assert cfg.state_bits_per_context == 960


def test_mapping_address_round_trip_and_padding():
    for index in range(1200):
        assert unflatten_address(*flat_address(index)) == index
    assert flat_address(575) == (0, 15, 35)
    assert flat_address(576) == (1, 0, 0)
    data = np.ones(577, int)
    mapped = split_vector(data)
    assert mapped.shape == (2, 16, 36)
    assert mapped[1, 0, 0] == 1 and np.sum(mapped[1]) == 1
    with pytest.raises(ValueError):
        map_fc([1, 0], [3])


@pytest.mark.parametrize("fanin", [1, 37, 576, 577, 1201])
def test_fc_mapping_preserves_full_dot_product(fanin):
    rng = np.random.default_rng(fanin)
    x, w = rng.integers(0, 2, fanin), rng.integers(-16, 16, fanin)
    c = Cluster(ClusterConfig(mode="integer_reference", theta_count=10**9))
    r = c.step_tiles(map_fc(x, w))
    assert r.diagnostics["total_u"] == int(x @ w)
    assert len(r.diagnostics["folds"]) == (fanin+575)//576


@pytest.mark.parametrize("scu_bits", [3, 4])
def test_small_two_layer_snn_matches_independent_if_oracle(scu_bits):
    rng = np.random.default_rng(50)
    w1, w2 = rng.integers(-2, 5, (4, 8)), rng.integers(-2, 5, (2, 4))
    cfg1 = ClusterConfig(scu_bits=scu_bits, theta_count=6)
    cfg2 = replace(cfg1, theta_count=4)
    layers = [FCLayer(w1, cfg1, layer_id="hidden"), FCLayer(w2, cfg2, layer_id="output")]
    oracle_u = [np.zeros((2, 4), dtype=np.int64), np.zeros((2, 2), dtype=np.int64)]
    for _ in range(6):
        actual = expected = rng.integers(0, 2, (2, 8))
        for i, (weights, layer, theta) in enumerate(zip([w1, w2], layers, [6, 4])):
            oracle_u[i] += expected @ weights.T
            expected = (oracle_u[i] >= theta).astype(int)
            oracle_u[i][expected.astype(bool)] = 0
            actual, details = layer.forward(actual)
            np.testing.assert_array_equal(actual, expected)
            for (batch, out), result in details.items():
                assert reduce_state(result.state, layer.cluster.config)["total_u"] == oracle_u[i][batch, out]
    assert len(layers[0].memory.states) == 8  # batch*logical outputs, not physical lanes


def test_conv_mapping_matches_direct_convolution_and_separate_pixel_state():
    rng = np.random.default_rng(7)
    x, w = rng.integers(0, 2, (2, 4, 5)), rng.integers(-3, 4, (2, 2, 3, 3))
    cfg = ClusterConfig(theta_count=5)
    c, mem = Cluster(cfg), LogicalMemory(cfg)
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    expected_u = {}
    for step in range(2):
        seen = 0
        for (out, y, xpos), tiles in iter_conv_tiles(x, w, padding=1, stride=2):
            lines = []
            for f, (_, wt) in enumerate(tiles):
                line = ("conv", out, f)
                if line not in mem.weights:
                    mem.write_weights(line, wt)
                lines.append(line)
            key = (out, y, xpos)
            context = f"conv/{out}/{y}/{xpos}"
            result = mem.run_context(c, context, [it[0] for it in tiles], lines)
            increment = int(np.sum(padded[:, y*2:y*2+3, xpos*2:xpos*2+3] * w[out]))
            candidate = expected_u.get(key, 0) + increment
            assert result.diagnostics["total_u"] == candidate
            assert result.spike == (candidate >= 5)
            expected_u[key] = 0 if result.spike else candidate
            seen += 1
        assert seen == 12
    assert len(mem.weights) == 2 and len(mem.states) == 12


def test_structured_trace_and_config_round_trip(tmp_path):
    cfg = ClusterConfig()
    path = tmp_path / "config.json"
    cfg.save(path)
    assert ClusterConfig.load(path) == cfg
    c = Cluster(cfg)
    result = c.step(*single(3))
    path = tmp_path / "trace.jsonl"
    write_jsonl(path, [result])
    import json
    saved = json.loads(path.read_text())
    assert saved["candidate"]["scu"][0][0] == 1
    assert saved["diagnostics"]["operation_counts"]["comparisons"] == 1
    assert saved["events"][-1]["name"] == "state_write_done"
    assert first_divergence([result], [result]) is None

def test_full_576_input_extremes_and_required_signed_range():
    x = np.ones((16, 36), dtype=int)
    for weight, expected in [(15, 8640), (-16, -9216)]:
        c = Cluster(ClusterConfig(theta_count=10000))
        r = c.step(x, np.full((16, 36), weight, dtype=int))
        assert r.diagnostics["total_u"] == expected
        assert not r.spike


def test_capacity_error_in_second_fold_is_located_and_transaction_is_atomic():
    cfg = ClusterConfig(mr_bits=1, mode="finite_digital", theta_count=1000)
    c = Cluster(cfg)
    c.state.scu[0, 0], c.state.mr[0, 0] = 14, 1
    before = c.snapshot()
    with pytest.raises(MROverflowError) as exc:
        c.step_tiles([single(1), single(1)])
    assert exc.value.diagnostics["locations"][0]["fold"] == 1
    assert c.logical_step == 0
    np.testing.assert_array_equal(c.state.scu, before["state"].scu)


def test_preview_context_does_not_allocate_or_change_active_context():
    cfg = ClusterConfig(theta_count=100)
    c, memory = Cluster(cfg), LogicalMemory(cfg)
    c.step(*single(2), context="active")
    before = c.snapshot()
    x, w = single(3)
    memory.write_weights("W", w)
    result = memory.run_context(c, "preview", [x], ["W"], commit=False)
    assert result.diagnostics["total_u"] == 3
    assert memory.states == {}
    assert c.logical_step == before["logical_step"]
    assert c.physical_time == before["physical_time"]
    np.testing.assert_array_equal(c.state.scu, before["state"].scu)


def test_reference_keeps_settling_during_comparator_delay():
    c = Cluster(ClusterConfig(mode="nonideal_behavioral", theta_count=4),
                AnalogConfig(reference_time_constant=2, comparator_delay=2))
    r = c.step(*single(), timing=Timing(share_at=1, compare_at=2))
    assert r.analog["v_reference"] == pytest.approx(.25*(1-math.exp(-1)))
    assert c.reference_state == pytest.approx(.25*(1-math.exp(-2)))
    assert c.physical_time == 4


def test_zero_width_requires_no_sign_orientation_delay():
    c = Cluster(ClusterConfig(mode="nonideal_behavioral", theta_count=1), AnalogConfig(sign_delay=1))
    x, w = np.zeros((16, 36), dtype=int), np.zeros((16, 36), dtype=int)
    r = c.step(x, w)
    assert not r.spike and np.all(r.analog["q_cap"] == 0)


def test_charge_helpers_reject_nan_and_infinite_parameters():
    with pytest.raises(ValueError):
        advance_voltage(0, 1, math.nan, 1)
    with pytest.raises(ValueError):
        share_charges([math.nan]*16, [1]*16)
    with pytest.raises(ValueError):
        AnalogConfig(capacitances=[1e308]*16)
