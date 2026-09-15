"""Reference.md T01-T13 acceptance cases, with exact numerical expectations."""
from dataclasses import replace
import math
import numpy as np
import pytest
from cluster_model import (
    AnalogConfig, Cluster, ClusterConfig, ClusterState, DeviceInstance, EventSchedule,
    LogicalMemory, ModelError, MROverflowError, Timing, accumulate_tile, advance_voltage,
    convert, decode_weight, encode_weight, encode_weights, first_divergence, pmac,
    reduce_state, share_charges,
)


def tile(value=0, macro=0):
    x, w = np.zeros((16, 36), dtype=int), np.zeros((16, 36), dtype=int)
    x[macro, 0], w[macro, 0] = 1, value
    return x, w


def zero_tile():
    return np.zeros((16, 36), dtype=int), np.zeros((16, 36), dtype=int)


def config(**kwargs):
    return ClusterConfig(**kwargs)


@pytest.mark.parametrize("w", range(-16, 16))
def test_t01_all_int5_values_and_gated_counts(w):
    bits = encode_weight(w)
    assert decode_weight(bits) == w
    x, weights = tile(w)
    assert sum(pmac(x, encode_weights(weights))[0] * np.array([1, 2, 4, 8, -16])) == w
    x[:] = 0
    assert not np.any(pmac(x, encode_weights(weights)))


@pytest.mark.parametrize("backend", ["scalar", "numpy"])
@pytest.mark.parametrize("s,a,h_next,total_before,total_after,carry_expected", [
    (3, 7, 7, 23, 59, 5), (4, 15, 5, 47, 83, 3),
])
def test_t02_multi_carry_and_msb(backend, s, a, h_next, total_before, total_after, carry_expected):
    cfg = config(backend=backend, scu_bits=s)
    state = ClusterState.zeros()
    counts = np.zeros((16, 5), dtype=int)
    for b in (0, 4):
        state.scu[0, b], state.mr[0, b], counts[0, b] = a, 2, 36
    new, carry, errors = accumulate_tile(state, counts, cfg)
    assert not errors
    for b in (0, 4):
        assert (new.scu[0, b], new.mr[0, b], carry[0, b]) == (3, h_next, carry_expected)
        assert state.scu[0, b] + cfg.L*state.mr[0, b] == total_before
        assert new.scu[0, b] + cfg.L*new.mr[0, b] == total_after
    if s == 4:
        assert (-16*total_before, -16*total_after, -16*(total_after-total_before)) == (-752, -1328, -576)
    with pytest.raises(MROverflowError):
        accumulate_tile(state, counts, replace(cfg, mr_bits=2))


@pytest.mark.parametrize("a,sign", [([3, 2, 0, 1, 1], -1), ([5, 2, 0, 1, 1], 1)])
def test_t03_signed_conversion(a, sign):
    cluster = Cluster(config(theta_count=100))
    cluster.state.scu[0] = a
    result = cluster.step(*zero_tile())
    assert result.diagnostics["local_s"][0] == sign
    assert result.analog["tw"][0] == 1
    assert result.analog["sign"][0] == sign
    assert result.analog["q_cap"][0] == sign
    assert result.analog["v_signal"] == sign / 16


@pytest.mark.parametrize("theta,fire,reference", [(560, True, -13), (561, False, -12.9375)])
def test_t04_mr_correction_and_negative_reference(theta, fire, reference):
    cluster = Cluster(config(theta_count=theta))
    cluster.state.scu[:] = [3, 0, 0, 0, 1]
    cluster.state.mr[:] = [1, 1, 0, 0, 0]
    result = cluster.step(*zero_tile())
    d, a = result.diagnostics, result.analog
    assert (d["total_s"], d["total_g"], d["total_u"]) == (-208, 48, 560)
    assert np.all(d["local_u"] == 35)
    assert a["v_signal"] == -13 and a["v_reference"] == reference
    assert a["theta_eff"] == theta - 768
    assert result.spike == fire
    if fire:
        assert not np.any(result.state.scu) and not np.any(result.state.mr)
        assert np.any(result.candidate.scu) and np.any(result.candidate.mr)


@pytest.mark.parametrize("policy", ["error", "wide_reference", "wrap", "saturate"])
def test_t05_128_steps_zero_net_input_still_overflows(policy):
    c = Cluster(config(mr_bits=4, overflow_policy=policy, theta_count=1))
    x, w = zero_tile()
    x[0, :2], w[0, :2] = 1, [1, -1]
    for _ in range(127):
        r = c.step(x, w)
        assert not r.spike and r.diagnostics["total_u"] == 0
    assert c.state.scu[0].tolist() == [14, 15, 15, 15, 15]
    assert c.state.mr[0].tolist() == [15, 7, 7, 7, 7]
    saved = c.snapshot()
    if policy == "error":
        with pytest.raises(MROverflowError) as exc:
            c.step(x, w)
        diagnostic = exc.value.diagnostics
        assert diagnostic["locations"][0]["bit"] == 0
        assert diagnostic["candidate_mr"][0] == [16, 8, 8, 8, 8]
        assert diagnostic["candidate_scu"][0] == [0]*5
        assert np.array_equal(c.state.mr, saved["state"].mr)
        assert c.logical_step == 127
        assert c.last_error["code"] == "mr_overflow"
    else:
        r = c.step(x, w)
        assert r.diagnostics["overflow"][0]["value"] == 16
        assert r.diagnostics["total_u"] == {"wide_reference": 0, "wrap": -256, "saturate": -16}[policy]


def test_t06_cross_macro_cancellation_keeps_local_history():
    c = Cluster(config(theta_count=1))
    x, w = zero_tile()
    x[:2, 0], w[:2, 0] = 1, [10, -10]
    for n in range(1, 41):
        r = c.step(x, w)
        assert r.diagnostics["local_u"][:2].tolist() == [10*n, -10*n]
        assert r.diagnostics["total_u"] == 0 and not r.spike
    assert np.any(c.state.scu) and np.any(c.state.mr)


def test_t07_false_fire_resets_actual_state_and_zero_input_does_not_reintegrate():
    ref = Cluster(config(theta_count=10, mode="integer_reference"))
    hw = Cluster(config(theta_count=10, mode="nonideal_behavioral"))
    rr, hr = ref.step(*tile(3)), hw.step(*tile(3), margin_injection=8/16)
    assert not rr.spike and hr.spike
    rr, hr = ref.step(*zero_tile()), hw.step(*zero_tile())
    assert rr.diagnostics["total_u"] == 3 and hr.diagnostics["total_u"] == 0
    assert not hr.spike


def test_t07_single_transient_causes_recursive_divergence():
    ref = Cluster(config(theta_count=10, mode="integer_reference"))
    hw = Cluster(config(theta_count=10, mode="nonideal_behavioral"))
    refs, actual = [], []
    for step in range(7):
        refs.append(ref.step(*tile(3)))
        actual.append(hw.step(*tile(3), margin_injection=2/16 if step == 2 else 0))
    assert "".join(str(int(r.spike)) for r in refs) == "0001000"
    assert "".join(str(int(r.spike)) for r in actual) == "0010001"
    assert [r.diagnostics["total_u"] for r in actual] == [3, 6, 9, 3, 6, 9, 12]
    assert first_divergence(refs, actual)["step"] == 2


def test_t08_fold_transaction_no_early_fire_and_order_independent_wide_state():
    cfg = config(theta_count=10, overflow_policy="wide_reference", mode="integer_reference")
    a, b = Cluster(cfg), Cluster(cfg)
    r1 = a.step_tiles([tile(12), tile(-5)])
    r2 = b.step_tiles([tile(-5), tile(12)])
    assert r1.diagnostics["total_u"] == 7 and not r1.spike
    assert np.array_equal(r1.candidate.scu, r2.candidate.scu)
    assert np.array_equal(r1.candidate.mr, r2.candidate.mr)
    assert len(r1.diagnostics["folds"]) == 2
    assert r1.diagnostics["operation_counts"]["comparisons"] == 1
    assert sum(e.name == "compare_done" for e in r1.events) == 1
    assert a.logical_step == 1


def test_t08_overflow_in_fold_is_not_hidden_by_final_fire():
    cfg = config(mr_bits=1, theta_count=1, mode="finite_digital")
    c = Cluster(cfg)
    c.state.scu[0, 0], c.state.mr[0, 0] = 15, 1
    before = c.snapshot()
    with pytest.raises(MROverflowError) as exc:
        c.step_tiles([tile(1), tile(-1)])
    assert exc.value.diagnostics["locations"][0]["fold"] == 0
    assert np.array_equal(c.state.scu, before["state"].scu)
    assert c.logical_step == 0


def test_t09_shared_weight_line_separate_contexts_and_copy_ownership():
    cfg = config(theta_count=10)
    cluster, mem = Cluster(cfg), LogicalMemory(cfg)
    x, weights = tile(3)
    mem.write_weights(("conv", 0, 0), weights)
    def step(context):
        return mem.run_context(cluster, context, [x], [("conv", 0, 0)])
    step("pixel-A")
    step("pixel-B")
    step("pixel-B")
    assert step("pixel-A").diagnostics["total_u"] == 6
    assert step("pixel-A").diagnostics["total_u"] == 9
    assert step("pixel-A").spike
    assert reduce_state(mem.read_state("pixel-A").state, cfg)["total_u"] == 0
    assert reduce_state(mem.read_state("pixel-B").state, cfg)["total_u"] == 6
    copy = mem.read_state("pixel-B")
    copy.state.scu[:] = 0
    assert reduce_state(mem.read_state("pixel-B").state, cfg)["total_u"] == 6
    assert np.array_equal(mem.read_weights(("conv", 0, 0)), weights)


def test_t10_common_edge_shift_cancels_and_equal_nonzero_edges():
    state = ClusterState.zeros()
    state.scu[0] = [0, 0, 0, 2, 1]  # P=N=16, not merely all zero
    state.scu[1] = [3, 2, 0, 1, 1]
    cfg = AnalogConfig()
    device = DeviceInstance.create(cfg)
    a = convert(state.scu, 100, cfg, device)
    b = convert(state.scu, 100, replace(cfg, common_delay=100), device)
    assert a["tw"][0] == a["q_cap"][0] == 0
    np.testing.assert_array_equal(a["tw"], b["tw"])
    np.testing.assert_array_equal(a["q_cap"], b["q_cap"])
    np.testing.assert_array_equal(a["t_plus"]+100, b["t_plus"])
    assert b["done_at"]-a["done_at"] == 100
    mismatch = replace(cfg, plane_gains=(1., 1., 1., 1., 1.1))
    altered = convert(state.scu, 100, mismatch, device)
    assert altered["tw"][0] == pytest.approx(1.6)
    assert altered["q_cap"][0] == pytest.approx(-1.6)


def test_t10_equal_P_different_plane_composition_can_differ():
    state = ClusterState.zeros()
    state.scu[0, 0], state.scu[1, 1] = 2, 1
    cfg = AnalogConfig()
    device = DeviceInstance.create(cfg)
    a = convert(state.scu, 100, cfg, device)
    assert a["t_plus"][0] == a["t_plus"][1] == 2
    b = convert(state.scu, 100, replace(cfg, plane_gains=(1., 1.25, 1., 1., 1.)), device)
    assert b["t_plus"][0] == 2 and b["t_plus"][1] == 2.5


def test_t11_zero_contribution_caps_parasitic_and_finite_sharing():
    q = np.zeros(16)
    q[0] = 1
    assert share_charges(q, [1]*16) == 1/16
    assert share_charges(q, [1]*16, parasitic=4) == 1/20
    # Changing the source cap alone does NOT rescale its charge gain.
    assert share_charges(q, [2]+[1]*15) == 1/17
    assert share_charges(q, [1]*16, wait=2, time_constant=2) == pytest.approx((1-math.exp(-1))/16)
    assert share_charges(q, [1]*16, wait=0, time_constant=2) == 0


def test_t11_charge_and_hold_leakage_follows_rc_solution():
    cfg = AnalogConfig(hold_resistance=10.)
    state = ClusterState.zeros()
    state.scu[0, 0], state.scu[1, 0] = 1, 2
    result = convert(state.scu, 100, cfg, DeviceInstance.create(cfg))
    assert result["hold"][0] == 1 and result["hold"][1] == 0
    expected = 10 * (1-math.exp(-.1)) * math.exp(-.1)
    assert result["q_cap"][0] == pytest.approx(expected)
    assert result["q_cap"][1] == pytest.approx(10*(1-math.exp(-.2)))
    assert advance_voltage(2, 3, 4, 5) == pytest.approx(2+12/5)
    assert advance_voltage(2, 0, 4, 5, 2) == pytest.approx(2*math.exp(-.4))


@pytest.mark.parametrize("params,code", [
    ({"local_voltage_limit": .5}, "charge_voltage_out_of_range"),
    ({"reference_code_min": 0, "reference_code_max": 2}, "reference_out_of_range"),
    ({"reference_voltage_min": -.1, "reference_voltage_max": .1}, "reference_out_of_range"),
    ({"sign_delay": .1}, "sign_unstable"),
    ({"comparator_delay": 2., "comparator_timeout": 1.}, "comparator_timeout"),
    ({"comparator_input_limit": .01}, "comparator_input_out_of_range"),
    ({"sign_deadzone": 2.}, "ambiguous_sign"),
])
def test_t12_locatable_analog_failures_do_not_commit(params, code):
    c = Cluster(config(mode="nonideal_behavioral", theta_count=10), AnalogConfig(**params))
    saved = c.snapshot()
    with pytest.raises(ModelError) as exc:
        c.step(*tile(1), context="test-context")
    assert exc.value.code == code
    assert c.last_error["context"] == "test-context"
    assert c.last_error["logical_step"] == 0
    assert c.last_error["candidate_scu"][0][0] == 1
    assert not np.any(c.state.scu) and c.logical_step == 0
    assert c.reference_state == saved["reference_state"]


def test_t12_supplied_timing_not_repaired_and_resource_conflicts():
    c = Cluster(config(mode="nonideal_behavioral", theta_count=10))
    with pytest.raises(ModelError, match="charge_not_ready"):
        c.step(*tile(3), timing=Timing(share_at=1))
    with pytest.raises(ModelError, match="sharing_not_ready"):
        c.step(*tile(3), timing=Timing(share_at=3, compare_at=2))
    schedule = EventSchedule()
    rd = schedule.schedule("weight_read", 0, 3, resource="bank0")
    with pytest.raises(ModelError, match="resource_conflict"):
        schedule.schedule("weight_read", 2, 3, resource="bank0")
    with pytest.raises(ModelError, match="dependency_not_ready"):
        schedule.schedule("pmac", 2, dependencies=(rd,))
    schedule.schedule("pmac", 3, dependencies=(rd,))


def test_t12_repeatable_device_event_noise_and_serialized_device(tmp_path):
    ac = AnalogConfig(base_delay=5, static_gain_sigma=.02, common_jitter_sigma=.1,
                      branch_jitter_sigma=.03, comparator_noise_sigma=.001, chip_seed=11, event_seed=31)
    cfg = config(mode="nonideal_behavioral", theta_count=50)
    device = DeviceInstance.create(ac)
    path = tmp_path / "device.json"
    device.save(path)
    loaded = DeviceInstance.load(path)
    np.testing.assert_array_equal(loaded.plane_gains, device.plane_gains)
    c1, c2 = Cluster(cfg, ac, device), Cluster(cfg, ac, loaded)
    fixed = device.plane_gains.copy()
    for n in range(4):
        r1, r2 = c1.step(*tile(3)), c2.step(*tile(3))
        assert r1.spike == r2.spike
        for key in ("t_plus", "t_minus", "tw", "sign", "q_cap", "margin"):
            np.testing.assert_array_equal(r1.analog[key], r2.analog[key])
        np.testing.assert_array_equal(device.plane_gains, fixed)
    other = DeviceInstance.create(replace(ac, chip_seed=12))
    assert not np.array_equal(other.plane_gains, fixed)
    # Static parameters change only with chip sampling, not event seed.
    np.testing.assert_array_equal(DeviceInstance.create(replace(ac, event_seed=99)).plane_gains, fixed)
    # Evaluating another event in between cannot perturb a repeated event.
    a = convert(c1.state.scu, 50, ac, device, context="A", step=7)
    convert(c1.state.scu, 50, ac, device, context="B", step=99)
    b = convert(c1.state.scu, 50, ac, device, context="A", step=7)
    np.testing.assert_array_equal(a["t_plus"], b["t_plus"])


@pytest.mark.parametrize("scu_bits", [3, 4])
@pytest.mark.parametrize("mode", ["integer_reference", "ideal_behavioral"])
def test_t13_scalar_numpy_all_states_carries_and_analog_chain(scu_bits, mode):
    rng = np.random.default_rng(123)
    cfg = config(scu_bits=scu_bits, mode=mode, theta_count=150, mr_bits=12)
    a, b = Cluster(replace(cfg, backend="scalar")), Cluster(replace(cfg, backend="numpy"))
    full_u = 0
    for _ in range(8):
        tiles = [(rng.integers(0, 2, (16, 36)), rng.integers(-16, 16, (16, 36))) for _ in range(2)]
        full_u += sum(int(np.sum(x*w)) for x, w in tiles)
        ra, rb = a.step_tiles(tiles), b.step_tiles(tiles)
        assert ra.diagnostics["total_u"] == rb.diagnostics["total_u"] == full_u
        assert ra.spike == rb.spike == (full_u >= cfg.theta_count)
        for state in ("candidate", "state"):
            for key in ("scu", "mr"):
                np.testing.assert_array_equal(getattr(getattr(ra, state), key), getattr(getattr(rb, state), key))
        for fa, fb in zip(ra.diagnostics["folds"], rb.diagnostics["folds"]):
            for key in ("carry", "counts", "candidate_scu", "candidate_mr"):
                np.testing.assert_array_equal(fa[key], fb[key])
        if mode == "ideal_behavioral":
            # Tolerances are normalized time/charge/voltage units; exact decisions.
            for key in ("t_plus", "t_minus", "tw", "sign", "q_cap", "v_signal", "v_reference", "margin"):
                np.testing.assert_allclose(ra.analog[key], rb.analog[key], rtol=0, atol=1e-12)
        if ra.spike:
            full_u = 0
