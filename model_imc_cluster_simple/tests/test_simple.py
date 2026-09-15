"""Simple model smoke/equivalence tests."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
FULL=ROOT/'model_imc_cluster'
sys.path.insert(0,str(FULL)); sys.path.insert(0,str(ROOT))
from cluster_model import Cluster, ClusterConfig
from model_imc_cluster_simple import SimpleCluster, SimpleConfig, eigen_train

def test_simple_matches_full_random():
    rng=np.random.default_rng(20260911); x=rng.integers(0,2,(30,16,36)); w=rng.integers(-16,16,(16,36))
    full=Cluster(ClusterConfig(mode='integer_reference',backend='scalar',scu_bits=4,theta_count=560))
    simple=SimpleCluster(SimpleConfig(scu_bits=4,theta=560))
    for sample in x:
        a=full.step(sample,w); b=simple.step(sample,w)
        assert a.spike==b.spike
        assert np.array_equal(a.candidate.scu,b.candidate.scu); assert np.array_equal(a.candidate.mr,b.candidate.mr)
        assert a.diagnostics['total_u']==b.trace['total_u']

def test_tdp_waveform_has_fixed_slots_and_events():
    wave=eigen_train(13,bits=5,step=.5)
    assert wave.samples.size==32 and wave.pulse_count==13 and wave.duration==16
    assert len(wave.events)==13


def test_simple_matches_full_3bit_and_preview_is_nonmutating():
    rng=np.random.default_rng(7); x=rng.integers(0,2,(16,36)); w=rng.integers(-3,4,(16,36))
    full=Cluster(ClusterConfig(mode='integer_reference',backend='scalar',scu_bits=3,theta_count=100))
    simple=SimpleCluster(SimpleConfig(scu_bits=3,theta=100))
    preview=simple.step(x,w,commit=False)
    assert simple.logical_step == 0 and np.all(simple.state.scu == 0)
    actual=simple.step(x,w)
    ref=full.step(x,w)
    assert actual.spike == ref.spike
    assert np.array_equal(actual.candidate.scu,ref.candidate.scu)
    assert np.array_equal(actual.candidate.mr,ref.candidate.mr)

def test_simple_multifold_accumulates_before_one_decision():
    x=np.zeros((16,36),dtype=int); x[0,0]=1
    w=np.zeros((16,36),dtype=int); w[0,0]=3
    simple=SimpleCluster(SimpleConfig(theta=7))
    result=simple.step_tiles([(x,w),(x,w),(x,w)], commit=False)
    assert len(result.trace['folds']) == 3
    assert result.trace['total_u'] == 9
    assert result.spike is True
