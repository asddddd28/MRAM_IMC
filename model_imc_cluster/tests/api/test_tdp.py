import numpy as np
import pytest
from cluster_model import eigen_train, signed_eigen_train

def test_eigen_train_period_and_count():
    train = eigen_train(13, bits=5, step=0.25)
    assert train.duration == 8.0
    assert train.pulse_count == 13
    assert np.array_equal(np.flatnonzero(train.samples), [0, 4, 8, 12, 16, 20, 24, 28])
    assert int(train.samples.sum()) == 13

def test_signed_train_selects_tdp_branch():
    train = signed_eigen_train(-5)
    assert train.polarity == -1 and train.pulse_count == 5
    assert set(train.samples.tolist()) <= {-2, -1, 0}

def test_tdp_input_validation():
    with pytest.raises(ValueError): eigen_train(32)
    with pytest.raises(ValueError): signed_eigen_train(-17)
    with pytest.raises(ValueError): eigen_train(1, step=0)


