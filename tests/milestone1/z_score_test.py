import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


import math
import pytest
import numpy as np
from src.milestone1.p1_m2_z_score import Zscorer


def zscore_training_uniform_test_cases():
    rng = np.random.default_rng(1378981)
    return [rng.uniform(size=100000) for i in range(10)]


@pytest.mark.parametrize("test_case", zscore_training_uniform_test_cases())
def test_zscore_training_uniform(test_case):
    z_scorer = Zscorer(test_case)
    assert z_scorer.mu == pytest.approx(0.5, rel=1e-2)
    assert z_scorer.sigma == pytest.approx(math.sqrt(1.0/12.0), rel=1e-2)


def test_zscore_scoring():
    test_case = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected_scores = np.array([1.5666989, 1.21854359, 0.87038828, 0.52223297,
                                0.17407766, 0.17407766, 0.52223297, 0.87038828,
                                1.21854359,  1.5666989])

    z_scorer = Zscorer(test_case)
    assert z_scorer.score(test_case) == pytest.approx(expected_scores)
