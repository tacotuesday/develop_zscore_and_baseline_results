import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from itertools import permutations
import numpy as np
import pytest

from src.milestone1.p1_m1_roc_auc import roc
from src.milestone1.p1_m1_roc_auc import roc_conceptual
from src.milestone1.p1_m1_roc_auc import auc_roc


test_labels = np.array([1, 0, 0, 1])
test_prediction = np.array([0.3, 0.25, 0.7, 0.0])

test_permutations_data = [
    (test_labels[list(permutation)], test_prediction[list(permutation)])
    for permutation in permutations([0, 1, 2, 3])]


@pytest.mark.parametrize("test_case", test_permutations_data)
def test_roc_conceptual_with_permutations(test_case):

    fpr_test, tpr_test = roc_conceptual(test_case[0], test_case[1], 5)
    fpr_expected, tpr_expected = [0, 0.5, 0.5, 1, 1], [0, 0, 0, 0.5, 1]

    assert fpr_test == pytest.approx(fpr_expected)
    assert tpr_test == pytest.approx(tpr_expected)


@pytest.mark.parametrize("test_case", test_permutations_data)
def test_roc_with_permutations(test_case):

    fpr_test, tpr_test = roc(test_case[0], test_case[1])
    fpr_expected, tpr_expected, _ = roc_curve(test_case[0], test_case[1])

    assert fpr_test == pytest.approx(fpr_expected)
    assert tpr_test == pytest.approx(tpr_expected)


@pytest.mark.parametrize("test_case", test_permutations_data)
def test_roc_auc_with_permutations(test_case):

    roc_auc_test = auc_roc(test_case[0], test_case[1])
    roc_auc_expected = roc_auc_score(test_case[0], test_case[1])

    assert roc_auc_test == pytest.approx(roc_auc_expected)
