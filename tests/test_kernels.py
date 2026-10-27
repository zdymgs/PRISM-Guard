import numpy as np
import pytest

from prism.core.activity import TransitionModel as FirstOrderTransition
from prism.core.activity import _decode
from prism.core.alignment import align_probabilities
from prism.core.sequence import TransitionModel, decode_unique_beam
from prism.eval.metrics import metrics, paired


@pytest.mark.parametrize("length,classes", [(1, 2), (3, 4), (4, 4)])
def test_unique_beam_has_a_feasible_nonrepeating_path(length, classes):
    probability = np.random.default_rng(42).dirichlet(np.ones(classes), size=length)
    model = TransitionModel(
        np.full(classes, -np.log(classes)),
        np.full(classes, -np.log(classes)),
        np.full((classes, classes), -np.log(classes)),
        np.full((classes, classes, classes), -np.log(classes)),
    )

    prediction = decode_unique_beam(np.log(probability), model, .45, 50)

    assert prediction.shape == (length,)
    assert len(np.unique(prediction)) == length
    assert np.all((prediction >= 0) & (prediction < classes))


def test_first_order_zero_transition_weight_is_row_argmax():
    probability = np.array([[.1, .8, .1], [.7, .1, .2], [.1, .2, .7], [.2, .6, .2]])
    model = FirstOrderTransition(
        np.log([.8, .1, .1]),
        np.log([.2, .7, .1]),
        np.log([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]),
    )

    np.testing.assert_array_equal(_decode(probability, model, 0.), probability.argmax(1))


def test_probability_alignment_matches_are_strictly_monotone():
    first = np.eye(3)
    second = first[[0, 0, 1, 2]]

    matches, similarity = align_probabilities(first, second, .2)

    assert matches
    left, right = np.asarray(matches).T
    assert np.all(np.diff(left) > 0)
    assert np.all(np.diff(right) > 0)
    assert np.all((left >= 0) & (left < len(first)))
    assert np.all((right >= 0) & (right < len(second)))
    assert similarity == pytest.approx(1.)


def test_metrics_and_paired_match_a_hand_computed_example():
    labels = np.array([0, 0, 1, 1])
    reference = np.array([0, 0, 0, 1])
    prediction = np.array([0, 1, 1, 1])

    score = metrics(labels, prediction, 2)
    comparison = paired(labels, reference, prediction)

    assert score["n"] == 4
    assert score["correct"] == 3
    assert score["accuracy"] == pytest.approx(3 / 4)
    assert score["macro_f1_fixed2"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert comparison == {
        "rescues": 1, "harms": 1, "net": 0, "changed": 2, "mcnemar_exact_p": 1.,
    }
