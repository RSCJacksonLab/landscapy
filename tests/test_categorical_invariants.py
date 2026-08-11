"""Regression tests for categorical and probability-domain invariants."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_landscape.core.fitness import (
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
    make_fitness_layer,
)
from fitness_landscape.core.sequence import BaseNumpySequence


@pytest.mark.parametrize("categories", ([], ["A", "A"]))
def test_categorical_layers_require_nonempty_unique_categories(categories):
    with pytest.raises(ValueError, match="categories"):
        CategoricalFitness("categorical", ["A"], categories=categories)
    with pytest.raises(ValueError, match="categories"):
        ProbabilisticCategoricalFitness(
            "probabilistic",
            np.ones((1, max(1, len(categories)))),
            categories,
        )


def test_categorical_layer_owns_values_and_categories():
    values = ["A", "B"]
    categories = ["A", "B"]
    layer = CategoricalFitness("categorical", values, categories)

    values[0] = "B"
    categories[0] = "changed"
    returned_categories = layer.categories
    returned_categories[0] = "changed-again"

    assert layer.get_value(0) == "A"
    assert layer.categories == ["A", "B"]


@pytest.mark.parametrize(
    "probabilities, message",
    (
        (np.array([0.5, 0.5]), "2-D"),
        (np.ones((1, 2, 1)), "2-D"),
        (np.ones((1, 3)) / 3, "Shape"),
        (np.array([[np.nan, np.nan]]), "finite"),
        (np.array([[np.inf, 0.0]]), "finite"),
        (np.array([[1.2, -0.2]]), "non-negative"),
        (np.array([[2.0, -1.0]]), "non-negative"),
        (np.array([[0.0, 0.0]]), "sum to 1"),
        (np.array([[0.2, 0.2]]), "sum to 1"),
    ),
)
def test_probability_constructor_rejects_invalid_domains(probabilities, message):
    with pytest.raises(ValueError, match=message):
        ProbabilisticCategoricalFitness(
            "probabilistic",
            probabilities,
            ["A", "B"],
        )


def test_probability_factory_does_not_normalize_invalid_probability_mass():
    with pytest.raises(ValueError, match="sum to 1"):
        ProbabilisticCategoricalFitness.from_probabilities(
            "probabilistic",
            np.array([[2.0, 1.0]]),
            categories=["A", "B"],
        )


def test_generic_factory_routes_nested_categorical_data_to_validation():
    layer = make_fitness_layer(
        "probabilistic",
        [[0.25, 0.75]],
        dtype="categorical",
        categories=["A", "B"],
    )

    assert isinstance(layer, ProbabilisticCategoricalFitness)
    assert layer.probabilities.tolist() == [[0.25, 0.75]]

    with pytest.raises(ValueError, match="non-negative"):
        make_fitness_layer(
            "probabilistic",
            [[1.2, -0.2]],
            dtype="categorical",
            categories=["A", "B"],
        )


def test_probability_layer_owns_matrix_and_exposes_read_only_view():
    source = np.array([[0.25, 0.75], [0.6, 0.4]])
    categories = ["A", "B"]
    layer = ProbabilisticCategoricalFitness("probabilistic", source, categories)
    source[0] = [1.0, 0.0]
    categories[0] = "changed"

    assert layer.probabilities[0].tolist() == [0.25, 0.75]
    assert layer.categories == ["A", "B"]
    with pytest.raises(ValueError, match="read-only"):
        layer.probabilities[0, 0] = 1.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        layer.probabilities.setflags(write=True)


@pytest.mark.parametrize(
    "counts, alpha, message",
    (
        (np.array([1.0, 2.0]), 0.0, "2-D"),
        (np.ones((1, 3)), 0.0, "width"),
        (np.array([[np.nan, 1.0]]), 0.0, "finite"),
        (np.array([[np.inf, 1.0]]), 0.0, "finite"),
        (np.array([[-1.0, 2.0]]), 0.0, "non-negative"),
        (np.array([[0.0, 0.0]]), 0.0, "positive mass"),
        (np.array([[1.0, 2.0]]), -0.1, "alpha"),
        (np.array([[1.0, 2.0]]), np.nan, "alpha"),
        (np.array([[1.0, 2.0]]), np.inf, "alpha"),
    ),
)
def test_count_factory_rejects_invalid_domains(counts, alpha, message):
    with pytest.raises(ValueError, match=message):
        ProbabilisticCategoricalFitness.from_counts(
            "counts",
            counts,
            categories=["A", "B"],
            alpha=alpha,
        )


def test_positive_smoothing_turns_zero_count_rows_into_uniform_probabilities():
    layer = ProbabilisticCategoricalFitness.from_counts(
        "counts",
        np.array([[0.0, 0.0]]),
        categories=["A", "B"],
        alpha=1.0,
    )

    assert layer.probabilities.tolist() == [[0.5, 0.5]]


@pytest.mark.parametrize(
    "logits, message",
    (
        (np.array([0.0, 1.0]), "2-D"),
        (np.ones((1, 3)), "width"),
        (np.array([[np.nan, 0.0]]), "finite"),
        (np.array([[np.inf, 0.0]]), "finite"),
    ),
)
def test_logits_factory_requires_finite_aligned_matrices(logits, message):
    with pytest.raises(ValueError, match=message):
        ProbabilisticCategoricalFitness.from_logits(
            "logits",
            logits,
            categories=["A", "B"],
        )


@pytest.mark.parametrize(
    "one_hot, message",
    (
        (np.array([1.0, 0.0]), "2-D"),
        (np.ones((1, 3)), "width"),
        (np.array([[np.nan, 0.0]]), "finite"),
        (np.array([[0.6, 0.4]]), "zero and one"),
        (np.array([[1.0, 1.0]]), "exactly one"),
        (np.array([[0.0, 0.0]]), "exactly one"),
        (np.array([[-1.0, 2.0]]), "zero and one"),
    ),
)
def test_categorical_from_one_hot_is_strict(one_hot, message):
    with pytest.raises(ValueError, match=message):
        CategoricalFitness.from_one_hot(
            "categorical",
            one_hot,
            categories=["A", "B"],
        )


@pytest.mark.parametrize(
    "one_hot, message",
    (
        (np.array([1.0, 0.0]), "2D"),
        (np.ones((1, 3)), "width"),
        (np.array([[np.nan, 0.0]]), "finite"),
        (np.array([[0.6, 0.4]]), "zero and one"),
        (np.array([[1.0, 1.0]]), "exactly one"),
        (np.array([[0.0, 0.0]]), "exactly one"),
    ),
)
def test_sequence_from_one_hot_is_strict(one_hot, message):
    with pytest.raises(ValueError, match=message):
        BaseNumpySequence.from_one_hot(one_hot, alphabet=["A", "B"])


def test_valid_boolean_one_hot_inputs_remain_supported():
    one_hot = np.array([[True, False], [False, True]])

    categorical = CategoricalFitness.from_one_hot(
        "categorical",
        one_hot,
        categories=["A", "B"],
    )
    sequence = BaseNumpySequence.from_one_hot(one_hot, alphabet=["A", "B"])

    assert categorical.get_tensor().numpy().tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert sequence.to_str() == "AB"


@pytest.mark.parametrize(
    "probabilities",
    ([0.5], [0.5, 0.6], [-0.1, 1.1], [np.nan, np.nan]),
)
def test_categorical_random_validates_sampling_probabilities(probabilities):
    with pytest.raises(ValueError, match="p"):
        CategoricalFitness.random(
            "random",
            length=2,
            categories=["A", "B"],
            p=probabilities,
        )


def test_empty_sample_rows_require_explicit_smoothing_via_counts():
    with pytest.raises(ValueError, match="positive mass"):
        ProbabilisticCategoricalFitness.from_samples(
            "samples",
            [[]],
            categories=["A", "B"],
        )
