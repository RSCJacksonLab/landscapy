import numpy as np
import pytest
from scipy.stats import entropy

from fitness_landscape.core.fitness import (
    ProbabilisticCategoricalFitness,
    EntropyFitnessModifier,
    NumericFitness,
    ProbabilitySliceFitnessModifier,
)


def test_entropy_modifier_computes_entropy():
    probs = np.array(
        [
            [0.5, 0.5],
            [1.0, 0.0],
            [0.25, 0.75],
        ],
        dtype=float,
    )
    layer = ProbabilisticCategoricalFitness.from_probabilities(
        name="prob",
        probabilities=probs,
        categories=["a", "b"],
    )

    modifier = EntropyFitnessModifier()
    out = modifier(layer)

    expected = entropy(probs, axis=1)
    assert isinstance(out, NumericFitness)
    assert out.name == "prob_entropy"
    assert np.allclose(out.to_scalar(), expected)
    assert out.metadata["modifier"] == "entropy"
    assert out.metadata["source_layer"] == "prob"
    assert out.metadata["input_categories"] == ["a", "b"]


def test_probability_slice_modifier_by_label():
    probs = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.7, 0.2],
        ],
        dtype=float,
    )
    layer = ProbabilisticCategoricalFitness.from_probabilities(
        name="prob",
        probabilities=probs,
        categories=["cat", "dog", "fox"],
    )

    modifier = ProbabilitySliceFitnessModifier("dog")
    out = modifier(layer)

    assert isinstance(out, NumericFitness)
    assert out.name == "prob_prob_dog"
    assert np.allclose(out.to_scalar(), probs[:, 1])
    assert out.metadata["modifier"] == "probability_slice"
    assert out.metadata["target_category"] == "dog"
    assert out.metadata["target_index"] == 1


def test_apply_fitness_modifier_attaches_and_names_unique(binary_3bit_landscape, rng):
    landscape = binary_3bit_landscape
    n = len(landscape.sequences)
    probs = rng.random((n, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)

    layer = ProbabilisticCategoricalFitness.from_probabilities(
        name="prob_cat",
        probabilities=probs,
        categories=["x", "y", "z"],
    )
    landscape.attach(layer=layer)

    modifier = EntropyFitnessModifier(base=2.0)
    out = landscape.apply_fitness_modifier(
        modifier,
        source_layer="prob_cat",
        output_name="default",
    )

    expected = entropy(probs, axis=1, base=2.0)
    assert out.name == "default_1"  # collision with existing 'default' layer
    assert out is landscape.fitness_layers[out.name]
    assert np.allclose(out.to_scalar(), expected)
    assert out.metadata["base"] == 2.0


def test_probability_slice_on_landscape(binary_3bit_landscape, rng):
    landscape = binary_3bit_landscape
    n = len(landscape.sequences)
    probs = rng.random((n, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)

    layer = ProbabilisticCategoricalFitness.from_probabilities(
        name="prob_cat",
        probabilities=probs,
        categories=["x", "y", "z"],
    )
    landscape.attach(layer=layer)

    modifier = ProbabilitySliceFitnessModifier(category=1)
    out = landscape.apply_fitness_modifier(
        modifier,
        source_layer="prob_cat",
        output_name="default",
    )

    assert out.name == "default_1"
    assert np.allclose(out.to_scalar(), probs[:, 1])
    assert out.metadata["target_index"] == 1
    assert out.metadata["target_category"] == "y"
