import numpy as np
import pytest
from scipy.stats import entropy

from fitness_landscape.core.fitness import (
    ProbabilisticCategoricalFitness,
    EntropyFitnessModifier,
    NumericFitness,
    ProbabilitySliceFitnessModifier,
    GaussianNoiseFitnessModifier,
    GaussianDistributionFitnessModifier,
    ResampleFitnessModifier,
    ArithmeticFitnessModifier,
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


def test_gaussian_noise_modifier_adds_noise_to_replicates():
    reps = [[1.0, 2.0], [3.0, 4.0]]
    layer = NumericFitness.from_replicates("fit", reps)

    modifier = GaussianNoiseFitnessModifier(scale=0.1, seed=123)
    out = modifier(layer)

    rng = np.random.default_rng(123)
    expected = []
    for r in reps:
        arr = np.asarray(r, dtype=float)
        noise = rng.normal(loc=0.0, scale=0.1, size=len(arr))
        expected.append((arr + noise).tolist())

    np.testing.assert_allclose(out.get_tensor().numpy(), np.array(expected))
    assert out.metadata["modifier"] == "gaussian_noise"
    assert out.metadata["source_layer"] == "fit"


def test_gaussian_distribution_modifier_creates_replicates():
    layer = NumericFitness.from_scalars("fit", [1.0, 2.0])
    modifier = GaussianDistributionFitnessModifier(scale=0.0, reps=3)
    out = modifier(layer)

    tensor = out.get_tensor().numpy()
    assert tensor.shape == (2, 3)
    np.testing.assert_allclose(tensor, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert out.metadata["modifier"] == "gaussian_distribution"
    assert out.metadata["reps"] == 3


def test_resample_modifier_uses_replicate_distribution():
    reps = [[1.0, 3.0], [2.0]]
    layer = NumericFitness.from_replicates("fit", reps)

    modifier = ResampleFitnessModifier(reps=2, seed=42)
    out = modifier(layer)

    rng = np.random.default_rng(42)
    expected0 = rng.normal(loc=np.mean(reps[0]), scale=np.std(reps[0]), size=2)
    expected1 = rng.normal(loc=reps[1][0], scale=0.0, size=2)
    expected = np.vstack([expected0, expected1])

    np.testing.assert_allclose(out.get_tensor().numpy(), expected)
    assert out.metadata["modifier"] == "resample"
    assert out.metadata["reps"] == 2


def test_arithmetic_modifier_supports_builtin_and_callable():
    base = NumericFitness.from_scalars("a", [1.0, 2.0])
    other = NumericFitness.from_scalars("b", [10.0, 20.0])

    modifier = ArithmeticFitnessModifier(other, op="add")
    out = modifier(base)

    np.testing.assert_allclose(out.to_scalar(), np.array([11.0, 22.0]))
    assert out.name == "a_arithmetic"
    assert out.metadata["operation"] == "add"
    assert out.metadata["other_layers"] == ["b"]

    modifier2 = ArithmeticFitnessModifier(other, op=lambda x, y: x * y + 1.0)
    out2 = modifier2(base, name="custom")
    np.testing.assert_allclose(out2.to_scalar(), np.array([11.0, 41.0]))
