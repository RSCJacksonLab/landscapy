"""Regression tests for immutable, validated sequence objects."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_landscape.core.sequence import (
    BaseNumpySequence,
    BinarySequence,
    MultialleleSequence,
    SoftSequence,
)


def test_sequence_storage_is_owned_and_public_views_are_read_only():
    source = np.array([0, 1, 0])
    sequence = BinarySequence(source)
    source[0] = 1

    assert sequence.to_array().tolist() == [0, 1, 0]
    for public_array in (sequence.sequence, sequence.ndarray):
        with pytest.raises(ValueError, match="read-only"):
            public_array[0] = 1
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            public_array.setflags(write=True)


@pytest.mark.parametrize(
    "values",
    (
        [0.9, 1.0],
        [0.0, 1.9],
        [np.nan, 0.0],
        [np.inf, 1.0],
        [0, 2],
        [],
    ),
)
def test_binary_sequence_validates_values_before_casting(values):
    with pytest.raises(ValueError, match="0/1|empty"):
        BinarySequence(values)


def test_binary_mutation_preserves_subclass_and_explicit_identifier():
    sequence = BinarySequence.from_bits([0, 1], sequence_id="sample-a")

    mutated = sequence.mutate(0, values=[1])

    assert type(mutated) is BinarySequence
    assert mutated.id == "sample-a"
    assert mutated.to_array().tolist() == [1, 1]
    assert sequence.to_array().tolist() == [0, 1]


def test_derived_identifier_is_regenerated_after_mutation():
    sequence = BinarySequence([0, 1])

    mutated = sequence.mutate(0, values=[1])

    assert mutated.id != sequence.id
    assert "1" in mutated.id


def test_multiallele_mutation_preserves_subclass_alphabet_and_identifier():
    sequence = MultialleleSequence.from_string(
        "AB",
        alphabet=["A", "B"],
        sequence_id="multi-a",
    )

    mutated = sequence.mutate(0, values=["B"])

    assert type(mutated) is MultialleleSequence
    assert mutated.alphabet == ["A", "B"]
    assert mutated.id == "multi-a"
    assert mutated.to_str() == "BB"


@pytest.mark.parametrize(
    "posterior, message",
    (
        (np.array([0.5, 0.5]), "2D"),
        (np.ones((1, 2, 1)), "2D"),
        (np.ones((2, 3)) / 3, "width"),
        (np.array([[np.nan, np.nan]]), "finite"),
        (np.array([[np.inf, 0.0]]), "finite"),
        (np.array([[1.1, -0.1]]), "non-negative"),
        (np.array([[0.2, 0.2]]), "sum to one"),
        (np.empty((0, 2)), "at least one site"),
    ),
)
def test_soft_sequence_rejects_invalid_posteriors(posterior, message):
    with pytest.raises(ValueError, match=message):
        SoftSequence(posterior, alphabet=["A", "B"])


@pytest.mark.parametrize(
    "gap_posterior, message",
    (
        (np.array([0.2, 0.3]), "shape"),
        (np.array([[0.2]]), "shape"),
        (np.array([[np.nan], [0.2]]), "finite"),
        (np.array([[-0.1], [0.2]]), "between zero and one"),
        (np.array([[1.1], [0.2]]), "between zero and one"),
        (np.array([[0.2, 0.7], [0.3, 0.7]]), "sum to one"),
    ),
)
def test_soft_sequence_rejects_invalid_gap_posteriors(gap_posterior, message):
    amino_acids = np.array([[0.25, 0.75], [0.6, 0.4]])
    with pytest.raises(ValueError, match=message):
        SoftSequence(
            amino_acids,
            alphabet=["A", "B"],
            gap_posterior=gap_posterior,
        )


def test_soft_sequence_owns_posterior_and_exposes_read_only_view():
    source = np.array([[0.25, 0.75], [0.6, 0.4]])
    sequence = SoftSequence(source, alphabet=["A", "B"])
    source[0] = [1.0, 0.0]

    assert sequence.posterior[0].tolist() == [0.25, 0.75]
    assert np.all(sequence.entropy() >= 0.0)
    with pytest.raises(ValueError, match="read-only"):
        sequence.posterior[0, 0] = 1.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        sequence.posterior.setflags(write=True)


def test_soft_sequence_accepts_normalized_gap_and_non_gap_columns():
    sequence = SoftSequence(
        np.array([[0.25, 0.75], [0.6, 0.4]]),
        alphabet=["A", "B"],
        gap_posterior=np.array([[0.2, 0.8], [0.0, 1.0]]),
    )

    assert sequence.alphabet == ["A", "B", "gap"]
    assert np.allclose(sequence.posterior.sum(axis=1), 1.0)
    assert sequence.posterior[:, -1].tolist() == [0.2, 0.0]


def test_soft_mutation_preserves_subclass_posterior_and_identifier():
    sequence = SoftSequence(
        np.array([[0.8, 0.2], [0.3, 0.7]]),
        alphabet=["A", "B"],
        sequence_id="soft-a",
    )

    mutated = sequence.mutate(0, values=["B"])

    assert type(mutated) is SoftSequence
    assert mutated.id == "soft-a"
    assert mutated.to_str() == "BB"
    assert mutated.posterior.tolist() == [[0.0, 1.0], [0.3, 0.7]]


def test_sequence_factories_preserve_identifiers():
    binary = BinarySequence.from_integer_bits(3, length=3, sequence_id="binary")
    multiallele = MultialleleSequence.random(
        3,
        alphabet=["A", "B"],
        seed=1,
        sequence_id="multi",
    )
    soft = SoftSequence.from_posteriors(
        np.array([[0.5, 0.5]]),
        alphabet=["A", "B"],
        sequence_id="soft",
    )

    assert binary.id == "binary"
    assert multiallele.id == "multi"
    assert soft.id == "soft"


def test_from_cogent3_honors_an_explicit_identifier():
    from cogent3 import get_moltype

    cogent_sequence = get_moltype("protein").make_seq(seq="AC", name="source")

    sequence = BaseNumpySequence.from_cogent3(
        cogent_sequence,
        sequence_id="override",
    )

    assert sequence.id == "override"


@pytest.mark.parametrize(
    "factory",
    (
        lambda: BaseNumpySequence.random(0, alphabet=["A", "B"]),
        lambda: BinarySequence.random(-1),
        lambda: BinarySequence.from_integer_bits(1, length=0),
        lambda: MultialleleSequence.random(0, alphabet=["A", "B"]),
        lambda: BaseNumpySequence.from_integer([], alphabet=["A", "B"]),
    ),
)
def test_factories_reject_empty_or_non_positive_sequences(factory):
    with pytest.raises(ValueError, match="positive|empty"):
        factory()


@pytest.mark.parametrize("alphabet", ([], ["A", "A"]))
def test_multiallele_and_soft_alphabets_are_nonempty_and_unique(alphabet):
    with pytest.raises(ValueError, match="alphabet"):
        MultialleleSequence(["A"], alphabet=alphabet)
    with pytest.raises(ValueError, match="alphabet"):
        SoftSequence(np.ones((1, max(1, len(alphabet)))), alphabet=alphabet)
