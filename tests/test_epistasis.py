"""Known-answer and domain-contract tests for epistasis decompositions."""

from __future__ import annotations

from itertools import combinations, product

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.analysis.epistasis import (
    calculate_epistasis_ensemble,
    calculate_epistasis_reference_free,
    calculate_epistasis_regression,
    calculate_epistasis_walsh,
    get_epistasis_matrix,
)
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence


def _landscape(genotypes, fitness, *, binary: bool) -> FitnessLandscape:
    rows = [tuple(genotype) for genotype in genotypes]
    if binary:
        sequences = [BinarySequence.from_bits(row) for row in rows]
    else:
        alphabet = list(dict.fromkeys(allele for row in rows for allele in row))
        sequences = [BaseNumpySequence(row, alphabet=alphabet) for row in rows]

    graph = nx.Graph()
    for index, sequence in enumerate(sequences):
        graph.add_node(index, sequence=sequence)
    layer = NumericFitness("default", [[float(value)] for value in fitness])
    return FitnessLandscape(
        sequences=sequences,
        graph=graph,
        fitness_layers={"default": layer},
    )


def _effect_signal(genotypes, coefficients):
    values = []
    for genotype in genotypes:
        coded = 1.0 - 2.0 * np.asarray(genotype, dtype=float)
        value = float(coefficients.get((), 0.0))
        for positions, coefficient in coefficients.items():
            if positions:
                value += coefficient * float(np.prod(coded[list(positions)]))
        values.append(value)
    return values


def _walsh_term(positions):
    return ",".join(str(position) for position in positions)


def _regression_term(positions):
    return "*".join(f"pos{position}" for position in positions)


def _categorical_term(positions, alleles):
    return ",".join(
        f"{position}:{allele}" for position, allele in zip(positions, alleles)
    )


def test_binary_full_cube_recovers_known_coefficients_through_fourth_order():
    genotypes = list(product([0, 1], repeat=4))
    known = {
        (): 1.25,
        (0,): 0.5,
        (3,): -0.25,
        (0, 2): 0.75,
        (1, 2, 3): -0.4,
        (0, 1, 2, 3): 1.1,
    }
    landscape = _landscape(
        genotypes,
        _effect_signal(genotypes, known),
        binary=True,
    )

    walsh = calculate_epistasis_walsh(landscape, order=4)
    regression = calculate_epistasis_regression(landscape, order=4)

    assert walsh["domain"]["sequence_design"] == "full_binary_cube"
    assert regression["model"]["unregularized_coefficients_identifiable"] is True
    assert set(walsh["by_order"]) == {0, 1, 2, 3, 4}
    assert set(regression["by_order"]) == {0, 1, 2, 3, 4}
    assert walsh["coefficients"]["intercept"] == pytest.approx(known[()])
    assert regression["coefficients"]["intercept"] == pytest.approx(known[()])

    for term_order in range(1, 5):
        for positions in combinations(range(4), term_order):
            assert walsh["coefficients"][_walsh_term(positions)] == pytest.approx(
                regression["coefficients"][_regression_term(positions)]
            )

    for positions, expected in known.items():
        if not positions:
            continue
        walsh_term = _walsh_term(positions)
        regression_term = _regression_term(positions)
        assert walsh["coefficients"][walsh_term] == pytest.approx(expected)
        assert regression["coefficients"][regression_term] == pytest.approx(expected)
        assert walsh["orthonormal_coefficients"][walsh_term] == pytest.approx(
            expected * np.sqrt(len(genotypes))
        )

    # Asymmetric first-order effects ensure FWHT masks map back to the correct
    # big-endian biological sequence positions rather than reversed labels.
    assert walsh["coefficients"]["0"] == pytest.approx(0.5)
    assert walsh["coefficients"]["3"] == pytest.approx(-0.25)


def test_walsh_constant_signal_and_pairwise_matrix_have_known_answers():
    genotypes = list(product([0, 1], repeat=3))
    constant = calculate_epistasis_walsh(
        _landscape(genotypes, [3.0] * len(genotypes), binary=True),
        order=3,
    )
    assert set(constant["variance_explained"].values()) == {0.0}

    pairwise = _landscape(
        genotypes,
        _effect_signal(genotypes, {(): 1.0, (0, 2): 0.5}),
        binary=True,
    )
    expected = np.zeros((3, 3), dtype=float)
    expected[0, 2] = expected[2, 0] = 0.25
    assert get_epistasis_matrix(pairwise) == pytest.approx(expected)


@pytest.mark.parametrize(
    "analyzer",
    [calculate_epistasis_ensemble, calculate_epistasis_reference_free],
)
def test_general_decomposition_recovers_binary_fourth_order_cell_effects(analyzer):
    genotypes = list(product([0, 1], repeat=4))
    coefficient = 1.75
    signal = _effect_signal(
        genotypes,
        {(): -0.5, (0, 1, 2, 3): coefficient},
    )
    result = analyzer(_landscape(genotypes, signal, binary=True), order=4)

    for genotype in genotypes:
        term = _categorical_term(range(4), genotype)
        coded_product = float(np.prod(1.0 - 2.0 * np.asarray(genotype)))
        assert result["by_order"][4][term] == pytest.approx(
            coefficient * coded_product
        )
    assert result["decomposition"]["orthogonal_anova"] is True


@pytest.mark.parametrize(
    "analyzer",
    [calculate_epistasis_ensemble, calculate_epistasis_reference_free],
)
def test_multiallelic_complete_design_recovers_known_fourth_order_contrast(analyzer):
    levels = [("A", "B", "C"), ("x", "y"), ("u", "v"), ("m", "n")]
    codes = [
        {"A": -1.0, "B": 0.0, "C": 1.0},
        {"x": -1.0, "y": 1.0},
        {"u": -1.0, "v": 1.0},
        {"m": -1.0, "n": 1.0},
    ]
    genotypes = list(product(*levels))
    coefficient = 2.25
    signal = [
        3.0
        + coefficient
        * float(np.prod([codes[position][allele] for position, allele in enumerate(row)]))
        for row in genotypes
    ]
    result = analyzer(_landscape(genotypes, signal, binary=False), order=4)

    assert result["domain"]["complete_factorial"] is True
    assert result["domain"]["n_possible_genotype_cells"] == 24
    assert result["decomposition"]["orthogonal_anova"] is True
    for genotype in genotypes:
        expected = coefficient * float(
            np.prod([codes[position][allele] for position, allele in enumerate(genotype)])
        )
        term = _categorical_term(range(4), genotype)
        assert result["by_order"][4][term] == pytest.approx(expected)


@pytest.mark.parametrize(
    "analyzer",
    [calculate_epistasis_ensemble, calculate_epistasis_reference_free],
)
def test_incomplete_multiallelic_design_reconstructs_observed_fourth_order_cells(
    analyzer,
):
    levels = [("A", "B", "C"), ("x", "y"), ("u", "v"), ("m", "n")]
    missing_genotype = ("C", "y", "v", "n")
    genotypes = [row for row in product(*levels) if row != missing_genotype]
    signal = [
        0.4 * (index + 1)
        + (1.3 if row[0] == "C" and row[2] == "v" else 0.0)
        for index, row in enumerate(genotypes)
    ]
    result = analyzer(_landscape(genotypes, signal, binary=False), order=4)

    assert result["domain"]["complete_factorial"] is False
    assert result["domain"]["n_observed_genotype_cells"] == 23
    assert result["domain"]["n_possible_genotype_cells"] == 24
    assert result["decomposition"]["orthogonal_anova"] is False
    assert (
        result["decomposition"]["missing_genotype_cells"]
        == "omitted_without_imputation_or_extrapolation"
    )

    for genotype, observed in zip(genotypes, signal):
        reconstructed = result["coefficients"]["intercept"]
        for term_order in range(1, 5):
            for positions in combinations(range(4), term_order):
                alleles = tuple(genotype[position] for position in positions)
                reconstructed += result["coefficients"][
                    _categorical_term(positions, alleles)
                ]
        assert reconstructed == pytest.approx(observed)

    assert _categorical_term(range(4), missing_genotype) not in result["coefficients"]


def test_sampled_binary_regression_recovers_identifiable_incomplete_design():
    all_genotypes = list(product([0, 1], repeat=4))
    genotypes = all_genotypes[:-1]
    known = {(): 0.7, (0,): 0.3, (2,): -0.8, (0, 3): 1.2}
    landscape = _landscape(
        genotypes,
        _effect_signal(genotypes, known),
        binary=True,
    )

    result = calculate_epistasis_regression(landscape, order=2)

    assert result["domain"]["complete_binary_cube"] is False
    assert result["model"]["design_rank"] == result["model"]["n_parameters"]
    assert result["coefficients"]["intercept"] == pytest.approx(known[()])
    assert result["coefficients"]["pos0"] == pytest.approx(known[(0,)])
    assert result["coefficients"]["pos2"] == pytest.approx(known[(2,)])
    assert result["coefficients"]["pos0*pos3"] == pytest.approx(known[(0, 3)])
    assert result["model"]["r2_score"] == pytest.approx(1.0)


def test_rank_deficient_regression_requires_explicit_regularization():
    genotypes = list(product([0, 1], repeat=4))[:-1]
    signal = _effect_signal(genotypes, {(): 1.0, (0, 1, 2, 3): 0.5})
    landscape = _landscape(genotypes, signal, binary=True)

    with pytest.raises(ValueError, match="not identifiable"):
        calculate_epistasis_regression(landscape, order=4)

    result = calculate_epistasis_regression(
        landscape,
        order=4,
        regularization="l2",
        alpha=0.1,
    )
    assert result["model"]["design_rank"] < result["model"]["n_parameters"]
    assert result["model"]["unregularized_coefficients_identifiable"] is False
    assert result["model"]["coefficient_solution"] == "penalty_selected"


def test_regularized_full_rank_regression_labels_penalty_selected_solution():
    genotypes = list(product([0, 1], repeat=2))
    landscape = _landscape(
        genotypes,
        _effect_signal(genotypes, {(): 1.0, (0,): 2.0}),
        binary=True,
    )

    result = calculate_epistasis_regression(
        landscape,
        order=1,
        regularization="l2",
        alpha=1.0,
    )

    assert result["model"]["unregularized_coefficients_identifiable"] is True
    assert result["model"]["coefficient_solution"] == "penalty_selected"
    assert result["coefficients"]["pos0"] != pytest.approx(2.0)


@pytest.mark.parametrize("regularization", ["l1", "elastic_net"])
def test_other_supported_regression_penalties_return_penalty_selected_solutions(
    regularization,
):
    genotypes = list(product([0, 1], repeat=2))
    landscape = _landscape(
        genotypes,
        _effect_signal(genotypes, {(): 1.0, (0,): 2.0}),
        binary=True,
    )

    result = calculate_epistasis_regression(
        landscape,
        order=1,
        regularization=regularization,
        alpha=0.1,
        l1_ratio=0.25,
    )

    assert result["model"]["coefficient_solution"] == "penalty_selected"
    assert result["model"]["regularization"] == regularization


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"regularization": "unknown"}, "Unsupported regularization"),
        ({"regularization": "l2", "alpha": 0.0}, "alpha must be finite"),
        (
            {"regularization": "elastic_net", "alpha": 0.1, "l1_ratio": 1.5},
            "l1_ratio must be finite",
        ),
    ],
)
def test_regression_rejects_invalid_penalty_settings(kwargs, message):
    genotypes = list(product([0, 1], repeat=2))
    landscape = _landscape(genotypes, range(4), binary=True)

    with pytest.raises(ValueError, match=message):
        calculate_epistasis_regression(landscape, order=1, **kwargs)


@pytest.mark.parametrize(
    "analyzer",
    [calculate_epistasis_ensemble, calculate_epistasis_reference_free],
)
def test_unbalanced_categorical_design_is_explicit_and_reconstructs_cell_means(
    analyzer,
):
    genotypes = [
        ("A", "x"),
        ("A", "x"),
        ("A", "y"),
        ("B", "x"),
        ("B", "y"),
    ]
    fitness = [0.0, 2.0, 4.0, 6.0, 8.0]
    with pytest.warns(UserWarning, match="Duplicate sequences detected"):
        landscape = _landscape(genotypes, fitness, binary=False)

    result = analyzer(landscape, order=2)

    assert result["domain"]["complete_factorial"] is True
    assert result["domain"]["balanced_genotype_counts"] is False
    assert result["decomposition"]["observation_weighting"] == "equal"
    assert result["decomposition"]["orthogonal_anova"] is False

    cell_means = {
        ("A", "x"): 1.0,
        ("A", "y"): 4.0,
        ("B", "x"): 6.0,
        ("B", "y"): 8.0,
    }
    for genotype, observed_cell_mean in cell_means.items():
        reconstructed = result["coefficients"]["intercept"]
        reconstructed += result["coefficients"][_categorical_term((0,), genotype[:1])]
        reconstructed += result["coefficients"][_categorical_term((1,), genotype[1:])]
        reconstructed += result["coefficients"][
            _categorical_term((0, 1), genotype)
        ]
        assert reconstructed == pytest.approx(observed_cell_mean)


def test_binary_only_methods_reject_unsupported_sequence_domains():
    incomplete_binary = _landscape(
        [(0, 0), (0, 1), (1, 0)],
        [0.0, 1.0, 2.0],
        binary=True,
    )
    multiallelic = _landscape(
        [("A",), ("B",), ("C",)],
        [0.0, 1.0, 2.0],
        binary=False,
    )

    with pytest.raises(ValueError, match="complete, duplicate-free binary cube"):
        calculate_epistasis_walsh(incomplete_binary, order=2)
    with pytest.raises(ValueError, match="binary states 0 and 1"):
        calculate_epistasis_walsh(multiallelic, order=1)
    with pytest.raises(ValueError, match="binary states 0 and 1"):
        calculate_epistasis_regression(multiallelic, order=1)

    numeric_multiallelic = _landscape(
        [(0,), (1,), (2,)],
        [0.0, 1.0, 2.0],
        binary=False,
    )
    with pytest.raises(ValueError, match="binary states 0 and 1"):
        calculate_epistasis_regression(numeric_multiallelic, order=1)


def test_walsh_rejects_duplicate_binary_cube_rows():
    with pytest.warns(UserWarning, match="Duplicate sequences detected"):
        landscape = _landscape(
            [(0, 0), (0, 1), (1, 0), (1, 0)],
            [0.0, 1.0, 2.0, 3.0],
            binary=True,
        )
    with pytest.raises(ValueError, match="complete, duplicate-free binary cube"):
        calculate_epistasis_walsh(landscape, order=2)


@pytest.mark.parametrize(
    "analyzer",
    [
        calculate_epistasis_walsh,
        calculate_epistasis_regression,
        calculate_epistasis_ensemble,
        calculate_epistasis_reference_free,
    ],
)
def test_epistasis_methods_reject_nonfinite_or_nonscalar_fitness(analyzer):
    landscape = _landscape(
        list(product([0, 1], repeat=2)),
        [0.0, 1.0, 1.0, 2.0],
        binary=True,
    )
    landscape.get_signal = lambda: np.array([0.0, 1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="finite fitness"):
        analyzer(landscape, order=2)

    landscape.get_signal = lambda: np.ones((4, 1), dtype=float)
    with pytest.raises(ValueError, match="one scalar fitness"):
        analyzer(landscape, order=2)


def test_epistasis_validation_rejects_empty_and_unequal_length_landscapes():
    empty = FitnessLandscape(sequences=[], graph=nx.Graph())
    with pytest.raises(ValueError, match="at least one sequence"):
        calculate_epistasis_ensemble(empty, order=1)

    unequal = _landscape([(0, 0), (0, 1)], [0.0, 1.0], binary=True)
    unequal.sequences[1] = BinarySequence.from_bits([0, 1, 1])
    with pytest.raises(ValueError, match="equal-length sequences"):
        calculate_epistasis_regression(unequal, order=1)


@pytest.mark.parametrize(
    "analyzer",
    [
        calculate_epistasis_walsh,
        calculate_epistasis_regression,
        calculate_epistasis_ensemble,
        calculate_epistasis_reference_free,
    ],
)
@pytest.mark.parametrize("order", [0, 3])
def test_epistasis_methods_reject_orders_outside_sequence_length(analyzer, order):
    landscape = _landscape(
        list(product([0, 1], repeat=2)),
        [0.0, 1.0, 1.0, 2.0],
        binary=True,
    )
    with pytest.raises(ValueError, match="order must be between"):
        analyzer(landscape, order=order)
