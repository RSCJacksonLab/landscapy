"""Epistasis decompositions with explicit sequence-design contracts."""

from itertools import combinations
from typing import Dict, List, Literal, Tuple

import numpy as np

from .._optional import require_optional
from ..core.landscape import FitnessLandscape
from ..transforms.walsh_hadamard import walsh_coefficients


sklearn_linear = require_optional(
    "sklearn.linear_model",
    extra="analysis",
    purpose="regression-based epistasis analysis",
)
LinearRegression = sklearn_linear.LinearRegression
Lasso = sklearn_linear.Lasso
Ridge = sklearn_linear.Ridge
ElasticNet = sklearn_linear.ElasticNet


def calculate_epistasis_walsh(
    landscape: FitnessLandscape,
    order: int,
    **kwargs,
) -> Dict:
    """Decompose a complete binary fitness cube with Walsh contrasts.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape on every one of the ``2**L`` genotypes encoded with states
        zero and one. The graph itself need not be a Hamming hypercube.
    order : int
        Highest interaction order to return, between one and ``L``.
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.

    Returns
    -------
    dict
        Uniform-measure Fourier-Walsh coefficients, the corresponding
        orthonormal transform coefficients, coefficients grouped by order,
        variance fractions, domain metadata, and summary statistics.

    Raises
    ------
    ValueError
        If the sequences are not a complete, duplicate-free binary cube, the
        active fitness signal is not finite, or ``order`` is invalid.

    Notes
    -----
    :func:`~fitness_landscape.transforms.walsh_hadamard.walsh_coefficients`
    uses the orthonormal ``2**(-L/2)`` transform. This analysis function
    reports the uniform-measure Fourier-Walsh ``2**(-L)`` coefficients,
    so each reported coefficient is its orthonormal counterpart divided by
    ``sqrt(2**L)``. With ``0 -> +1`` and ``1 -> -1`` coding, those values
    equal the unregularized regression coefficients on a complete cube.
    """
    del kwargs
    sequence_matrix, _ = _validated_landscape_data(landscape, order)
    binary_matrix = _binary_matrix(sequence_matrix, method="Walsh epistasis")
    _require_complete_binary_cube(binary_matrix, method="Walsh epistasis")

    orthonormal_coeffs = walsh_coefficients(landscape, order=order)
    scale = float(np.sqrt(len(binary_matrix)))
    coeffs = {term: float(value / scale) for term, value in orthonormal_coeffs.items()}
    by_order = _group_binary_coefficients_by_order(coeffs)

    squared_coeffs_by_order: Dict[int, float] = {}
    for term, value in coeffs.items():
        if term == "intercept":
            continue
        term_order = len(term.split(","))
        squared_coeffs_by_order[term_order] = (
            squared_coeffs_by_order.get(term_order, 0.0) + value**2
        )
    total_variance = float(sum(squared_coeffs_by_order.values()))
    if total_variance > 0:
        variation_explained = {
            term_order: squared_sum / total_variance
            for term_order, squared_sum in squared_coeffs_by_order.items()
        }
    else:
        variation_explained = {
            term_order: 0.0 for term_order in squared_coeffs_by_order
        }

    return {
        "coefficients": coeffs,
        "orthonormal_coefficients": orthonormal_coeffs,
        "by_order": by_order,
        "variance_explained": variation_explained,
        "domain": {
            "sequence_design": "full_binary_cube",
            "n_positions": int(binary_matrix.shape[1]),
            "n_genotypes": int(binary_matrix.shape[0]),
            "complete": True,
        },
        "normalization": {
            "coefficients": "uniform_measure_fourier_walsh_2^-L",
            "orthonormal_coefficients": "orthonormal_2^(-L/2)",
            "orthonormal_to_reported_scale": float(1.0 / scale),
            "binary_coding": "0 -> +1; 1 -> -1",
        },
        "statistics": _calculate_epistasis_statistics(coeffs),
    }


def get_epistasis_matrix(landscape: FitnessLandscape) -> np.ndarray:
    """Compute pairwise Fourier-Walsh epistatic variance.

    Parameters
    ----------
    landscape : FitnessLandscape
        Complete binary fitness cube to analyze.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix whose off-diagonal values are squared second-order
        uniform-measure Fourier-Walsh coefficients.
    """
    results = calculate_epistasis_walsh(landscape, order=2)
    n = results["domain"]["n_positions"]
    epistasis_matrix = np.zeros((n, n), dtype=float)

    for term, value in results["by_order"].get(2, {}).items():
        i_str, j_str = term.split(",")
        i, j = int(i_str), int(j_str)
        epistasis_matrix[i, j] = value**2
        epistasis_matrix[j, i] = value**2
    return epistasis_matrix


def calculate_epistasis_regression(
    landscape: FitnessLandscape,
    order: int,
    regularization: Literal["l1", "l2", "elastic_net"] | None = None,
    alpha: float = 1.0,
    **kwargs,
) -> Dict:
    """Estimate epistasis on a sampled binary design by effect coding.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape with equal-length sequences encoded only with states zero
        and one. The genotype design may be incomplete.
    order : int
        Highest interaction order included in the design matrix, between one
        and the sequence length.
    regularization : {None, 'l1', 'l2', 'elastic_net'}, optional
        Regression penalty. Unregularized least squares is accepted only when
        the intercept-plus-contrast design has full column rank. Penalized
        estimators may be used for rank-deficient designs, but their returned
        coefficients are penalty-selected rather than identified by the data
        alone.
    alpha : float, default=1.0
        Strictly positive penalty strength for regularized models.
    **kwargs
        Additional model settings. ``l1_ratio`` is consumed by elastic net;
        other keys are ignored.

    Returns
    -------
    dict
        Coefficients grouped by interaction order, model and design-rank
        metadata, domain and normalization metadata, summary statistics, and
        the in-sample coefficient of determination ``r2_score``.

    Raises
    ------
    ValueError
        If the sequence domain, fitness signal, order, regularization, or
        unregularized design identifiability is invalid.

    Notes
    -----
    Binary states are coded as ``0 -> +1`` and ``1 -> -1``. Interaction
    columns are products of these main-effect columns. The intercept is fitted
    jointly with all effects; it equals the empirical mean only for a balanced
    design such as a complete binary cube.
    """
    sequence_matrix, fitness_values = _validated_landscape_data(landscape, order)
    binary_matrix = _binary_matrix(sequence_matrix, method="Regression epistasis")
    design, feature_names, index_by_order = _build_effect_design(
        binary_matrix, order
    )

    augmented_design = np.column_stack(
        [np.ones(binary_matrix.shape[0], dtype=float), design]
    )
    design_rank = int(np.linalg.matrix_rank(augmented_design))
    n_parameters = int(augmented_design.shape[1])
    full_rank = design_rank == n_parameters

    if regularization is None:
        if not full_rank:
            raise ValueError(
                "Unregularized epistasis regression is not identifiable: "
                f"the intercept-plus-contrast design rank is {design_rank} for "
                f"{n_parameters} parameters. Reduce order, add observations, or "
                "choose regularization='l2', 'l1', or 'elastic_net'."
            )
        model = LinearRegression()
    else:
        if regularization not in {"l1", "l2", "elastic_net"}:
            raise ValueError(f"Unsupported regularization: {regularization}")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and greater than zero")
        if regularization == "l1":
            model = Lasso(alpha=alpha)
        elif regularization == "l2":
            model = Ridge(alpha=alpha)
        else:
            l1_ratio = float(kwargs.get("l1_ratio", 0.5))
            if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
                raise ValueError("l1_ratio must be finite and between zero and one")
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    model.fit(design, fitness_values)
    fitted_intercept = float(model.intercept_)
    coeffs = {"intercept": fitted_intercept}
    coeffs.update(
        {name: float(model.coef_[index]) for index, name in enumerate(feature_names)}
    )

    by_order: Dict[int, Dict[str, float]] = {0: {"intercept": fitted_intercept}}
    for term_order, indices in index_by_order.items():
        by_order[term_order] = {
            feature_names[index]: float(model.coef_[index]) for index in indices
        }

    predictions = np.asarray(model.predict(design), dtype=float)
    fitness_mean = float(np.mean(fitness_values))
    ss_res = float(np.sum((fitness_values - predictions) ** 2))
    ss_tot = float(np.sum((fitness_values - fitness_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    complete_cube = _is_complete_binary_cube(binary_matrix)

    return {
        "coefficients": coeffs,
        "by_order": by_order,
        "model": {
            "r2_score": r2,
            "model_type": model.__class__.__name__,
            "regularization": regularization,
            "alpha": float(alpha) if regularization is not None else None,
            "n_observations": int(binary_matrix.shape[0]),
            "n_parameters": n_parameters,
            "design_rank": design_rank,
            "unregularized_coefficients_identifiable": full_rank,
            "coefficient_solution": (
                "data_identified"
                if regularization is None and full_rank
                else "penalty_selected"
            ),
        },
        "domain": {
            "sequence_design": "sampled_binary_design",
            "n_positions": int(binary_matrix.shape[1]),
            "n_genotypes": int(binary_matrix.shape[0]),
            "complete_binary_cube": complete_cube,
        },
        "normalization": {
            "coefficients": "binary_effect_coding",
            "binary_coding": "0 -> +1; 1 -> -1",
            "full_cube_walsh_equivalence": "uniform_measure_fourier_walsh_2^-L",
        },
        "statistics": _calculate_epistasis_statistics(coeffs),
    }


def _build_effect_design(
    sequences,
    order: int,
) -> Tuple[np.ndarray, List[str], Dict[int, List[int]]]:
    """Build binary effect-coded interaction columns through ``order``."""
    binary_matrix = np.asarray(sequences, dtype=float)
    if binary_matrix.ndim != 2:
        raise ValueError("Binary effect design requires a two-dimensional matrix")
    n_observations, n_positions = binary_matrix.shape
    z_matrix = 1.0 - 2.0 * binary_matrix

    columns: List[np.ndarray] = []
    names: List[str] = []
    index_by_order: Dict[int, List[int]] = {}
    next_column = 0
    for term_order in range(1, order + 1):
        start = next_column
        for position_indices in combinations(range(n_positions), term_order):
            columns.append(
                np.prod(z_matrix[:, position_indices], axis=1, dtype=float)[:, None]
            )
            names.append("*".join(f"pos{position}" for position in position_indices))
            next_column += 1
        index_by_order[term_order] = list(range(start, next_column))

    design = (
        np.hstack(columns)
        if columns
        else np.zeros((n_observations, 0), dtype=float)
    )
    return design, names, index_by_order


def calculate_epistasis_ensemble(
    landscape: FitnessLandscape,
    order: int,
    **kwargs,
) -> Dict:
    """Compute empirical background-averaged categorical contrasts.

    Parameters
    ----------
    landscape : FitnessLandscape
        General categorical or multi-allelic landscape. Equal-length complete,
        incomplete, balanced, and unbalanced observed genotype designs are
        supported.
    order : int
        Highest interaction order to return, between one and the sequence
        length.
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.

    Returns
    -------
    dict
        Empirical marginal Möbius coefficients grouped by interaction order,
        design and decomposition metadata, and summary statistics.

    Raises
    ------
    ValueError
        If sequences have different lengths, the active fitness signal is not
        finite, or ``order`` is invalid.

    Notes
    -----
    For each observed allele cell ``a_S`` at a position subset ``S``, the
    marginal mean is the equally weighted mean fitness of observations matching
    that cell. Its coefficient is the marginal mean minus coefficients for
    *every* proper subset of ``S``. This observed-support Möbius inversion is a
    balanced functional-ANOVA decomposition on a complete balanced factorial
    design. On incomplete or unbalanced designs it remains an exact hierarchy
    of empirical marginal cell means, but it is not an orthogonal ANOVA and
    does not impute or extrapolate unobserved genotype cells.
    """
    del kwargs
    return _calculate_empirical_mobius(landscape, order, method="ensemble")


def calculate_epistasis_reference_free(
    landscape: FitnessLandscape,
    order: int,
    **kwargs,
) -> Dict:
    """Compute reference-free empirical categorical contrasts.

    Parameters
    ----------
    landscape : FitnessLandscape
        General categorical or multi-allelic landscape. Equal-length complete,
        incomplete, balanced, and unbalanced observed genotype designs are
        supported.
    order : int
        Highest interaction order to return, between one and the sequence
        length.
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.

    Returns
    -------
    dict
        Empirical marginal Möbius coefficients grouped by interaction order,
        design and decomposition metadata, and summary statistics.

    Raises
    ------
    ValueError
        If sequences have different lengths, the active fitness signal is not
        finite, or ``order`` is invalid.

    Notes
    -----
    This reference-free API uses the same observed-support empirical marginal
    Möbius estimand as :func:`calculate_epistasis_ensemble`. No allele is chosen
    as a reference. Missing genotype cells are omitted without imputation, and
    unbalanced observations receive equal empirical weight. Consequently, an
    incomplete or unbalanced result must not be interpreted as an orthogonal
    population ANOVA.
    """
    del kwargs
    return _calculate_empirical_mobius(landscape, order, method="reference_free")


def _calculate_empirical_mobius(
    landscape: FitnessLandscape,
    order: int,
    *,
    method: str,
) -> Dict:
    sequence_matrix, fitness_values = _validated_landscape_data(landscape, order)
    n_observations, n_positions = sequence_matrix.shape
    coefficients: Dict[str, float] = {}
    by_order: Dict[int, Dict[str, float]] = {}

    intercept = float(np.mean(fitness_values))
    coefficients["intercept"] = intercept
    by_order[0] = {"intercept": intercept}

    for term_order in range(1, order + 1):
        by_order[term_order] = {}
        for position_combo in combinations(range(n_positions), term_order):
            grouped_fitness: Dict[Tuple, List[float]] = {}
            for row, fitness in zip(sequence_matrix, fitness_values):
                allele_combo = tuple(row[position] for position in position_combo)
                grouped_fitness.setdefault(allele_combo, []).append(float(fitness))

            for allele_combo, values in grouped_fitness.items():
                coefficient = float(np.mean(values) - intercept)
                for lower_order in range(1, term_order):
                    for local_indices in combinations(range(term_order), lower_order):
                        lower_positions = tuple(position_combo[i] for i in local_indices)
                        lower_alleles = tuple(allele_combo[i] for i in local_indices)
                        lower_term = _categorical_term(lower_positions, lower_alleles)
                        coefficient -= coefficients[lower_term]

                term = _categorical_term(position_combo, allele_combo)
                coefficients[term] = coefficient
                by_order[term_order][term] = coefficient

    design = _categorical_design_metadata(sequence_matrix)
    return {
        "coefficients": coefficients,
        "by_order": by_order,
        "domain": {
            "sequence_design": "observed_general_categorical_design",
            "n_positions": int(n_positions),
            "n_observations": int(n_observations),
            **design,
        },
        "decomposition": {
            "method": method,
            "estimand": "empirical_marginal_mobius",
            "observation_weighting": "equal",
            "orthogonal_anova": bool(
                design["complete_factorial"] and design["balanced_genotype_counts"]
            ),
            "missing_genotype_cells": "omitted_without_imputation_or_extrapolation",
        },
        "statistics": _calculate_epistasis_statistics(coefficients),
    }


def _validated_landscape_data(
    landscape: FitnessLandscape,
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = list(landscape.sequences)
    if not sequences:
        raise ValueError("Epistasis analysis requires at least one sequence")

    arrays = [np.asarray(sequence.to_array()).reshape(-1) for sequence in sequences]
    n_positions = len(arrays[0])
    if n_positions == 0:
        raise ValueError("Epistasis analysis requires non-empty sequences")
    if any(len(array) != n_positions for array in arrays):
        raise ValueError("Epistasis analysis requires equal-length sequences")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be an integer")
    if order < 1 or order > n_positions:
        raise ValueError(
            f"order must be between 1 and the sequence length ({n_positions})"
        )

    sequence_matrix = np.empty((len(arrays), n_positions), dtype=object)
    for row_index, array in enumerate(arrays):
        sequence_matrix[row_index] = [
            value.item() if isinstance(value, np.generic) else value for value in array
        ]

    fitness_values = np.asarray(landscape.get_signal(), dtype=float)
    if fitness_values.shape != (len(sequences),):
        raise ValueError("Epistasis analysis requires one scalar fitness per sequence")
    if not np.all(np.isfinite(fitness_values)):
        raise ValueError("Epistasis analysis requires finite fitness values")
    return sequence_matrix, fitness_values


def _binary_matrix(sequence_matrix: np.ndarray, *, method: str) -> np.ndarray:
    try:
        binary_matrix = np.asarray(sequence_matrix.tolist(), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{method} requires sequences encoded only with binary states 0 and 1"
        ) from error
    if not np.all(np.isfinite(binary_matrix)) or not np.all(
        (binary_matrix == 0.0) | (binary_matrix == 1.0)
    ):
        raise ValueError(
            f"{method} requires sequences encoded only with binary states 0 and 1"
        )
    return binary_matrix.astype(np.int8)


def _is_complete_binary_cube(binary_matrix: np.ndarray) -> bool:
    n_observations, n_positions = binary_matrix.shape
    genotypes = {tuple(int(value) for value in row) for row in binary_matrix}
    return n_observations == 2**n_positions and len(genotypes) == n_observations


def _require_complete_binary_cube(binary_matrix: np.ndarray, *, method: str) -> None:
    if not _is_complete_binary_cube(binary_matrix):
        n_observations, n_positions = binary_matrix.shape
        n_unique = len({tuple(int(value) for value in row) for row in binary_matrix})
        raise ValueError(
            f"{method} requires a complete, duplicate-free binary cube with "
            f"{2**n_positions} genotypes for {n_positions} positions; found "
            f"{n_observations} observations and {n_unique} unique genotypes"
        )


def _group_binary_coefficients_by_order(
    coefficients: Dict[str, float],
) -> Dict[int, Dict[str, float]]:
    by_order: Dict[int, Dict[str, float]] = {}
    for term, value in coefficients.items():
        term_order = 0 if term == "intercept" else len(term.split(","))
        by_order.setdefault(term_order, {})[term] = value
    return by_order


def _categorical_term(positions: Tuple[int, ...], alleles: Tuple) -> str:
    return ",".join(
        f"{position}:{allele}" for position, allele in zip(positions, alleles)
    )


def _categorical_design_metadata(sequence_matrix: np.ndarray) -> Dict:
    levels_by_position = []
    for position in range(sequence_matrix.shape[1]):
        levels = list(dict.fromkeys(sequence_matrix[:, position].tolist()))
        levels_by_position.append(levels)

    possible_genotypes = int(np.prod([len(levels) for levels in levels_by_position]))
    genotype_counts: Dict[Tuple, int] = {}
    for row in sequence_matrix:
        genotype = tuple(row.tolist())
        genotype_counts[genotype] = genotype_counts.get(genotype, 0) + 1
    complete = len(genotype_counts) == possible_genotypes
    balanced = complete and len(set(genotype_counts.values())) == 1
    return {
        "levels_by_position": levels_by_position,
        "n_observed_genotype_cells": int(len(genotype_counts)),
        "n_possible_genotype_cells": possible_genotypes,
        "complete_factorial": complete,
        "balanced_genotype_counts": balanced,
    }


def _calculate_epistasis_statistics(coefficients: Dict) -> Dict:
    """Calculate scalar summaries of non-intercept coefficients."""
    coefficient_values = [
        value for term, value in coefficients.items() if term != "intercept"
    ]
    if not coefficient_values:
        return {"mean": 0, "std": 0, "max": 0, "min": 0, "abs_mean": 0}
    return {
        "mean": float(np.mean(coefficient_values)),
        "std": float(np.std(coefficient_values)),
        "max": float(np.max(coefficient_values)),
        "min": float(np.min(coefficient_values)),
        "abs_mean": float(np.mean(np.abs(coefficient_values))),
    }
