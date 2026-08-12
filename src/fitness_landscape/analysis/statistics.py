import copy
import warnings
from itertools import combinations

import numpy as np
import scipy.stats as stats
from typing import List, Optional, Tuple, Dict, Any, Callable, Mapping, Sequence

from ..core.landscape import FitnessLandscape
from .._optional import ray_runtime, require_optional

sklearn_linear = require_optional(
    "sklearn.linear_model",
    extra="analysis",
    purpose="regression-based landscape statistics",
)
sklearn_selection = require_optional(
    "sklearn.model_selection",
    extra="analysis",
    purpose="regression-based landscape statistics",
)
sklearn_metrics = require_optional(
    "sklearn.metrics",
    extra="analysis",
    purpose="regression-based landscape statistics",
)
LinearRegression = sklearn_linear.LinearRegression
Ridge = sklearn_linear.Ridge
Lasso = sklearn_linear.Lasso
ElasticNet = sklearn_linear.ElasticNet
train_test_split = sklearn_selection.train_test_split
cross_val_score = sklearn_selection.cross_val_score
mean_squared_error = sklearn_metrics.mean_squared_error
r2_score = sklearn_metrics.r2_score
from ..utils import sample_observed_induced_connected


_SUPPORTED_TESTS = {"ttest", "mannwhitney", "ks"}
_SUPPORTED_ALTERNATIVES = {"two-sided", "greater", "less"}
_SUPPORTED_CORRECTIONS = {"bonferroni", "holm", "fdr_bh"}
_SHAPIRO_MIN_N = 3
_SHAPIRO_MAX_N = 5000


def _validate_alpha(alpha: float) -> float:
    """Return a validated significance threshold."""
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise TypeError("alpha must be a real number strictly between 0 and 1.")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "alpha must be a real number strictly between 0 and 1."
        ) from error
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1.")
    return value


def _validate_nan_policy(nan_policy: str) -> str:
    if not isinstance(nan_policy, str):
        raise TypeError("nan_policy must be either 'raise' or 'omit'.")
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be either 'raise' or 'omit'.")
    return nan_policy


def _clean_numeric_sample(
    values: Any,
    *,
    name: str,
    nan_policy: str,
) -> tuple[np.ndarray, int]:
    """Coerce a sample and apply the requested missing-value policy."""
    try:
        sample = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numeric values.") from error

    if np.any(np.isinf(sample)):
        raise ValueError(f"{name} must contain only finite values; infinity found.")

    missing = int(np.count_nonzero(np.isnan(sample)))
    if missing and nan_policy == "raise":
        raise ValueError(
            f"{name} contains {missing} NaN value(s); use nan_policy='omit' "
            "to remove them explicitly."
        )
    if missing:
        sample = sample[~np.isnan(sample)]

    if sample.size == 0:
        raise ValueError(f"{name} must contain at least one finite value.")
    return sample, missing


def _normality_result(
    values: np.ndarray,
    *,
    alpha: float,
) -> Dict[str, Any]:
    """Apply the documented Shapiro-Wilk sample-size policy."""
    sample_size = int(values.size)
    base: Dict[str, Any] = {
        "test": "shapiro_wilk",
        "alpha": alpha,
        "valid_sample_size": [_SHAPIRO_MIN_N, _SHAPIRO_MAX_N],
    }

    if sample_size < _SHAPIRO_MIN_N:
        return {
            **base,
            "shapiro_statistic": np.nan,
            "shapiro_p_value": np.nan,
            "is_normal": None,
            "status": "not_run",
            "reason": "Shapiro-Wilk requires at least 3 observations.",
        }

    if sample_size > _SHAPIRO_MAX_N:
        warnings.warn(
            "Shapiro-Wilk was not run because its p-value is not calibrated "
            "for samples larger than 5000 observations.",
            UserWarning,
            stacklevel=3,
        )
        return {
            **base,
            "shapiro_statistic": np.nan,
            "shapiro_p_value": np.nan,
            "is_normal": None,
            "status": "not_run",
            "reason": "Sample size exceeds the supported maximum of 5000.",
        }

    if np.ptp(values) == 0.0:
        return {
            **base,
            "shapiro_statistic": np.nan,
            "shapiro_p_value": np.nan,
            "is_normal": None,
            "status": "not_run",
            "reason": "Shapiro-Wilk is undefined for a constant sample.",
        }

    shapiro_test = stats.shapiro(values)
    statistic = float(shapiro_test.statistic)
    p_value = float(shapiro_test.pvalue)
    return {
        **base,
        "shapiro_statistic": statistic,
        "shapiro_p_value": p_value,
        "is_normal": bool(p_value >= alpha),
        "status": "performed",
        "reason": None,
    }


def analyze_fitness_distribution(
    landscape: FitnessLandscape,
    *,
    alpha: float = 0.05,
    nan_policy: str = "raise",
) -> Dict[str, Any]:
    """
    Analyze the distribution of fitness values in a landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    alpha : float, default=0.05
        Significance threshold used to interpret the Shapiro-Wilk p-value.
        It must be finite and strictly between zero and one.
    nan_policy : {"raise", "omit"}, default="raise"
        Missing-value policy. ``"raise"`` rejects NaNs, while ``"omit"``
        removes them and records how many were omitted. Infinite values are
        always rejected.
        
    Returns
    -------
    results : dict
        Distribution summaries and a ``normality_test`` record. Shapiro-Wilk
        is not run for fewer than 3 observations, more than 5000 observations,
        or constant samples; these cases return ``status="not_run"`` and a
        reason instead of an invalid p-value. Empty samples raise
        ``ValueError``.
    """
    alpha = _validate_alpha(alpha)
    nan_policy = _validate_nan_policy(nan_policy)

    raw_values = [landscape.get_fitness(seq) for seq in landscape.sequences]
    try:
        raw_array = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("Fitness distribution analysis requires numeric values.") from error
    if raw_array.ndim != 1:
        raise ValueError("Fitness distribution analysis requires scalar values.")
    fitness_values, omitted_count = _clean_numeric_sample(
        raw_array,
        name="fitness values",
        nan_policy=nan_policy,
    )
    
    # Calculate basic statistics
    mean = np.mean(fitness_values)
    median = np.median(fitness_values)
    std = np.std(fitness_values)
    min_val = np.min(fitness_values)
    max_val = np.max(fitness_values)
    range_val = max_val - min_val
    
    # Calculate percentiles
    percentiles = np.percentile(fitness_values, [25, 50, 75, 90, 95, 99])
    
    # Shape moments are not estimable for very small or constant samples.
    is_constant = np.ptp(fitness_values) == 0.0
    skewness = (
        float(stats.skew(fitness_values))
        if fitness_values.size >= 3 and not is_constant
        else np.nan
    )
    kurtosis = (
        float(stats.kurtosis(fitness_values))
        if fitness_values.size >= 4 and not is_constant
        else np.nan
    )

    normality_test = _normality_result(fitness_values, alpha=alpha)
    
    # Create histogram
    hist, bin_edges = np.histogram(fitness_values, bins='auto')
    
    return {
        'mean': mean,
        'median': median,
        'std': std,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'percentiles': {
            '25': percentiles[0],
            '50': percentiles[1],
            '75': percentiles[2],
            '90': percentiles[3],
            '95': percentiles[4],
            '99': percentiles[5]
        },
        'skewness': skewness,
        'kurtosis': kurtosis,
        'normality_test': normality_test,
        'histogram': {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        },
        'sample_size': len(fitness_values),
        'input_sample_size': len(raw_values),
        'omitted_count': omitted_count,
        'nan_policy': nan_policy,
    }


# Wrappers for multi landcape analysis.
def _coerce_groups(*,
                   groups: Optional[Dict[str, np.ndarray]] = None,
                   landscapes: Optional[Mapping[str, FitnessLandscape]] = None,
                   value_fn: Optional[Callable[[FitnessLandscape], np.ndarray]] = None,
                   landscape: Optional[FitnessLandscape] = None,
                   layer_names: Optional[Sequence[str]] = None,
                   value_fn_layers: Optional[Callable[[FitnessLandscape, str], np.ndarray]] = None,
                   nan_policy: str = "raise") -> Dict[str, np.ndarray]:
    """
    Normalize input into {group_name: 1D float array}. Exactly one of
    the following input modes must be provided, groups, landscapes with
    value_fn or landscape + layer_names + value_fn_layers. Helper
    function to cleanly collate data. 

    Parameters
    ----------
    groups : Dict

    landscapes : Mapping

    value_fn : Callable

    landscape : FitnessLandscape

    layer_names : Sequence[str]

    value_fn_layers : Callable

    Returns
    -------
    out : Dict
        Cleaned dictionary of
        - group name (str) : (, N) shaped array.
    """
    groups_mode = groups is not None
    landscapes_mode = landscapes is not None or value_fn is not None
    layers_mode = (
        landscape is not None
        or layer_names is not None
        or value_fn_layers is not None
    )
    if sum((groups_mode, landscapes_mode, layers_mode)) != 1:
        raise ValueError(
            "Provide exactly ONE of: "
            "(groups) OR (landscapes + value_fn) OR (landscape + layer_names + value_fn_layers)."
        )
    if landscapes_mode and (landscapes is None or value_fn is None):
        raise ValueError("landscapes and value_fn must be provided together.")
    if layers_mode and (
        landscape is None or layer_names is None or value_fn_layers is None
    ):
        raise ValueError(
            "landscape, layer_names, and value_fn_layers must be provided together."
        )

    nan_policy = _validate_nan_policy(nan_policy)
    out: Dict[str, np.ndarray] = {}

    if groups is not None:
        for name, arr in groups.items():
            out[name], _ = _clean_numeric_sample(
                arr,
                name=f"group {name!r}",
                nan_policy=nan_policy,
            )
        return out

    if (landscapes is not None) and (value_fn is not None):
        for name, L in landscapes.items():
            out[name], _ = _clean_numeric_sample(
                value_fn(L),
                name=f"group {name!r}",
                nan_policy=nan_policy,
            )
        return out

    # landscape + layer_names + value_fn_layers
    for lname in layer_names:
        if lname in out:
            raise ValueError(f"layer_names contains duplicate name {lname!r}.")
        out[lname], _ = _clean_numeric_sample(
            value_fn_layers(landscape, lname),
            name=f"group {lname!r}",
            nan_policy=nan_policy,
        )
    return out


def _validate_group_count(groups: Mapping[str, np.ndarray]) -> None:
    if len(groups) < 2:
        raise ValueError("At least two non-empty groups are required.")


def _normalize_correction_method(correction_method: Optional[str]) -> str:
    if correction_method is None or correction_method == "none":
        return "none"
    if not isinstance(correction_method, str):
        raise TypeError("correction_method must be a string or None.")
    if correction_method not in _SUPPORTED_CORRECTIONS:
        supported = ", ".join(sorted(_SUPPORTED_CORRECTIONS))
        raise ValueError(
            "correction_method must be None, 'none', or one of "
            f"{supported}."
        )
    return correction_method


def _adjust_pvalues(p_values: np.ndarray, method: str) -> np.ndarray:
    """Adjust a finite family of p-values without an optional dependency."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be finite and between zero and one.")
    if p_values.size == 0 or method == "none":
        return p_values.copy()

    family_size = p_values.size
    if method == "bonferroni":
        return np.minimum(1.0, p_values * family_size)

    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    if method == "holm":
        ordered_adjusted = np.maximum.accumulate(
            ordered * np.arange(family_size, 0, -1)
        )
    else:  # Benjamini-Hochberg false-discovery-rate control.
        ordered_adjusted = np.minimum.accumulate(
            (ordered * family_size / np.arange(1, family_size + 1))[::-1]
        )[::-1]

    adjusted = np.empty_like(ordered_adjusted)
    adjusted[order] = np.minimum(1.0, ordered_adjusted)
    return adjusted


def _apply_multiple_testing(
    records: list[Dict[str, Any]],
    *,
    alpha: float,
    correction_method: str,
) -> None:
    """Add adjusted inference fields to a family of result records."""
    finite_records = [
        record for record in records if np.isfinite(float(record["p_value"]))
    ]
    adjusted = _adjust_pvalues(
        np.asarray([record["p_value"] for record in finite_records], dtype=float),
        correction_method,
    )
    family_size = len(finite_records)
    for record in records:
        raw_p = float(record["p_value"])
        record["raw_significant"] = bool(np.isfinite(raw_p) and raw_p < alpha)
        record["adjusted_p_value"] = np.nan
        record["significant"] = False
        record["alpha"] = alpha
        record["correction_method"] = correction_method
        record["correction_family_size"] = family_size

    for record, adjusted_p in zip(finite_records, adjusted):
        record["adjusted_p_value"] = float(adjusted_p)
        record["significant"] = bool(adjusted_p < alpha)


def _validate_run_tests(run_tests: Sequence[str]) -> tuple[str, ...]:
    if isinstance(run_tests, str):
        raise TypeError("run_tests must be a non-empty sequence of test names.")
    try:
        normalized = tuple(run_tests)
    except TypeError as error:
        raise TypeError(
            "run_tests must be a non-empty sequence of test names."
        ) from error
    if not normalized:
        raise ValueError("run_tests must contain at least one test name.")
    if any(not isinstance(test_name, str) for test_name in normalized):
        raise TypeError("Every run_tests value must be a string.")
    unknown = sorted(set(normalized) - _SUPPORTED_TESTS)
    if unknown:
        raise ValueError(
            f"Unsupported run_tests value(s): {unknown}; supported tests are "
            f"{sorted(_SUPPORTED_TESTS)}."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("run_tests must not contain duplicate test names.")
    return normalized


def hypothesis_testing(*,
                       groups: Optional[Dict[str, np.ndarray]] = None,
                       landscapes: Optional[Mapping[str, FitnessLandscape]] = None,
                       value_fn: Optional[Callable[[FitnessLandscape], np.ndarray]] = None,
                       landscape: Optional[FitnessLandscape] = None,
                       layer_names: Optional[Sequence[str]] = None,
                       value_fn_layers: Optional[Callable[[FitnessLandscape, str], np.ndarray]] = None,
                       alpha: float = 0.05,
                       equal_var: bool = False,
                       run_tests: Tuple[str, ...] = ("ttest", "mannwhitney", "ks"),
                       correction_method: Optional[str] = "holm",
                       nan_policy: str = "raise") -> Dict[str, Any]:
    """
    Pairwise hypothesis tests across groups of values. compares groups
    of numerical values using standard statistical hypothesis tests
    (t-test, Mann-Whitney U, Kolmogorov-Smirnov). Groups can be
    provided directly as arrays, derived from multiple landscapes via a
    user-defined function, or extracted from multiple fitness layers of
    a single landscape.

    Parameters
    ----------
    groups : Dict[str, np.ndarray], optional
        Dictionary mapping group names to 1D numerical arrays.
        Each array is treated as one group's sample values.
        Mutually exclusive with the other input modes.

    landscapes : Mapping[str, FitnessLandscape], optional
        Mapping from names to FitnessLandscape objects. Used together
        with `value_fn` to extract numerical arrays from each landscape.

    value_fn : Callable[[FitnessLandscape], np.ndarray], optional
        Function that takes a landscape and returns a 1D array of values
        (e.g. fitness values, spectral coefficients). Must be provided if
        `landscapes` is used.

    landscape : FitnessLandscape, optional
        A single landscape containing multiple layers. Used together
        with `layer_names` and `value_fn_layers` to extract values.

    layer_names : Sequence[str], optional
        Names of layers to extract from `landscape`.

    value_fn_layers : Callable[[FitnessLandscape, str], np.ndarray], optional
        Function that takes a landscape and layer name, and returns a
        1D array of values for that layer.

    alpha : float, default=0.05
        Significance threshold for all hypothesis tests.

    equal_var : bool, default=False
        If True, assume equal variance in the independent t-test
        (`scipy.stats.ttest_ind`). If False, Welch's t-test is used.

    run_tests : Tuple[str], default=("ttest", "mannwhitney", "ks")
        Tuple of test names to run. Supported values are:
        - `"ttest"`: independent two-sample t-test
        - `"mannwhitney"`: Mann-Whitney U test
        - `"ks"`: two-sample Kolmogorov-Smirnov test

    correction_method : {"holm", "bonferroni", "fdr_bh", "none"}, optional
        Multiple-testing correction applied across every finite p-value
        returned by this call. Holm correction is the default. Pass ``None``
        or ``"none"`` to opt out explicitly.

    nan_policy : {"raise", "omit"}, default="raise"
        Missing-value policy for group samples. Infinite values are always
        rejected. ``"omit"`` explicitly removes NaNs before validating group
        sizes.

    Returns
    -------
    out : Dict[str, Any]
        Dictionary with the following keys:
        
        - `"group_stats"` : Dict[str, Dict]
            Per-group descriptive statistics, including:
            `mean`, `median`, `std`, `min`, `max`, `n`.

        - `"pairwise_tests"` : Dict[str, Dict[str, Dict]]
            Nested dictionary of results for each group pair. Each test
            reports:
                - `"statistic"` : float
                - `"p_value"` : float, unadjusted p-value
                - `"adjusted_p_value"` : float
                - `"significant"` : bool, based on adjusted p-value
        The top-level ``correction_method`` and ``correction_family_size``
        fields define the multiple-testing family.
    """
    alpha = _validate_alpha(alpha)
    run_tests = _validate_run_tests(run_tests)
    correction_method = _normalize_correction_method(correction_method)
    if not isinstance(equal_var, (bool, np.bool_)):
        raise TypeError("equal_var must be a boolean.")

    clean = _coerce_groups(
        groups=groups,
        landscapes=landscapes, value_fn=value_fn,
        landscape=landscape, layer_names=layer_names, value_fn_layers=value_fn_layers,
        nan_policy=nan_policy,
    )
    _validate_group_count(clean)

    if "ttest" in run_tests:
        undersized = [name for name, values in clean.items() if values.size < 2]
        if undersized:
            raise ValueError(
                "ttest requires at least two observations in every group; "
                f"undersized groups: {undersized}."
            )

    # per-group stats
    group_stats = {}
    for name, x in clean.items():
        group_stats[name] = {
            'mean' : float(np.mean(x)) if x.size else np.nan,
            'median': float(np.median(x)) if x.size else np.nan,
            'std' : float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            'min' : float(np.min(x)) if x.size else np.nan,
            'max' : float(np.max(x)) if x.size else np.nan,
            'n' : int(x.size),
        }

    # pairwise tests
    pairwise = {}
    inference_records: list[Dict[str, Any]] = []
    names = list(clean.keys())
    for a, b in combinations(names, 2):
        x, y = clean[a], clean[b]
        out = {}

        if "ttest" in run_tests:
            t_stat, t_p = stats.ttest_ind(x, y, equal_var=equal_var)
            out['t_test'] = {'statistic': float(t_stat), 'p_value': float(t_p)}
            inference_records.append(out['t_test'])

        if "mannwhitney" in run_tests:
            u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
            out['mann_whitney'] = {'statistic': float(u_stat), 'p_value': float(u_p)}
            inference_records.append(out['mann_whitney'])

        if "ks" in run_tests:
            ks_stat, ks_p = stats.ks_2samp(x, y, alternative="two-sided", mode="auto")
            out['ks_test'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
            inference_records.append(out['ks_test'])

        pairwise.setdefault(a, {})[b] = out

    _apply_multiple_testing(
        inference_records,
        alpha=alpha,
        correction_method=correction_method,
    )
    return {
        'group_stats': group_stats,
        'pairwise_tests': pairwise,
        'alpha': alpha,
        'correction_method': correction_method,
        'correction_family_size': len(
            [record for record in inference_records if np.isfinite(record['p_value'])]
        ),
        'nan_policy': nan_policy,
    }


def _json_compatible(value: Any) -> Any:
    """Convert NumPy random-state values to portable built-in objects."""
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _resolve_random_state(
    random_state: Optional[int | np.random.Generator],
) -> tuple[np.random.Generator, Dict[str, Any]]:
    """Create or validate an RNG and describe its reproducibility source."""
    if isinstance(random_state, np.random.Generator):
        return random_state, {"kind": "generator"}
    if random_state is None:
        return np.random.default_rng(), {"kind": "entropy"}
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError("random_state must be None, an integer, or a Generator.")
    seed = int(random_state)
    if seed < 0:
        raise ValueError("random_state integer seeds must be non-negative.")
    return np.random.default_rng(seed), {"kind": "seed", "seed": seed}


def _snapshot_random_state(
    rng: np.random.Generator,
    source: Mapping[str, Any],
) -> Dict[str, Any]:
    """Record the exact generator state immediately before a comparison."""
    return {
        **source,
        "bit_generator": type(rng.bit_generator).__name__,
        "state": _json_compatible(copy.deepcopy(rng.bit_generator.state)),
    }


def _evaluate_statistic(
    statistic_func: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Require a finite scalar permutation statistic."""
    value = np.asarray(statistic_func(x, y), dtype=float)
    if value.ndim != 0 or not np.isfinite(value.item()):
        raise ValueError("statistic_func must return one finite scalar value.")
    return float(value.item())

def permutation_test(*,
                     groups: Optional[Dict[str, np.ndarray]] = None,
                     landscapes: Optional[Mapping[str, FitnessLandscape]] = None,
                     value_fn: Optional[Callable[[FitnessLandscape], np.ndarray]] = None,
                     landscape: Optional[FitnessLandscape] = None,
                     layer_names: Optional[Sequence[str]] = None,
                     value_fn_layers: Optional[Callable[[FitnessLandscape, str], np.ndarray]] = None,
                     statistic_func: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: float(np.mean(a) - np.mean(b)),
                     n_permutations: int = 1000,
                     alpha: float = 0.05,
                     alternative: str = "two-sided",
                     random_state: Optional[int | np.random.Generator] = None,
                     correction_method: Optional[str] = "holm",
                     nan_policy: str = "raise") -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Pairwise permutation tests across groups of values. Compares
    groups of numerical values by estimating the
    null distribution of a test statistic under random permutation of group
    labels. Groups can be provided directly as arrays, derived from multiple
    landscapes via a user-defined function, or extracted from multiple
    fitness layers of a single landscape.

    Parameters
    ----------
    groups : Dict[str, np.ndarray], optional
        Dictionary mapping group names to 1D numerical arrays.
        Each array is treated as one group's sample values.
        Mutually exclusive with the other input modes.

    landscapes : Mapping[str, FitnessLandscape], optional
        Mapping from names to FitnessLandscape objects. Used together
        with `value_fn` to extract numerical arrays from each landscape.

    value_fn : Callable[[FitnessLandscape], np.ndarray], optional
        Function that takes a landscape and returns a 1D array of values
        (e.g. fitness values, spectral coefficients). Must be provided if
        `landscapes` is used.

    landscape : FitnessLandscape, optional
        A single landscape containing multiple layers. Used together
        with `layer_names` and `value_fn_layers` to extract values.

    layer_names : Sequence[str], optional
        Names of layers to extract from `landscape`.

    value_fn_layers : Callable[[FitnessLandscape, str], np.ndarray], optional
        Function that takes a landscape and layer name, and returns a
        1D array of values for that layer.

    statistic_func : callable, optional
        Two-sample statistic. By default, compute ``mean(a) - mean(b)``.
    n_permutations : int, default=1000
        Number of random permutations of group labels to perform.

    alpha : float, default=0.05
        Significance threshold for determining whether observed
        statistics differ from the permutation null distribution.

    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Defines the alternative hypothesis for p-value computation:
        - `"two-sided"` : p-value is proportion of permuted statistics
          at least as extreme as the observed (absolute) statistic.
        - `"greater"` : p-value is proportion of permuted statistics
          greater than or equal to the observed statistic.
        - `"less"` : p-value is proportion of permuted statistics
          less than or equal to the observed statistic.

    random_state : int, numpy.random.Generator, or None, default=None
        Random source for label permutations. Integer seeds create a fresh
        generator; a supplied ``Generator`` is consumed in place. Each result
        records the exact bit-generator state used for its comparison.

    correction_method : {"holm", "bonferroni", "fdr_bh", "none"}, optional
        Multiple-testing correction across all pairwise comparisons. Holm
        correction is the default. Pass ``None`` or ``"none"`` to opt out.

    nan_policy : {"raise", "omit"}, default="raise"
        Missing-value policy for group samples. Infinite values are always
        rejected. ``"omit"`` explicitly removes NaNs.

    Returns
    -------
    results : Dict[Tuple[str, str], Dict[str, Any]]
        A mapping from (group_name_a, group_name_b) -> result dict with:
            - "observed": float
            - "p_value": float, estimated as ``(b + 1) / (B + 1)``
            - "adjusted_p_value": float
            - "significant": bool, based on the adjusted p-value
            - "n_permutations": int
            - "alternative": str
            - "extreme_count": int
            - "p_value_resolution": float
            - "monte_carlo_standard_error": float
            - "random_state": dict
            - "correction_method": str
    """
    alpha = _validate_alpha(alpha)
    if isinstance(n_permutations, (bool, np.bool_)) or not isinstance(
        n_permutations, (int, np.integer)
    ):
        raise TypeError("n_permutations must be a positive integer.")
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer.")
    if not isinstance(alternative, str):
        raise TypeError("alternative must be a string.")
    if alternative not in _SUPPORTED_ALTERNATIVES:
        raise ValueError(
            "alternative must be one of 'two-sided', 'greater', or 'less'."
        )
    if not callable(statistic_func):
        raise TypeError("statistic_func must be callable.")
    correction_method = _normalize_correction_method(correction_method)

    clean = _coerce_groups(
        groups=groups,
        landscapes=landscapes, value_fn=value_fn,
        landscape=landscape, layer_names=layer_names, value_fn_layers=value_fn_layers,
        nan_policy=nan_policy,
    )
    _validate_group_count(clean)

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    names = list(clean.keys())
    inference_records: list[Dict[str, Any]] = []

    rng, random_source = _resolve_random_state(random_state)
    for a, b in combinations(names, 2):
        x, y = clean[a], clean[b]
        rng_state = _snapshot_random_state(rng, random_source)
        observed = _evaluate_statistic(statistic_func, x, y)

        pooled = np.concatenate([x, y])
        n1 = x.size
        perm_stats = np.empty(n_permutations, dtype=float)

        for i in range(n_permutations):
            permuted = rng.permutation(pooled)
            perm_stats[i] = _evaluate_statistic(
                statistic_func,
                permuted[:n1],
                permuted[n1:],
            )

        if alternative == "two-sided":
            extreme_count = int(np.count_nonzero(np.abs(perm_stats) >= abs(observed)))
        elif alternative == "greater":
            extreme_count = int(np.count_nonzero(perm_stats >= observed))
        else:
            extreme_count = int(np.count_nonzero(perm_stats <= observed))

        p = float((extreme_count + 1) / (n_permutations + 1))
        standard_error = float(
            np.sqrt(n_permutations * p * (1.0 - p))
            / (n_permutations + 1)
        )

        results[(a, b)] = {
            'observed': observed,
            'p_value': p,
            'n_permutations': n_permutations,
            'alternative': alternative,
            'extreme_count': extreme_count,
            'p_value_estimator': '(b + 1) / (B + 1)',
            'p_value_resolution': float(1.0 / (n_permutations + 1)),
            'monte_carlo_standard_error': standard_error,
            'random_state': rng_state,
        }
        inference_records.append(results[(a, b)])

    _apply_multiple_testing(
        inference_records,
        alpha=alpha,
        correction_method=correction_method,
    )

    return results

 # Aggregation helpers
def _is_scalar_list(xs: Sequence[Any]) -> bool:
    try:
        arr = np.asarray(xs, dtype=float)
        return arr.ndim == 1 and arr.size == len(xs)
    except Exception:
        return False

def _summarize_arr(arr: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": float(np.percentile(arr, lo)),
        "ci_high": float(np.percentile(arr, hi)),
        "alpha": alpha,
    }

def _analyze_worker(subL, layer_name, analysis_func):
    if layer_name is not None:
        subL.view(layer_name)
    return analysis_func(subL)

def _parallel_analyze_landscapes(landscape_samples: List[FitnessLandscape],
                                 analysis_func: Callable[[FitnessLandscape], Any],
                                 layer_name: str | None = None,
                                 use_ray: bool = True,
                                 num_workers: int | None = None) -> list:
    """
    Helper function to analyze multiple fitness landscapes in parallel
    using Ray. If `use_ray` is False, runs serially.
    """
    if not use_ray:
        out = []
        for subL in landscape_samples:
            if layer_name is not None:
                subL.view(layer_name)
            out.append(analysis_func(subL))
        return out

    if not landscape_samples:
        return []

    workers = len(landscape_samples) if num_workers is None else int(num_workers)
    workers = max(1, min(workers, len(landscape_samples)))
    with ray_runtime(workers, purpose="parallel landscape statistics") as ray:
        analyze_worker_remote = ray.remote(_analyze_worker)
        func_ref = ray.put(analysis_func)
        obj_refs = [ray.put(sl) for sl in landscape_samples]
        futures = [
            analyze_worker_remote.remote(sl_ref, layer_name, func_ref)
            for sl_ref in obj_refs
        ]
        return ray.get(futures)

def subsample_analysis(landscape: FitnessLandscape,
                       analysis_func: Callable[FitnessLandscape, Any],
                       n_samples: int = 100,
                       subsample_node_prop: float = 0.9,
                       subsample_edge_prop: float = 0.9,
                       seed: int = None,
                       layer_name: Optional[str] = None,
                       use_ray: bool = True,
                       num_workers: int = None) -> Dict:
    """
    Function to subsample a fitness landscape object into connected
    component subgraphs and compute an analysis function. Edges are not
    recomputed on subsampled nodes and all subgraphs used in sampling
    are subgraphs of the intput fitness landscape. 

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    analysis_func : Callable
        The analysis function to call on the subsampled fitness
        landscape graphs. 
    
    n_samples : int, default=100
        The number of subsamples to draw. 
    
    subsample_node_prop : float, default=0.9
        The proportion of nodes in `landscape` that are subsampled in
        each induced subgraph. 
    
    subsample_edge_prop : float, default=0.9
        The proportion of edges in `landscape` that are subsampled in
        each induced subgraph. 
    
    seed : int, optional
        Seed used to derive an independent seed for each subsample.

    layer_name : str, optional
        Fitness layer activated on each subsampled landscape before analysis.

    use_ray : bool, default=True
        Whether to use Ray for parallel processing. If False, runs
        serially.
    
    num_workers : int, optional
        Number of parallel workers to use with Ray. If None, uses all
        available CPUs. Ignored if `use_ray` is False.

    Returns
    -------
    dict
        Raw per-sample results and, when results are numeric or numeric
        dictionaries, empirical means, sample standard deviations, and 95%
        percentile intervals.
    """
    rng = np.random.default_rng(seed)
    results: list[Any] = []
    landscape_samples: list[FitnessLandscape] = []

    # Collect samples.
    for i in range(n_samples):
        sub_seed = int(rng.integers(0, 2**63 - 1))
        subL = sample_observed_induced_connected(
            landscape,
            node_keep=subsample_node_prop,
            edge_keep=subsample_edge_prop,
            seed=sub_seed,
            return_graph=False,
        )
        landscape_samples.append(subL)

    # Run in parallel.
    results: list[Any] = _parallel_analyze_landscapes(
        landscape_samples=landscape_samples,
        analysis_func=analysis_func,
        layer_name=layer_name,
        use_ray=use_ray,
        num_workers=num_workers,
)

    # scalar outputs easiest, summarize directly
    if _is_scalar_list(results):
        arr = np.asarray(results, dtype=float)
        return {
        "results": results,
        "summary": _summarize_arr(arr, alpha=0.05),
    }

    # dict-of-numerics outputs with consistent keys, harder per-key summary
    if all(isinstance(r, dict) for r in results):
        # find common keys across all dicts
        keys = set(results[0].keys())
        for r in results[1:]:
            keys &= set(r.keys())
        # keep only numeric keys
        per_key_samples: Dict[str, list[float]] = {k: [] for k in keys}
        for r in results:
            for k in list(per_key_samples.keys()):
                v = r.get(k, None)
                try:
                    per_key_samples[k].append(float(v))
                except Exception:
                    # Not numeric → drop this key from aggregation
                    per_key_samples.pop(k, None)

        
        per_key_summary: Dict[str, Dict[str, Any]] = {}
        for k, vals in per_key_samples.items():
            arr = np.asarray(vals, dtype=float)
            per_key_summary[k] = {
                **_summarize_arr(arr, alpha=0.05),
                "samples": vals,
            }

        # If nothing numeric return raw results.
        if not per_key_summary:
            return {"results": results}

        return {
            "results": results,
            "per_key": per_key_summary,
        }

    # Fallback on heterogeneous or non-numeric outputs and return raw list.
    return {"results": results}
