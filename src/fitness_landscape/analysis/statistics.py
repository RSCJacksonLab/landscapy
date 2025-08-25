from itertools import combinations
import numpy as np
import scipy.stats as stats
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Mapping, Sequence
from ..core.landscape import FitnessLandscape
from ..core.superscape import FitnessSuperscape
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from ..utils import sample_observed_induced_connected

def analyze_fitness_distribution(landscape: FitnessLandscape,
                                 **kwargs) -> Dict:
    """
    Analyze the distribution of fitness values in a landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
        
    Returns
    -------
    dict
        Distribution analysis results.
    """
    # Extract fitness values
    fitness_values = np.array([landscape.get_fitness(seq) for seq in landscape.sequences])
    
    # Calculate basic statistics
    mean = np.mean(fitness_values)
    median = np.median(fitness_values)
    std = np.std(fitness_values)
    min_val = np.min(fitness_values)
    max_val = np.max(fitness_values)
    range_val = max_val - min_val
    
    # Calculate percentiles
    percentiles = np.percentile(fitness_values, [25, 50, 75, 90, 95, 99])
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(fitness_values)
    kurtosis = stats.kurtosis(fitness_values)
    
    # Test for normality
    shapiro_test = stats.shapiro(fitness_values)
    
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
        'normality_test': {
            'shapiro_statistic': shapiro_test[0],
            'shapiro_p_value': shapiro_test[1],
            'is_normal': shapiro_test[1] > 0.05
        },
        'histogram': {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        },
        'sample_size': len(fitness_values)
    }


# Wrappers for multi landcape analysis.
def _coerce_groups(*,
                   groups: Optional[Dict[str, np.ndarray]] = None,
                   landscapes: Optional[Mapping[str, FitnessLandscape]] = None,
                   value_fn: Optional[Callable[[FitnessLandscape], np.ndarray]] = None,
                   landscape: Optional[FitnessLandscape] = None,
                   layer_names: Optional[Sequence[str]] = None,
                   value_fn_layers: Optional[Callable[[FitnessLandscape, str], np.ndarray]] = None) -> Dict[str, np.ndarray]:
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
    mode_flags = [
        groups is not None,
        (landscapes is not None) and (value_fn is not None),
        (landscape is not None) and (layer_names is not None) and (value_fn_layers is not None),
    ]
    if sum(bool(f) for f in mode_flags) != 1:
        raise ValueError(
            "Provide exactly ONE of: "
            "(groups) OR (landscapes + value_fn) OR (landscape + layer_names + value_fn_layers)."
        )

    out: Dict[str, np.ndarray] = {}

    if groups is not None:
        for name, arr in groups.items():
            x = np.asarray(arr, dtype=float).ravel()
            out[name] = x[~np.isnan(x)]
        return out

    if (landscapes is not None) and (value_fn is not None):
        for name, L in landscapes.items():
            vec = np.asarray(value_fn(L), dtype=float).ravel()
            out[name] = vec[~np.isnan(vec)]
        return out

    # landscape + layer_names + value_fn_layers
    for lname in layer_names:
        vec = np.asarray(value_fn_layers(landscape, lname), dtype=float).ravel()
        out[lname] = vec[~np.isnan(vec)]
    return out


def hypothesis_testing(*,
                       groups: Optional[Dict[str, np.ndarray]] = None,
                       landscapes: Optional[Mapping[str, FitnessLandscape]] = None,
                       value_fn: Optional[Callable[[FitnessLandscape], np.ndarray]] = None,
                       landscape: Optional[FitnessLandscape] = None,
                       layer_names: Optional[Sequence[str]] = None,
                       value_fn_layers: Optional[Callable[[FitnessLandscape, str], np.ndarray]] = None,
                       alpha: float = 0.05,
                       equal_var: bool = False,
                       run_tests: Tuple[str, ...] = ("ttest", "mannwhitney", "ks")) -> Dict[str, Any]:
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
                - `"p_value"` : float
                - `"significant"` : bool (p < alpha)
    """
    clean = _coerce_groups(
        groups=groups,
        landscapes=landscapes, value_fn=value_fn,
        landscape=landscape, layer_names=layer_names, value_fn_layers=value_fn_layers,
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
    names = list(clean.keys())
    for a, b in combinations(names, 2):
        x, y = clean[a], clean[b]
        out = {}

        if "ttest" in run_tests and x.size > 1 and y.size > 1:
            t_stat, t_p = stats.ttest_ind(x, y, equal_var=equal_var)
            out['t_test'] = {'statistic': float(t_stat), 'p_value': float(t_p), 'significant': bool(t_p < alpha)}

        if "mannwhitney" in run_tests and x.size > 0 and y.size > 0:
            u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
            out['mann_whitney'] = {'statistic': float(u_stat), 'p_value': float(u_p), 'significant': bool(u_p < alpha)}

        if "ks" in run_tests and x.size > 0 and y.size > 0:
            ks_stat, ks_p = stats.ks_2samp(x, y, alternative="two-sided", mode="auto")
            out['ks_test'] = {'statistic': float(ks_stat), 'p_value': float(ks_p), 'significant': bool(ks_p < alpha)}

        pairwise.setdefault(a, {})[b] = out

    return {'group_stats': group_stats, 'pairwise_tests': pairwise}

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
                     alternative: str = "two-sided") -> Dict[Tuple[str, str], Dict[str, Any]]:
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

    
    n_permutations : int, default=1000
        Number of random permutations of group labels to perform.

    alpha : float, default=0.05
        Significance threshold for determining whether observed
        statistics differ from the permutation null distribution.

    equal_var : bool, default=False
        Placeholder for compatibility with `hypothesis_testing`.
        Does not affect permutation tests, unless the provided
        `statistic_func` explicitly uses it.

    run_tests : Tuple[str], optional
        Optional names of test variants to run in parallel. 
        If provided, must correspond to keys in a registry of
        `statistic_func` implementations.

    alternative : {"two_sided", "greater", "less"}, default="two_sided"
        Defines the alternative hypothesis for p-value computation:
        - `"two_sided"` : p-value is proportion of permuted statistics
          at least as extreme as the observed (absolute) statistic.
        - `"greater"` : p-value is proportion of permuted statistics
          greater than or equal to the observed statistic.
        - `"less"` : p-value is proportion of permuted statistics
          less than or equal to the observed statistic.

    Returns
    -------
    results : Dict[Tuple[str, str], Dict[str, Any]]
        A mapping from (group_name_a, group_name_b) -> result dict with:
            - "observed": float
            - "p_value": float
            - "significant": bool  (p_value < alpha)
            - "n_permutations": int
            - "alternative": str
    """
    clean = _coerce_groups(
        groups=groups,
        landscapes=landscapes, value_fn=value_fn,
        landscape=landscape, layer_names=layer_names, value_fn_layers=value_fn_layers,
    )

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    names = list(clean.keys())

    rng = np.random.default_rng()
    for a, b in combinations(names, 2):
        x, y = clean[a], clean[b]
        if x.size == 0 or y.size == 0:
            results[(a, b)] = {
                'observed': np.nan, 'p_value': np.nan, 'significant': False,
                'n_permutations': n_permutations, 'alternative': alternative
            }
            continue

        observed = float(statistic_func(x, y))

        pooled = np.concatenate([x, y])
        n1 = x.size
        perm_stats = np.empty(n_permutations, dtype=float)

        for i in range(n_permutations):
            rng.shuffle(pooled)
            perm_stats[i] = statistic_func(pooled[:n1], pooled[n1:])

        if alternative == "two-sided":
            p = float(np.mean(np.abs(perm_stats) >= abs(observed)))
        elif alternative == "greater":
            p = float(np.mean(perm_stats >= observed))
        else:  # 'less'
            p = float(np.mean(perm_stats <= observed))

        results[(a, b)] = {
            'observed': observed,
            'p_value': p,
            'significant': bool(p < alpha),
            'n_permutations': n_permutations,
            'alternative': alternative,
        }

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

def subsample_analysis(landscape: FitnessLandscape,
                       analysis_fn: Callable[FitnessLandscape, Any],
                       n_samples: int = 100,
                       subsample_node_prop: float = 0.9,
                       subsample_edge_prop: float = 0.9,
                       seed: int = None,
                       layer_name: Optional[str] = None) -> Dict:
    """
    Function to subsample a fitness landscape object into connected
    component subgraphs and compute an analysis function. Edges are not
    recomputed on subsampled nodes and all subgraphs used in sampling
    are subgraphs of the intput fitness landscape. 

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    analysis_fn : Callable
        The analysis function to call on the subsampled fitness
        landscape graphs. 
    
    n_samples : int, default=100
        The number of subsamples to draw. 
    
    subsample_node_prop: float, default=0.9
        The proportion of nodes in `landscape` that are subsampled in
        each induced subgraph. 
    
    subsample_edge_prop: float, default=0.9
        The proportion of edges in `landscape` that are subsampled in
        each induced subgraph. 
    """
    rng = np.random.default_rng(seed)
    results: list[Any] = []

    for i in range(n_samples):
        sub_seed = int(rng.integers(0, 2**63 - 1))
        subL = sample_observed_induced_connected(
            landscape,
            node_keep=subsample_node_prop,
            edge_keep=subsample_edge_prop,
            seed=sub_seed,
            return_graph=False,
        )
        if layer_name is not None:
            # No-op if it's already active; raises if the name is invalid
            subL.view(layer_name)

        out = analysis_fn(subL)
        results.append(out)

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

def sample_latent_graph_analysis(landscape: FitnessSuperscape,
                                 analysis_fn: Callable[[FitnessLandscape], Any],
                                 n_samples: int = 100,
                                 layer_name: Optional[str] = None,
                                 seed: int = None) -> Dict:
    """
    Function to sample latent graphs from a superscape and compute
    an analysis function on each sampled graph.

    Parameters
    ----------
    landscape : FitnessSuperscape
        The superscape to analyze.
    
    analysis_fn : Callable
        The analysis function to call on the sampled fitness
        landscape graphs. Should be a `lambda L: ...` function that
        takes a single FitnessLandscape object and returns a scalar or
        dictionary of numeric values.

    n_samples : int, default=100
        The number of latent graphs to sample and analyze.

    subsample_edge_prop: float, default=0.9
        The proportion of edges in `landscape` that are subsampled in
        each induced subgraph. 

    Returns
    ------- 
    """
    if not isinstance(landscape, FitnessSuperscape):
        raise ValueError("landscape must be a FitnessSuperscape.")

    if not hasattr(landscape, 'latent_landscape'):
        raise RuntimeError("The latent landscape has not been constructed yet. "
                            "Run `construct_latent_landscape()` first.")

    results: list[Any] = []

    landscape_samples = landscape.sample_latent_landscapes(n_samples=n_samples, seed=seed)
    for landscape in landscape_samples:
        if layer_name is not None:
            # No-op if it's already active; raises if the name is invalid
            landscape.view(layer_name)

        out = analysis_fn(subL)
        results.append(out)

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
                    # Not numeric  drop this key from aggregation
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