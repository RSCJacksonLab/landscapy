"""Estimate fitness-landscape diffusion scales and uncertainty."""

import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import chi2
from typing import Literal, Optional, Tuple, TypedDict, Union
from ..transforms.eigenmode import eigenmode_decomposition
from ..transforms.graph_fourier import graph_fourier_transform
from ..core.landscape import FitnessLandscape


class DiffusionScaleResult(TypedDict):
    """Result schema for diffusion-scale estimation."""

    t_map: float
    t_lower_confidence_interval: float
    t_upper_confidence_interval: float
    t_logposterior_map: float
    variance_approximate: float

def _precompute_GMRF_stats(G: nx.Graph,
                           signal: np.ndarray,
                           _eigenvalues: Optional[np.ndarray] = None,
                           _eigenvectors: Optional[np.ndarray] = None) -> Tuple:
    """
    Function to precompute GMRF spectral and statistical quantities.

    Parameters
    ----------
    G : nx.Graph
        The fitness landscape graph. 
    
    Returns:
    --------
    f_hat : np.array
        The Fourier transformed graph signal. 
    
    eigenvalues : np.array  
        The Laplacian eigenvalues. 
    
    sigma_squared : float
        The empirical variance in the signal. 
    """
    if (_eigenvalues is None) != (_eigenvectors is None):
        raise ValueError("Provide both _eigenvalues and _eigenvectors or neither.")
    if _eigenvalues is not None:
        eigenvalues = np.asarray(_eigenvalues, dtype=float)
        eigenvectors = np.asarray(_eigenvectors, dtype=float)
    else:
        eigenvalues, eigenvectors = eigenmode_decomposition(G,
                                                            matrix='norm_laplacian',
                                                            weight_key=None)

    mu = np.mean(signal)
    
    # Centre signal on average
    signal_centered = signal - mu
    
    # GFT on norm Laplacian
    f_hat = eigenvectors.T @ signal_centered
    
    sigma_squared = np.var(signal_centered, ddof=1)
    return f_hat, eigenvalues, sigma_squared


def _precompute_GMRF_stats_with_evecs(G: nx.Graph,
                                      signal: np.ndarray,
                                      _eigenvalues: Optional[np.ndarray] = None,
                                      _eigenvectors: Optional[np.ndarray] = None) -> Tuple:
    """
    Variant of _precompute_GMRF_stats that also returns eigenvectors.
    """
    if (_eigenvalues is None) != (_eigenvectors is None):
        raise ValueError("Provide both _eigenvalues and _eigenvectors or neither.")
    if _eigenvalues is not None:
        eigenvalues = np.asarray(_eigenvalues, dtype=float)
        eigenvectors = np.asarray(_eigenvectors, dtype=float)
    else:
        eigenvalues, eigenvectors = eigenmode_decomposition(
            G,
            matrix='norm_laplacian',
            weight_key=None,
        )
    mu = np.mean(signal)
    signal_centered = signal - mu
    f_hat = eigenvectors.T @ signal_centered
    sigma_squared = np.var(signal_centered, ddof=1)
    return f_hat, eigenvalues, eigenvectors, sigma_squared, mu


def _grid_posterior_from_stats(f_hat: np.ndarray,
                               eigenvalues: np.ndarray,
                               sigma_squared: float,
                               t_grid: np.ndarray,
                               epsilon: float = 1e-8,
                               prior: Literal["uniform", "log_uniform"] = "log_uniform"
                               ) -> Tuple[float, float, float, float, float]:
    """
    Compute MAP and credible interval on a fixed t-grid.
    """
    loglik = np.array(
        [
            compute_log_likelihood_H0(
                f_hat=f_hat,
                eigenvalues=eigenvalues,
                t=float(t),
                sigma_squared=sigma_squared,
                epsilon=epsilon,
            )[0]
            for t in t_grid
        ],
        dtype=float,
    )

    if prior == "uniform":
        log_prior = np.zeros_like(loglik)
    elif prior == "log_uniform":
        log_prior = -np.log(t_grid)
    else:
        raise ValueError(f"Unknown prior '{prior}'. Use 'uniform' or 'log_uniform'.")

    logpost = loglik + log_prior

    # Weight by grid spacing to approximate posterior mass in t.
    delta_t = np.empty_like(t_grid)
    delta_t[0] = t_grid[1] - t_grid[0]
    delta_t[-1] = t_grid[-1] - t_grid[-2]
    if len(t_grid) > 2:
        delta_t[1:-1] = 0.5 * (t_grid[2:] - t_grid[:-2])
    delta_t = np.clip(delta_t, 1e-12, np.inf)

    log_weights = logpost + np.log(delta_t)
    log_norm = logsumexp(log_weights)
    weights = np.exp(log_weights - log_norm)

    if not np.isfinite(weights).all() or np.sum(weights) <= 0:
        raise ValueError("Posterior grid normalization failed; check t_min/t_max or signal variance.")

    idx_map = int(np.argmax(logpost))
    t_map = float(t_grid[idx_map])
    logpost_map = float(logpost[idx_map])

    mean = float(np.sum(weights * t_grid))
    var = float(np.sum(weights * (t_grid - mean) ** 2))

    cdf = np.cumsum(weights)
    cdf = np.clip(cdf, 0.0, 1.0)
    ci_lower = float(np.interp(0.025, cdf, t_grid))
    ci_upper = float(np.interp(0.975, cdf, t_grid))

    return t_map, ci_lower, ci_upper, logpost_map, var

def compute_log_likelihood_H0(f_hat: np.ndarray,
                              eigenvalues: np.ndarray,
                              t: float,
                              sigma_squared: float,
                              epsilon: float = 1e-8) -> tuple:
    """Compute the Gaussian log likelihood under the heat-kernel GMRF.

    Parameters
    ----------
    f_hat : ndarray
        Mean-centred graph signal expressed in the Laplacian eigenbasis.
    eigenvalues : ndarray
        Graph-Laplacian eigenvalues corresponding to ``f_hat``.
    t : float
        Non-negative heat-kernel diffusion scale.
    sigma_squared : float
        Empirical variance of the centred signal.
    epsilon : float, default=1e-8
        Positive eigenvalue offset for numerical stability.

    Returns
    -------
    log_likelihood : float
        Gaussian log likelihood.
    log_det : float
        Log determinant of the scaled heat-kernel covariance.
    quadratic_form : float
        Signal quadratic form under the inverse covariance.

    Notes
    -----
    Heat-kernel eigenvalues are ``exp(-t * (lambda + epsilon))`` and
    rescaled so their sum equals ``n * sigma_squared``.
    """

    n = len(f_hat)

    # Adjust eigenvalues to avoid zero
    lambda_adjusted = eigenvalues + epsilon

    # Compute heat kernel eigenvalues
    h_i = np.exp(-t * lambda_adjusted)

    # Compute scaling factor
    scaling_factor = (sigma_squared * n) / np.sum(h_i)

    # Scale heat kernel eigenvalues
    h_i_scaled = h_i * scaling_factor

    # Compute inverse of scaled heat kernel eigenvalues
    inv_h_i_scaled = 1 / h_i_scaled

    # Compute quadratic form
    quadratic_form = np.sum(inv_h_i_scaled * (f_hat ** 2))

    # Compute log-determinant
    log_det = np.sum(np.log(h_i_scaled))

    # Compute log-likelihood
    log_likelihood = -0.5 * quadratic_form - 0.5 * log_det - (n / 2) * np.log(2 * np.pi)

    return log_likelihood, log_det, quadratic_form

def fit_t_bayesian_laplace(G: nx.Graph,
                           signal: np.ndarray,
                           t_min: float = 0.01,
                           t_max: float = 1000.0,
                           epsilon: float = 1e-8,
                           _eigenvalues: Optional[np.ndarray] = None,
                           _eigenvectors: Optional[np.ndarray] = None) -> Tuple:
    """
    Function to estimate the Posterior probability distribution of t
    using the Laplace approximation.

    Parameters
    ----------
    G : nx.Graph
        The fitness landscape graph. 

    signal : ndarray
        Scalar graph signal in graph-node order.

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.
    
    _eigenvalues : ndarray, optional
        Precomputed normalized-Laplacian eigenvalues.
    _eigenvectors : ndarray, optional
        Precomputed normalized-Laplacian eigenvectors.

    Returns
    -------
    t_map : float
        The maximum a posteriori t value.
    
    ci_lower : float
        The lower bound of the confidence interval on t.
    
    ci_upper : float
        The upper bound of the confidence interval on t.
    
    logpost_map: float
        The log posterior probability of the MAP t value.
    
    var_approx : float
        The variance approximated as the second derivative of 
        the negative log posterior with respect to t. 
    """

    f_hat, eigenvalues, sigma_squared = _precompute_GMRF_stats(
        G,
        signal,
        _eigenvalues=_eigenvalues,
        _eigenvectors=_eigenvectors,
    )

    # Build a closure that has access to the above variables
    def neg_log_post(t: float) -> float:
        # enforce support of the (improper) uniform prior
        if (t < t_min) or (t > t_max):
            return np.inf
        ll, _, _ = compute_log_likelihood_H0(
            f_hat=f_hat,
            eigenvalues=eigenvalues,
            t=t,
            sigma_squared=sigma_squared,
            epsilon=epsilon
        )
        log_pri = 0.0  # uniform on [t_min, t_max]
        return -(ll + log_pri)

    # Optimize on the bounded interval
    result = minimize_scalar(
        neg_log_post,
        bounds=(t_min, t_max),
        method='bounded',
        options={'maxiter': 200}
    )
    t_map = float(result.x)

    # Numerical second derivative at t_map for Laplace variance
    base_h = 1e-4 * max(1.0, abs(t_map))
    h_left  = t_map - max(t_min, t_map - base_h)
    h_right = min(t_max, t_map + base_h) - t_map
    h = min(h_left if h_left > 0 else base_h, h_right if h_right > 0 else base_h)

    # If t_map is at a boundary, use a one-sided curvature fallback
    f0 = neg_log_post(t_map)
    if t_map - h >= t_min and t_map + h <= t_max:
        f_minus = neg_log_post(t_map - h)
        f_plus  = neg_log_post(t_map + h)
        second_deriv = (f_plus - 2.0*f0 + f_minus) / (h**2)
    elif t_map + 2*h <= t_max:
        f1 = neg_log_post(t_map + h)
        f2 = neg_log_post(t_map + 2*h)
        # second derivative from forward finite differences
        second_deriv = (f2 - 2.0*f1 + f0) / (h**2)
    elif t_map - 2*h >= t_min:
        f1 = neg_log_post(t_map - h)
        f2 = neg_log_post(t_map - 2*h)
        # second derivative from backward finite differences
        second_deriv = (f2 - 2.0*f1 + f0) / (h**2)
    else:
        # Degenerate interval; pick a tiny positive curvature to avoid blow-up
        second_deriv = 1e-12

    if not np.isfinite(second_deriv) or second_deriv <= 0:
        second_deriv = 1e-12

    var_approx = 1.0 / second_deriv
    std_approx = float(np.sqrt(var_approx))

    # 95% CI (Gaussian/Laplace approx), clipped to [t_min, t_max]
    ci_lower = max(t_min, t_map - 1.96 * std_approx)
    ci_upper = min(t_max, t_map + 1.96 * std_approx)

    logpost_map = -f0

    return t_map, ci_lower, ci_upper, logpost_map, var_approx


def fit_t_grid_posterior(G: nx.Graph,
                         signal: np.ndarray,
                         t_min: float = 0.01,
                         t_max: float = 1000.0,
                         epsilon: float = 1e-8,
                         grid_size: int = 512,
                         prior: Literal["uniform", "log_uniform"] = "log_uniform",
                         _eigenvalues: Optional[np.ndarray] = None,
                         _eigenvectors: Optional[np.ndarray] = None,
                         ) -> Tuple:
    """Estimate diffusion scale with a grid posterior in log-space.

    Parameters
    ----------
    G : networkx.Graph
        Undirected landscape graph.
    signal : ndarray
        Scalar graph signal in graph-node order.
    t_min : float, default=0.01
        Lower grid bound; must be positive.
    t_max : float, default=1000.0
        Upper grid bound.
    epsilon : float, default=1e-8
        Eigenvalue offset for likelihood stability.
    grid_size : int, default=512
        Number of logarithmically spaced candidate scales.
    prior : {'uniform', 'log_uniform'}, default='log_uniform'
        Prior density evaluated on the scale grid.
    _eigenvalues : ndarray, optional
        Precomputed normalized-Laplacian eigenvalues.
    _eigenvectors : ndarray, optional
        Precomputed normalized-Laplacian eigenvectors.

    Returns
    -------
    t_map : float
        Maximum-a-posteriori scale.
    ci_lower : float
        Lower bound of the equal-tail 95% credible interval.
    ci_upper : float
        Upper bound of the equal-tail 95% credible interval.
    logpost_map : float
        Log posterior at ``t_map``.
    var_approx : float
        Posterior variance on the discrete grid.

    Raises
    ------
    ValueError
        If fewer than ten grid points are requested.
    """
    f_hat, eigenvalues, sigma_squared = _precompute_GMRF_stats(
        G,
        signal,
        _eigenvalues=_eigenvalues,
        _eigenvectors=_eigenvectors,
    )

    if grid_size < 10:
        raise ValueError("grid_size must be >= 10 for a stable posterior estimate.")

    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), grid_size)
    return _grid_posterior_from_stats(
        f_hat=f_hat,
        eigenvalues=eigenvalues,
        sigma_squared=sigma_squared,
        t_grid=t_grid,
        epsilon=epsilon,
        prior=prior,
    )


def fit_t_profile_likelihood(G: nx.Graph,
                             signal: np.ndarray,
                             t_min: float = 0.01,
                             t_max: float = 1000.0,
                             epsilon: float = 1e-8,
                             grid_size: int = 512,
                             alpha: float = 0.05,
                             _eigenvalues: Optional[np.ndarray] = None,
                             _eigenvectors: Optional[np.ndarray] = None,
                             ) -> Tuple:
    """Estimate diffusion scale with a profile-likelihood grid.

    Parameters
    ----------
    G : networkx.Graph
        Undirected landscape graph.
    signal : ndarray
        Scalar graph signal in graph-node order.
    t_min : float, default=0.01
        Lower scale-grid bound.
    t_max : float, default=1000.0
        Upper scale-grid bound.
    epsilon : float, default=1e-8
        Eigenvalue offset for likelihood stability.
    grid_size : int, default=512
        Number of logarithmically spaced candidate scales.
    alpha : float, default=0.05
        Tail probability for the chi-square likelihood-ratio interval.
    _eigenvalues : ndarray, optional
        Precomputed normalized-Laplacian eigenvalues.
    _eigenvectors : ndarray, optional
        Precomputed normalized-Laplacian eigenvectors.

    Returns
    -------
    t_map : float
        Maximum-likelihood scale.
    ci_lower : float
        Lower profile-likelihood confidence bound.
    ci_upper : float
        Upper profile-likelihood confidence bound.
    logpost_map : float
        Log likelihood at ``t_map``.
    var_approx : float
        Variance proxy derived from the interval width.

    Raises
    ------
    ValueError
        If fewer than ten grid points are requested.
    """
    f_hat, eigenvalues, sigma_squared = _precompute_GMRF_stats(
        G,
        signal,
        _eigenvalues=_eigenvalues,
        _eigenvectors=_eigenvectors,
    )

    if grid_size < 10:
        raise ValueError("grid_size must be >= 10 for a stable likelihood profile.")

    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), grid_size)
    loglik = np.array(
        [
            compute_log_likelihood_H0(
                f_hat=f_hat,
                eigenvalues=eigenvalues,
                t=float(t),
                sigma_squared=sigma_squared,
                epsilon=epsilon,
            )[0]
            for t in t_grid
        ],
        dtype=float,
    )

    idx_map = int(np.argmax(loglik))
    t_map = float(t_grid[idx_map])
    logpost_map = float(loglik[idx_map])

    thresh = logpost_map - 0.5 * float(chi2.ppf(1.0 - alpha, df=1))
    mask = loglik >= thresh
    if not np.any(mask):
        ci_lower, ci_upper = float(t_min), float(t_max)
    else:
        idx = np.where(mask)[0]
        lo_idx, hi_idx = idx[0], idx[-1]
        ci_lower = float(t_grid[lo_idx])
        ci_upper = float(t_grid[hi_idx])

        # Linear interpolation for bounds when possible
        if lo_idx > 0:
            x0, x1 = loglik[lo_idx - 1], loglik[lo_idx]
            t0, t1 = t_grid[lo_idx - 1], t_grid[lo_idx]
            if x1 != x0:
                frac = (thresh - x0) / (x1 - x0)
                ci_lower = float(t0 + frac * (t1 - t0))
        if hi_idx < len(t_grid) - 1:
            x0, x1 = loglik[hi_idx], loglik[hi_idx + 1]
            t0, t1 = t_grid[hi_idx], t_grid[hi_idx + 1]
            if x1 != x0:
                frac = (thresh - x0) / (x1 - x0)
                ci_upper = float(t0 + frac * (t1 - t0))

    width = max(1e-12, ci_upper - ci_lower)
    var_approx = float((width / (2.0 * 1.96)) ** 2)

    return t_map, ci_lower, ci_upper, logpost_map, var_approx


def fit_t_bootstrap(G: nx.Graph,
                    signal: np.ndarray,
                    t_min: float = 0.01,
                    t_max: float = 1000.0,
                    epsilon: float = 1e-8,
                    grid_size: int = 256,
                    n_bootstrap: int = 200,
                    random_state: Optional[int] = None,
                    prior: Literal["uniform", "log_uniform"] = "log_uniform",
                    _eigenvalues: Optional[np.ndarray] = None,
                    _eigenvectors: Optional[np.ndarray] = None,
                    ) -> Tuple:
    """Estimate diffusion-scale uncertainty by parametric bootstrap.

    Parameters
    ----------
    G : networkx.Graph
        Undirected landscape graph.
    signal : ndarray
        Scalar graph signal in graph-node order.
    t_min : float, default=0.01
        Lower scale-grid bound.
    t_max : float, default=1000.0
        Upper scale-grid bound.
    epsilon : float, default=1e-8
        Eigenvalue offset for likelihood stability.
    grid_size : int, default=256
        Number of candidate scales used in each refit.
    n_bootstrap : int, default=200
        Number of parametric bootstrap signals.
    random_state : int, optional
        Random-number-generator seed.
    prior : {'uniform', 'log_uniform'}, default='log_uniform'
        Prior used for the initial fit and bootstrap refits.
    _eigenvalues : ndarray, optional
        Precomputed normalized-Laplacian eigenvalues.
    _eigenvectors : ndarray, optional
        Precomputed normalized-Laplacian eigenvectors.

    Returns
    -------
    t_map : float
        Initial maximum-a-posteriori scale.
    ci_lower : float
        2.5th percentile of bootstrap scale estimates.
    ci_upper : float
        97.5th percentile of bootstrap scale estimates.
    logpost_map : float
        Initial log posterior at ``t_map``.
    var_approx : float
        Sample variance of bootstrap scale estimates.

    Raises
    ------
    ValueError
        If fewer than ten bootstrap replicates are requested.
    """
    if n_bootstrap < 10:
        raise ValueError("n_bootstrap must be >= 10 for a stable bootstrap estimate.")

    f_hat, eigenvalues, eigenvectors, sigma_squared, mu = _precompute_GMRF_stats_with_evecs(
        G,
        signal,
        _eigenvalues=_eigenvalues,
        _eigenvectors=_eigenvectors,
    )
    n = len(eigenvalues)

    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), grid_size)
    t_map, _, _, logpost_map, _ = _grid_posterior_from_stats(
        f_hat=f_hat,
        eigenvalues=eigenvalues,
        sigma_squared=sigma_squared,
        t_grid=t_grid,
        epsilon=epsilon,
        prior=prior,
    )

    lambda_adjusted = eigenvalues + epsilon
    h_i = np.exp(-t_map * lambda_adjusted)
    scaling_factor = (sigma_squared * n) / np.sum(h_i)
    h_i_scaled = h_i * scaling_factor
    sqrt_h = np.sqrt(h_i_scaled)

    rng = np.random.default_rng(random_state)
    boot_estimates = []
    for _ in range(n_bootstrap):
        z = rng.standard_normal(n)
        f_hat_sample = sqrt_h * z
        signal_sample = eigenvectors @ f_hat_sample + mu
        mu_s = float(np.mean(signal_sample))
        signal_centered = signal_sample - mu_s
        f_hat_centered = eigenvectors.T @ signal_centered
        sigma_s = float(np.var(signal_centered, ddof=1))
        t_hat, _, _, _, _ = _grid_posterior_from_stats(
            f_hat=f_hat_centered,
            eigenvalues=eigenvalues,
            sigma_squared=sigma_s,
            t_grid=t_grid,
            epsilon=epsilon,
            prior=prior,
        )
        boot_estimates.append(t_hat)

    boot_estimates = np.array(boot_estimates, dtype=float)
    ci_lower = float(np.quantile(boot_estimates, 0.025))
    ci_upper = float(np.quantile(boot_estimates, 0.975))
    var_approx = float(np.var(boot_estimates, ddof=1))

    return t_map, ci_lower, ci_upper, logpost_map, var_approx

def _compute_variances(eigenvectors: np.ndarray,
                       eigenvalues: np.ndarray,
                       sigma_squared: float,
                       t: float,
                       epsilon: float = 1e-8) -> tuple:
    """
    Function to compute the variance vector and covariance matrix of a
    GMRF.

    Parameters
    ----------
    eigenvectors : np.ndarray   
        The Graph Laplacian eigenvectors.

    eigenvalues : np.ndarray    
        The Graph Laplacian eigenvalues. 
    
    t : float
        The heat diffusion kernel timestep parameter. 
    
    sigma_squared : float
        The empirical variance in the signal. 
    
    epsilon : float, default = `1e-8`.
        Small float for numerical stability.

    Returns
    -------
    variances_H0 : np.ndarray  
        The variance (the diagonal of the covariance matrix).
    
    Sigma_H0 : np.ndarray
        The covariance matrix. 
    """
    n=len(eigenvalues)
    lambda_adjusted = eigenvalues + epsilon
    h_i = np.exp(-t * lambda_adjusted)
    scaling_factor = (sigma_squared * n) / np.sum(h_i)
    h_i_scaled = h_i * scaling_factor
    Sigma_H0 = eigenvectors @ np.diag(h_i_scaled) @ eigenvectors.T
    
    return Sigma_H0


def _resolve_diffusion_scale_signal(
    landscape: FitnessLandscape,
    fitness_layer: str | None,
) -> np.ndarray:
    """Return a finite scalar signal without changing the active layer."""
    if fitness_layer is None:
        layer_name = landscape.active_layer_name
        if layer_name is None:
            raise ValueError(
                "Diffusion-scale analysis requires an active fitness layer or "
                "an explicit fitness_layer."
            )
        layer = landscape.active_layer
    else:
        layer_name = fitness_layer
        layer = landscape.get_layer(fitness_layer, allow_active_default=False)

    try:
        scalar_values = np.asarray(layer.to_scalar(), dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Fitness layer {layer_name!r} must be scalarizable to numeric values."
        ) from error

    expected_rows = len(landscape.sequences)
    if scalar_values.ndim != 1 or scalar_values.shape[0] != expected_rows:
        actual_shape = scalar_values.shape
        raise ValueError(
            f"Fitness layer {layer_name!r} must provide one scalar per sequence; "
            f"got shape {actual_shape}, expected ({expected_rows},)."
        )

    node_order = list(landscape.graph.nodes())
    if len(node_order) != expected_rows:
        raise ValueError(
            "Diffusion-scale analysis requires one graph node per sequence; "
            f"found {len(node_order)} nodes and {expected_rows} sequences."
        )

    signal = np.asarray(
        [
            scalar_values[landscape.sequence_index_for_node(node)]
            for node in node_order
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"Fitness layer {layer_name!r} contains missing or non-finite values; "
            "diffusion-scale analysis requires a complete finite signal."
        )
    return signal

def compute_ruggedness_diffusion_scale(landscape: FitnessLandscape,
                                       fitness_layer: str | None = None,
                                       t_min: float = 0.01,
                                       t_max: float = 100.0,
                                       epsilon: float = 1e-8,
                                       method: Literal["grid", "profile", "bootstrap", "laplace"] = "grid",
                                       grid_size: int = 512,
                                       prior: Literal["uniform", "log_uniform"] = "log_uniform",
                                       bootstrap_samples: int = 200,
                                       random_state: Optional[int] = None,
                                       _eigenvalues: Optional[np.ndarray] = None,
                                       _eigenvectors: Optional[np.ndarray] = None,
                                       ) -> DiffusionScaleResult:
    """Estimate the heat-kernel diffusion scale of a fitness signal.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape to analyse.
    fitness_layer : str, optional
        Layer to fit without changing the landscape's active view. If omitted,
        use the active layer. The selected layer must yield one finite numeric
        scalar per sequence; missing values are not supported.

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.

    method : {"grid", "profile", "bootstrap", "laplace"}
        Estimation strategy for t. Defaults to "grid".

    grid_size : int
        Number of grid points for grid/profile/bootstrapped estimation.

    prior : {"uniform", "log_uniform"}
        Prior used for grid-based posterior (and bootstrap refits).

    bootstrap_samples : int
        Number of bootstrap samples for method="bootstrap".

    random_state : int, optional
        RNG seed for method="bootstrap".

    _eigenvalues : np.ndarray, optional
        Precomputed eigenvalues of the normalized Laplacian (private override).

    _eigenvectors : np.ndarray, optional
        Precomputed eigenvectors of the normalized Laplacian (private override).
    
    Returns
    -------
    DiffusionScaleResult
        Dictionary containing ``t_map``,
        ``t_lower_confidence_interval``,
        ``t_upper_confidence_interval``, ``t_logposterior_map``, and
        ``variance_approximate``.

    Notes
    -----
    The model is a zero-mean Gaussian Markov random field with covariance
    spectrum proportional to ``exp(-t * lambda)`` and normalized to the
    empirical signal variance. Larger ``t`` concentrates variance in smoother
    graph modes. Interval semantics depend on ``method``: Bayesian credible
    intervals for ``grid`` and ``laplace``, a likelihood-ratio confidence
    interval for ``profile``, and percentile uncertainty for ``bootstrap``.
    """
    # Make sure not directed graph.
    if not isinstance(landscape.graph, nx.Graph):
        raise ValueError(f"Expected `landscape.graph` to be `nx.Graph`, found `{type(landscape.graph)}`")

    signal = _resolve_diffusion_scale_signal(landscape, fitness_layer)

    if method == "laplace":
        t_map, ci_lower, ci_upper, logpost_map, var_approx = fit_t_bayesian_laplace(
            landscape.graph,
            signal,
            t_min=t_min,
            t_max=t_max,
            epsilon=epsilon,
            _eigenvalues=_eigenvalues,
            _eigenvectors=_eigenvectors,
        )
    elif method == "grid":
        t_map, ci_lower, ci_upper, logpost_map, var_approx = fit_t_grid_posterior(
            landscape.graph,
            signal,
            t_min=t_min,
            t_max=t_max,
            epsilon=epsilon,
            grid_size=grid_size,
            prior=prior,
            _eigenvalues=_eigenvalues,
            _eigenvectors=_eigenvectors,
        )
    elif method == "profile":
        t_map, ci_lower, ci_upper, logpost_map, var_approx = fit_t_profile_likelihood(
            landscape.graph,
            signal,
            t_min=t_min,
            t_max=t_max,
            epsilon=epsilon,
            grid_size=grid_size,
            _eigenvalues=_eigenvalues,
            _eigenvectors=_eigenvectors,
        )
    elif method == "bootstrap":
        t_map, ci_lower, ci_upper, logpost_map, var_approx = fit_t_bootstrap(
            landscape.graph,
            signal,
            t_min=t_min,
            t_max=t_max,
            epsilon=epsilon,
            grid_size=grid_size,
            n_bootstrap=bootstrap_samples,
            random_state=random_state,
            prior=prior,
            _eigenvalues=_eigenvalues,
            _eigenvectors=_eigenvectors,
        )
    else:
        raise ValueError(
            "Unknown method for diffusion scale. Use one of: "
            "'grid', 'profile', 'bootstrap', 'laplace'."
        )

    return {
        't_map': float(t_map),
        't_lower_confidence_interval': float(ci_lower),
        't_upper_confidence_interval': float(ci_upper),
        't_logposterior_map': float(logpost_map),
        'variance_approximate': float(var_approx),
        }
    
def _expected_local_global_dirichlet_energy(G: nx.Graph,
                                            sigma: np.ndarray,
                                            mean: np.ndarray,
                                            normalized: bool = True,
                                            weight_key: str = "weight") -> tuple[np.ndarray, float]:
    """
    Function to compute the expected local and global Dirichlet energy
    under N(mean, Sigma).

    Parameters
    ----------
    G : nx.Graph
        The network graph to analyze. 
    
    sigma : np.ndarray
        The covariance matrix. 
    
    mean : np.ndarray
        The mean vector. 
    
    normalized : bool, default=`True`
        Whether to use the normalized Lapalcian. If `True`, computes
        energy for the normalized Laplacian: 
        E = f^T L_norm f = 1/2 sum_ij A_ij (f_i/sqrt(d_i) - f_j/sqrt(d_j))^2
        Otherwise uses the combinatorial Laplacian:
        E = f^T (D-A) f = 1/2 sum_ij w_ij (f_i - f_j)^2
    
    weight_key : str, default=`weight`
        The key that edge weights are stored under.
    
    Returns
    -------
    local_energy : np.ndarray
        The trace component of the Dirichlet energy expectation. 

    global_expectation : np.ndarray
        The global Dirichlet expectation.
    """
    n = G.number_of_nodes()

    # Consistent node order
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}

    # Degree vector and adjacency (in this node order)
    d = np.zeros(n, dtype=float)
    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        i, j = idx[u], idx[v]
        w = float(data.get(weight_key, 1.0))
        A[i, j] = A[j, i] = w
        d[i] += w
        d[j] += w

    if normalized:

        # Transform to g = D^{-1/2} f space
        invsqrt_d = np.zeros_like(d)
        nz = d > 0
        invsqrt_d[nz] = 1.0 / np.sqrt(d[nz])
        T = np.diag(invsqrt_d)
        sigma_prime = T @ sigma @ T
        m_prime = T @ mean
        # For normalized energy use A's weights in the pairwise sum
        W = A
    
    else:

        sigma_prime = sigma
        m_prime = mean
        # combinatorial uses the same W = A with weights
        W = A  

    # Local contributions: 1/2 sum_j w_ij E[(g_i - g_j)^2]
    diag_S = np.diag(sigma_prime)

    # E[(g_i - g_j)^2] = Var + (mean diff)^2
    # Precompute the (mean diff)^2 matrix efficiently
    m_diff2 = (m_prime[:, None] - m_prime[None, :]) ** 2
    
    # And the variance term via broadcasting:
    var_pair = diag_S[:, None] + diag_S[None, :] - 2.0 * sigma_prime

    pair_expect = var_pair + m_diff2

    # Weight by W and sum half to each endpoint
    local_energy = 0.5 * (W * pair_expect).sum(axis=1)

    # Global expected energy:
    # combinatorial: Tr(L Sigma) + m^T L m
    # normalized:    Tr(L_norm Sigma) + m^T L_norm m

    if normalized:
        # L_norm = I - D^{-1/2} A D^{-1/2}
        I = np.eye(n)
        Dmhalf = np.diag(invsqrt_d)
        L = I - (Dmhalf @ A @ Dmhalf)
    
    else:
        # L = D - A
        L = np.diag(d) - A

    expected_global = float(np.trace(L @ sigma) + mean.T @ L @ mean)

    return local_energy, expected_global

def compute_ruggedness_variance_energy(landscape: FitnessLandscape,
                                       t: float = None,
                                       t_min: float = 0.01,
                                       t_max: float = 100.0,
                                       epsilon: float = 1e-8,
                                       normalized: bool = True,
                                       weight_key: str = "weight",
                                       method: Literal["grid", "profile", "bootstrap", "laplace"] = "grid",
                                       grid_size: int = 512,
                                       prior: Literal["uniform", "log_uniform"] = "log_uniform",
                                       bootstrap_samples: int = 200,
                                       random_state: Optional[int] = None,
                                       _eigenvalues: Optional[np.ndarray] = None,
                                       _eigenvectors: Optional[np.ndarray] = None,
                                       ) -> dict:
    """
    Function to compute the expected local and global Dirichlet energy
    under the GMRF diffusion prior defined by the heat diffusion scale.
    
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape whose active scalar fitness signal defines the GMRF.
    t : float
        The heat diffusion scale. 
    
    t_min : float, default=0.01
        The minimum diffusion scale value to use during fitting. 
    
    t_max : float, default=100
        The maximum diffusion scale value to use during fitting. 
    
    epsilon : float, default=1e-08
        Small value added for numerical stability.
        
    normalized : bool, default=`True`
        Whether to use the normalized Laplacian. 
    
    weight_key : str, default=`weight`
        The key that edge weights are stored under.

    method : {"grid", "profile", "bootstrap", "laplace"}
        Estimation strategy for t when t is None.

    grid_size : int
        Number of grid points for grid/profile/bootstrapped estimation.

    prior : {"uniform", "log_uniform"}
        Prior used for grid-based posterior (and bootstrap refits).

    bootstrap_samples : int
        Number of bootstrap samples for method="bootstrap".

    random_state : int, optional
        RNG seed for method="bootstrap".

    _eigenvalues : ndarray, optional
        Precomputed normalized-Laplacian eigenvalues.
    _eigenvectors : ndarray, optional
        Precomputed normalized-Laplacian eigenvectors.
    
    Returns
    -------
    Dict
        Dictionary of results with entries:
        - covariance_matrix : covariance array computed from t. 
        - expected_local_energy : the trace term contribution. 
        - expected_global_energy : the total expected Dirichlet energy.
        - t_used : the diffusion scale used to define the covariance
        matrix. 
        If the signal is centered on 0, the mean contribution will be
        the 0 vector and ther expected global energy will be the
        expected local energy.
    """
    G = landscape.graph
    node_order = list(G.nodes())
    signal = landscape.get_node_signal(node_order)

    # Estimate t if not given
    if t is None:
        fit = compute_ruggedness_diffusion_scale(landscape,
                                                 t_min=t_min,
                                                 t_max=t_max,
                                                 epsilon=epsilon,
                                                 method=method,
                                                 grid_size=grid_size,
                                                 prior=prior,
                                                 bootstrap_samples=bootstrap_samples,
                                                 random_state=random_state,
                                                 _eigenvalues=_eigenvalues,
                                                 _eigenvectors=_eigenvectors)
        t_value = float(fit['t_map'])
    else:
        t_value = float(t)

    if (_eigenvalues is None) != (_eigenvectors is None):
        raise ValueError("Provide both _eigenvalues and _eigenvectors or neither.")
    if _eigenvalues is not None:
        eigenvalues = np.asarray(_eigenvalues, dtype=float)
        eigenvectors = np.asarray(_eigenvectors, dtype=float)
    else:
        eigenvalues, eigenvectors = eigenmode_decomposition(
            G,
            matrix='norm_laplacian',
            weight_key=None,
        )

    # Centered mean used in the likelihood is zero-mean; keep both:
    mu_emp = float(np.mean(signal))
    mean_centered = signal - mu_emp
    sigma_squared = np.var(mean_centered, ddof=1)

    Sigma = _compute_variances(eigenvectors=eigenvectors,
                               eigenvalues=eigenvalues,
                               sigma_squared=sigma_squared,
                               t=t_value,
                               epsilon=epsilon)

    local_E, global_E = _expected_local_global_dirichlet_energy(
        G=G,
        sigma=Sigma,
        mean=np.zeros_like(signal, dtype=float), # Signal is centered on 0!
        normalized=normalized,
        weight_key=weight_key,
    )

    return {
        'covariance_matrix': Sigma,
        'expected_local_energy': local_E,
        'expected_local_energy_by_node': {
            node: float(local_E[index]) for index, node in enumerate(node_order)
        },
        'expected_global_energy': float(global_E),
        't_used': t_value,
        'node_order': node_order,
    }
