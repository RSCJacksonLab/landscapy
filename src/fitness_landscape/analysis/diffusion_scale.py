import networkx as nx
import numpy as np
from typing import Tuple, Union
from ..transforms.eigenmode import eigenmode_decomposition
from ..transforms.graph_fourier import graph_fourier_transform

def _precompute_GMRF_stats(G: nx.Graph,
                           signal: np.ndarray) -> Tuple:
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
    eigenvalues, eigenvectors = eigenmode_decomposition(G,
                                                        matrix = 'norm_laplacian')

    mu = np.mean(signal)
    
    # Centre signal on average
    signal_centered = signal - mu
    
    # GFT on norm Laplacian
    f_hat = eigenvectors.T @ signal_centered
    
    sigma_squared = np.var(signal_centered, ddof=1)
    return f_hat, eigenvalues, sigma_squared

def compute_log_likelihood_H0(f_hat: np.ndarray,
                              eigenvalues: np.ndarray,
                              t: float,
                              sigma_squared: float,
                              epsilon: float = 1e-8) -> tuple:
    """
    Function to compute the log likelihood under a GMRF landscape
    model. 

    Arguments:
    ----------
    f_hat : np.array    
        The graph signal transformed into the Fourier basis. 
    
    eigenvalues : np.ndarray    
        The Graph Laplacian eigenvalues. 
    
    t : float
        The heat diffusion kernel timestep parameter. 
    
    sigma_squared : float
        The empirical variance in the signal. 
    
    epsilon : float, default = `1e-8`.
        Small float for numerical stability.
    
    Returns:
    --------
    log_likelihood : float
        The log likelihood. 
    
    log_det : float
        The log determinant of the Gaussian. 
    
    quadratic form : float
        The qudratic form of the Gaussian. 
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


def _neg_log_posterior(t: float) -> float:
    """
    Helper function to return negative log posterior.

    Parameters
    ----------
    t : float
        The timestep of the heat kernel. 
    
    Returns
    -------
    float
        The negative log posterior probability.
    """
    if t < t_min or t > t_max:
        return np.inf

    log_pri = 0.0
    
    ll = compute_log_likelihood_H0(
        f_hat=f_hat,
        eigenvalues=eigenvalues,
        t=t,
        sigma_squared=sigma_squared,
        epsilon=epsilon
    )[0]

    return -(ll + log_pri)


def fit_t_bayesian_laplace(G: nx.Graph,
                           signal: float,
                           t_min: float = 0.01,
                           t_max: float = 1000.0,
                           epsilon: float = 1e-8) -> Tuple:
    """
    Function to estimate the Posterior probability distribution of t
    using the Laplace approximation.

    Parameters
    ----------
    G : nx.Graph
        The fitness landscape graph. 

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.
    
    verbose : bool, default=`False`
        Boolean for verbose output. 

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

    f_hat, eigenvalues, sigma_squared = _precompute_GMRF_stats(G, signal)
    
    result = minimize_scalar(
        _neg_log_posterior,
        bounds=(t_min, t_max),
        method='bounded',
        options={'maxiter': 200}
    )

    t_map = result.x

    h = 1e-4 * max(1.0, abs(t_map))  # step size ~ 1e-4 * scale_of_t
    t_minus = max(t_min, t_map - h)
    t_plus  = min(t_max, t_map + h)

    f0 = neg_log_posterior(t_map)
    f_minus = neg_log_posterior(t_minus)
    f_plus  = neg_log_posterior(t_plus)

    second_deriv = (f_plus - 2.0*f0 + f_minus) / ( (t_plus - t_minus)/2.0 )**2

    if second_deriv <= 0:
        # Derivative negative, default to small value.
        second_deriv = 1e-12

    var_approx = 1.0 / second_deriv
    std_approx = np.sqrt(var_approx)

    # 95% credible interval (approx Gaussian)
    ci_lower = t_map - 1.96 * std_approx
    ci_upper = t_map + 1.96 * std_approx

    # Clip to [t_min, t_max]
    ci_lower = max(ci_lower, t_min)
    ci_upper = min(ci_upper, t_max)

    logpost_map = -(f0) 

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

def compute_ruggedness_diffusion_scale(landscape: FitnessLandscape,
                                       fitness_layer: str = None,
                                       t_min: float = 0.01,
                                       t_max: float = 100.0,
                                       epsilon: float = 1e-8) -> float:
    """
    Function to compute the diffusion scale (T_map) of a single fitness
    landscape.

    Parameters
    ----------
    lanscape : FitnessLandscape
        The fitness landscape to analyze. Default behabviour will
        measure only the current active fitness layer.

    t_min : float
        The prior lower bound on t. 
    
    t_max : float
        The prior upper bound on t. 
    
    epsilon : float
        Small float for numerical stability.
    
    Returns
    -------
    Dict
        The results dictionary with
        - t_map : the maximum a posteriori diffusion scale.
        - t_lower_confidence_interval : the lower confidence bound on
        the diffusion scale.
        - t_upper_confidence_interval : the upper confidence bound on
        the diffusion scale.
        - t_logposterior_map : the log posterior likelihood of the 
        maximum a posteriori diffusion scale value.
        - var_approx : the approximated variance in the signal.
    """
    # Use current active fitness layer.
    if fitness_layer is None:
        signal = landscape.get_signal()
    # View a key valued fitness layer instead.
    else:
        _ = fitness_landscape.view(fitness_layer)
        signal = landscape.get_signal()
    
    # Make sure not directed graph.
    if not isinstance(landscape, nx.Graph):
        raise ValueError(f"Expected `landscape.graph` to be `nx.Graph`, found `{type(landscape.graph)}`")

    t_map, ci_lower, ci_upper, logpost_map, var_approx = fit_t_bayesian_laplace(landscape.graph,
                                                                                signal,
                                                                                t_min=t_min,
                                                                                t_max=t_max,
                                                                                epsilon=epsilon)

    return {
        't_map': t_map,
        't_lower_confidence_interval': ci_lower,
        't_upper_confidence_interval': ci_upper,
        't_logposterior_map': logpost_map,
        'variance_approximate': var_approx,
        }
    

def compute_ruggedness_variance_energy(landscape: FitnessLandscape,
                                       t: float = None, 
                                       **kwargs):
    """
    Function to compute expectations on the Dirichlet energy according
    to a diffusion process. 

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    t : float, default=`None`
        The diffusion scale parameter. If `None`, the maximum a
        posteriori value is computed and used. 
    
    **kwargs
        Keyword arguments used in fitting the t_map.
    
    Returns
    -------
    Dict 
        Dictionary of results with entriesL
        - covariance_matrix : the covariance array
        global_dirichlet_energy_expectation 
        
    """

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
        Sigma_prime = T @ Sigma @ T
        m_prime = T @ mean
        # For normalized energy use A's weights in the pairwise sum
        W = A
    
    else:

        Sigma_prime = Sigma
        m_prime = mean
        # combinatorial uses the same W = A with weights
        W = A  

    # Local contributions: 1/2 sum_j w_ij E[(g_i - g_j)^2]
    diag_S = np.diag(Sigma_prime)

    # E[(g_i - g_j)^2] = Var + (mean diff)^2
    # Precompute the (mean diff)^2 matrix efficiently
    m_diff2 = (m_prime[:, None] - m_prime[None, :]) ** 2
    
    # And the variance term via broadcasting:
    var_pair = diag_S[:, None] + diag_S[None, :] - 2.0 * Sigma_prime

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

    expected_global = float(np.trace(L @ Sigma) + mean.T @ L @ mean)

    return local_energy, expected_global

def compute_ruggedness_variance_energy(landscape: FitnessLandscape,
                                       t: float = None,
                                       t_min: float = 0.01,
                                       t_max: float = 100.0,
                                       epsilon: float = 1e-8,
                                       normalized: bool = True,
                                       weight_key: str = "weight") -> dict:
    """
    Function to compute the expected local and global Dirichlet energy
    under the GMRF diffusion prior defined by the heat diffusion scale.
    
    
    Parameters
    ----------
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
    
    weight_key: str, default=`weight`
        The key that edge weights are stored under.
    
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
    signal = landscape.get_signal() 

    # Estimate t if not given
    if t is None:
        fit = compute_ruggedness_diffusion_scale(landscape,
                                                 t_min=t_min,
                                                 t_max=t_max,
                                                 epsilon=epsilon)
        t_value = float(fit['t_map'])
    else:
        t_value = float(t)

    eigenvalues, eigenvectors = eigenmode_decomposition(G, matrix='norm_laplacian')

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
        Sigma=Sigma,
        mean=np.zeros_like(signal, dtype=float), # Signal is centered on 0!
        normalized=normalized,
        weight_key=weight_key,
    )

    return {
        'covariance_matrix': Sigma,
        'expected_local_energy': local_E,
        'expected_global_energy': float(global_E),
        't_used': t_value,
    }