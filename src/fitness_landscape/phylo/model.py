import numpy as np

def lower_triangle_to_symmetric(tril_values: np.ndarray,
                                n: int = 20) -> np.ndarray:
    """
    Function to convert the lower triangle of a PAML formatted
    replacement matrix into a symmetric matrix with 0 diags.

    Parameters
    ----------
    tril_values : np.ndarray
        The lower triangle values. 
    
    n : int, default=20
        The size of the square matrix in either axis.
    
    Returns
    -------
    S : np.ndarray
        The symmetric and 0 diagn matrix.
    """
    S = np.zeros((n, n), float)
    tril_idx = np.tril_indices(n, k=-1)
    S[tril_idx] = tril_values
    S = S + S.T
    return S

def as_rate_matrix_from_exchangeabilities(S: np.ndarray,
                                          pi: np.ndarray) -> np.ndarray:
    """
    Function to construct rate matrix from symmetric exchangeabilities.

    Parameters
    ----------
    S : np.ndarray
        The symmetric 0 diag replacement matrix.
    
    pi : np.ndarray
        The equilibrium frequencies.
    
    Returns
    -------
    Q : np.ndarray
        The instantaneous rate matrix.
    """
    Q = S * pi[None, :]              # Q_ij = S_ij * pi_j
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q

def rescale_to_rate1(Q: np.ndarray,
                     pi: np.ndarray) -> np.ndarry:
    """
    Function to rescale rate matrix Q so that -sum_i pi_i Q_ii = 1.
    
    Parameters
    ----------
    Q : np.ndarray
        The rate matrix. 
    
    pi : np.ndarry
        The equilibrium frequencies.
    
    Returns
    -------
    np.ndarray
        The rescaled rate matrix.
    """
    mu = -(pi * np.diag(Q)).sum()
    return Q / mu, mu

def normalise_Q(Q: np.ndarray,
                pi: np.ndarray) -> np.ndarray:
    """
    Function to normalise a rate matrix.

    Parameters
    ----------
    Q : np.ndarray
        The rate matrix. 
    
    pi : np.ndarry
        The equilibrium frequencies.
    
    Returns
    -------
    Q_norm : np.ndarray
        The normalisd rate matrix.
    """
    Q = np.array(Q, float)
    # small numeric cleanup to enforce row sums.
    np.fill_diagonal(Q, -Q[:, :].sum(axis=1) + np.diag(Q))
    Q_norm, mu = rescale_to_rate1(Q, pi)
    return Q_norm

def build_Q(tril_values: np.ndarray,
            pi: np.ndarray) -> np.ndarray:
    """
    Function to build a symmetric, normalised rate matrix from the
    lower triangle (flat) array. 

    Parameters
    -----------
    tril_values : np.ndarray
        The lower triangle values. 
    
    pi : np.ndarry
        The equilibrium frequencies.

    Returns
    -------
    Q_norm : np.ndarray
        The normalisd rate matrix.
    """
    S = lower_triangle_to_symmetric(tril_values)
    Q = as_rate_matrix_from_exchangeabilities(S, pi)
    Q_norm, mu = rescale_to_rate1(Q, pi)
    return Q_norm