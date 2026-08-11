import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from ..core.landscape import FitnessLandscape
from ..core.fitness import NumericFitness
from ..utils import check_full_hamming
from ..transforms.walsh_hadamard import walsh_transform
from ..transforms.eigenmode import eigenmode_decomposition

def _get_layer_matrix(landscape: FitnessLandscape,
                      layer_names: List[str],
                      *,
                      numeric_agg: callable = np.mean,
                      categorical_rank_maps: Optional[Dict[str, Dict[str, int]]] = None
                     ) -> np.ndarray:
    """
    Function to collate fitness layer scalar values into an
    (n_nodes x N_layers) shaped array in the node order of the
    `landscape.graph` attribute.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to collate fitness layers from.

    layer_names : List[str]
        The fitness layer string keys to collate.

    numeric_agg : callable, default=`np.mean`
        The numeric value aggregation function.

    Returns
    -------
    X : np.ndarray
        The (n_nodes, N_layers) shaped array of nodes and corresponding
        fitness values over the N fitness layers.
    """
    G = landscape.graph
    nodes = list(G.nodes())
    seq_to_idx = {tuple(seq.to_array()): i for i, seq in enumerate(landscape.sequences)}
    node_to_seq_idx = []
    for _, data in G.nodes(data=True):
        seq = data.get('sequence', None)
        if seq is None:
            raise ValueError("Graph nodes must carry a 'sequence' attribute.")
        idx = seq_to_idx.get(tuple(seq.to_array()))
        if idx is None:
            raise ValueError("A node's sequence not found among `landscape.sequences`.")
        node_to_seq_idx.append(idx)

    N = len(layer_names)
    X = np.zeros((len(nodes), N), dtype=float)
    for j, lname in enumerate(layer_names):
        layer = landscape.get_layer(lname)
        if layer.dtype == 'numeric':
            vals = layer.to_scalar(aggregate_func=numeric_agg)
            X[:, j] = vals[node_to_seq_idx]
        else:
            raise ValueError(f'Expected only `numeric` fitness layers suppoorted. Found {layer.dtype}.')
    return X

def _walsh_coeffs_for_layer(landscape: FitnessLandscape,
                            layer_name: str) -> np.ndarray:
    """
    Helper function to perform WHT on a currently non-active fitness
    layer.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.

    layer_name : str
        The layer key name.

    Returns
    -------
    np.ndarray
        The Walsh coefficients from the WHT performed on the
        `layer_name` matched fitness scalars.
    """
    prev = landscape._active_view_name
    try:
        lyr = landscape.get_layer(layer_name)
        landscape.view(lyr.name)
        coef = walsh_transform(landscape, order=None)
        return np.asarray(coef, dtype=float)
    finally:
        if prev is not None and prev in landscape.fitness_layers:
            landscape.view(prev)

def _default_gft_bands(evals: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Helper function to define default GFT band binning.

    Parameters
    ----------
    evals : np.ndarray
        The GFT eigenvalues.

    Returns
    -------
    Dict
        The binned eigenvalues according to thirds.
    """
    q = np.quantile(evals, [0.33, 0.66])

    return {
        'low':  (evals <= q[0]),
        'mid':  (evals > q[0]) & (evals <= q[1]),
        'high': (evals > q[1]),
    }

def cross_spectral_coherence(landscape: FitnessLandscape,
                             layer_names: List[str],
                             *,
                             basis: Literal['auto','walsh','gft'] = 'auto',
                             n_eigs: Optional[int] = None,
                             walsh_aggregate: Literal['order','none'] = 'order',
                             gft_bands: Optional[Dict[str, np.ndarray]] = None,
                             random_state: Optional[int] = None,
                             return_phase: bool = False) -> Dict:
    """
    Function to compute the squared magnitude cross spectral coherence
    among N fitness layers in either the Hadamard or Laplacian
    eigenvector basis. Cross spectral coherence is defined for only
    numeric fitness layers that can be converted to floating point
    scalar values.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.

    layer_names : List[str]
        The fitness layers to using cross spectral coherence.

    basis : str, default=`auto`
        The basis to use for common spectral projection. If `auto`,
        the `walsh` basis is used preferentially if the landscape
        graph is a fully connected hamming graph, otherwise the `gft`
        Laplacian eigenvector basis is used.

    n_eigs : int, default=`None`
        The number of eigenvectors to use.

    walsh_aggregrate : str, default=`order`
        The aggregation method for Walsh coefficients. If `order`,
        cross spectral coherence is returned with respect to the order
        of epistasis.

    gft_bands : Dict[str, np.ndarray], default=`None`
        The GFT spectral bands to and classifications to use in
        aggregating coherence results if the Laplacian eigenvector
        basis is used.

    random_state : int, default=`None`
        The random state seed for deterministic results.

    return_phase : bool, default=`False`
        Boolean to return the phase as part of the coherence
        comparison.

    Returns
    -------
    results : Dict
        The spectral coherence results dictionary with entries:
        - evals : the eigenvalues.
        - coherence : the cross spectral coherence for the indexed
        eigenmode.
        - agg : the cross spectral coherence results aggregated.
    """
    rng = np.random.default_rng(random_state)
    N = len(layer_names)

    use_walsh = False
    if basis == 'walsh':
        use_walsh = True
    elif basis == 'gft':
        use_walsh = False
    else:

        res = check_full_hamming(landscape, check_graph=False)
        use_walsh = bool(res.is_full_hamming)

    X = _get_layer_matrix(landscape, layer_names)

    #Transform to spectral coefficients (F: modes x N).
    if use_walsh:
        # One walsh_transform call per layer (uses your implementation)
        F_cols = [ _walsh_coeffs_for_layer(landscape, lname) for lname in layer_names ]
        F = np.vstack(F_cols).T  # (2^L x N)
        evals = None

        # per-order aggregation (bitcount masks) if requested
        if walsh_aggregate == 'order':
            L = int(np.log2(F.shape[0]))
            orders = np.array([int(i).bit_count() for i in range(1 << L)], dtype=int)
            mode_masks = [orders == r for r in range(L+1)]
        else:
            mode_masks = None
    else:
        # Use Laplacian of norm Laplacian?
        evals, U = eigenmode_decomposition(landscape, matrix='laplacian', k=n_eigs)

        # Project all layers with the same basis (m x N)
        F = U.T @ X
        mode_masks = (_default_gft_bands(evals) if gft_bands is None else gft_bands)

    # Per-mode coherence and phase.
    K = F.shape[0]
    coherence_per_mode: List[np.ndarray] = []
    phase_per_mode: Optional[List[np.ndarray]] = [] if return_phase else None

    for k in range(K):
        fk = F[k, :] # (N,)
        power = np.abs(fk)**2
        denom = np.outer(power, power)
        Sab = np.outer(fk, np.conj(fk)) # (N x N)
        coh = np.zeros((N, N), dtype=float)
        # (1e-12)^2 as a coherent power^2 floor
        tiny = 1e-24
        nz = denom > tiny
        coh[nz] = (np.abs(Sab[nz])**2) / denom[nz]
        np.fill_diagonal(coh, 1.0)
        coherence_per_mode.append(coh)

        if return_phase:
            ph = np.angle(Sab)
            np.fill_diagonal(ph, 0.0)
            phase_per_mode.append(ph)

    results: Dict[str, Any] = {'evals': evals, 'coherence': coherence_per_mode}
    if return_phase:
        results['phase'] = phase_per_mode

    # aggregate by orders/bands
    if mode_masks is not None:
        agg: Dict[str, np.ndarray] = {}
        named_masks = (enumerate(mode_masks) if isinstance(mode_masks, list) else mode_masks.items())
        for name, mask in named_masks:
            band_name = f'order_{name}' if isinstance(mode_masks, list) else str(name)
            idxs = np.where(np.asarray(mask, dtype=bool))[0]
            if len(idxs) == 0:
                agg[band_name] = np.full((N, N), np.nan)
                continue
            M = np.zeros((N, N), dtype=float)
            for k in idxs:
                M += coherence_per_mode[k]
            agg[band_name] = M / len(idxs)
        results['bands'] = agg

    return results