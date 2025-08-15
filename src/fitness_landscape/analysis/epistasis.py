import numpy as np
from typing import List, Tuple, Dict, Literal
from ..core.landscape import FitnessLandscape
from ..transforms.walsh_hadamard import walsh_coefficients
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from itertools import combinations
from itertools import combinations, product



def calculate_epistasis_walsh(landscape: FitnessLandscape,
                               order: int,
                               **kwargs) -> Dict:
    """
    Function to measure epistasis using the Walsh-Hadamard
    transformation. Supports binary and higher dimensional state
    spaces. 

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to transform.
    
    order : int
        The order of interactions to test up to. 
    
    Returns
    -------
    resutls : dict
        Dictionary of results on the transformation.
    """
# Check if sequences are binary
    is_binary = True
    for seq in landscape.sequences:
        if not set(seq.sequence).issubset({0, 1}):
            is_binary = False
            break
    
    if is_binary:
        # Use standard Walsh transform for binary sequences
        coeffs = walsh_coefficients(landscape,
                                    order=order)
        
        # Organize coefficients by order
        result = {
            'coefficients': coeffs,
            'by_order': {}
        }
        
        for term, value in coeffs.items():
            if term == 'intercept':
                order_key = 0
            else:
                order_key = len(term.split(','))
            
            if order_key not in result['by_order']:
                result['by_order'][order_key] = {}
            
            result['by_order'][order_key][term] = value
        
        # Calculate the proportion of variance explained by each order
        squared_coeffs_by_order = {}
        total_variance = 0
        for term, value in coeffs.items():
            if term != 'intercept':
                squared_value = value**2
                total_variance += squared_value
                order_key = len(term.split(','))
                squared_coeffs_by_order.setdefault(order_key, 0)
                squared_coeffs_by_order[order_key] += squared_value
        
        if total_variance > 0:
            variation_explained = {
                order: s_sq / total_variance 
                for order, s_sq in squared_coeffs_by_order.items()
            }
        else:
            variation_explained = {order: 0.0 for order in squared_coeffs_by_order}

        result['variance_explained'] = variation_explained

        # Calculate summary statistics
        result['statistics'] = _calculate_epistasis_statistics(coeffs)
        
        return result
    

def get_epistasis_matrix(landscape: FitnessLandscape) -> np.ndarray:
    """
    Computes an nxn matrix for pairwise variance from the WHT.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.

    Returns
    -------
    epistasis_matrix : np.ndarray
        An nxn matrix where values represent the variance (squared
        Walsh coefficient) that each pair of mutations contributes to
        the total fitness signal.
    """
    n = len(landscape.sequences[0])
    epistasis_matrix = np.zeros((n, n), dtype=float)

    # Calculate second-order epistasis using the Walsh-Hadamard transform
    results = calculate_epistasis_walsh(landscape, order=2)
    
    if 2 in results['by_order']:
        for term, value in results['by_order'][2].items():
            i_str, j_str = term.split(',')
            i, j = int(i_str), int(j_str)
            epistasis_matrix[i, j] = value**2
            epistasis_matrix[j, i] = value**2

    return epistasis_matrix

def calculate_epistasis_regression(landscape: FitnessLandscape,
                                    order: int,
                                    regularization: Literal['l1', 'l2', 'elastic_net'] = None,
                                    alpha: float = 1.0,
                                    **kwargs) -> Dict:
    """
    Function to measure epistasis with linear modelling using one-hot encoding.
    """
    # Sequences (M x N) as 0/1
    X01 = np.vstack([s.to_array().astype(int) for s in landscape.sequences])
    y = landscape.get_signal().astype(float)

    # Center the response so the intercept is the mean
    y_mean = float(np.mean(y))
    y_centered = y - y_mean

    # Build orthogonal (effect-coded) design up to `order`
    X, feature_names, index_by_order = _build_effect_design(landscape.sequences, order)

    # Choose linear model (no regularization by default)
    if regularization is None:
        model = LinearRegression()
    elif regularization == 'l1':
        model = Lasso(alpha=alpha)
    elif regularization == 'l2':
        model = Ridge(alpha=alpha)
    elif regularization == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=kwargs.get('l1_ratio', 0.5))
    else:
        raise ValueError(f"Unsupported regularization: {regularization}")

    # Fit to the centered response
    model.fit(X, y_centered)

    # Coefficients: intercept is the empirical mean
    coeffs = {'intercept': y_mean}
    for i, name in enumerate(feature_names):
        coeffs[name] = float(model.coef_[i])

    # Group by order from our index map
    by_order: Dict[int, Dict[str, float]] = {}
    for r, idxs in index_by_order.items():
        if r == 0:
            by_order.setdefault(0, {})['intercept'] = y_mean
            continue
        by_order.setdefault(r, {})
        for j in idxs:
            by_order[r][feature_names[j]] = float(model.coef_[j])

    # Compute R^2 using the full prediction with intercept
    y_hat = X @ model.coef_ + y_mean
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    result = {
        'coefficients': coeffs,
        'by_order': by_order,
        'model': {
            'r2_score': r2,
            'model_type': model.__class__.__name__,
        },
        'statistics': _calculate_epistasis_statistics(coeffs),
    }
    return result


def _build_effect_design(sequences,
                         order: int) -> Tuple[np.ndarray, List[str], Dict[int, List[int]]]:
    """
    Build an orthogonal design on the full 2^N cube using effect coding.
    """
    # Convert sequences to (M x N) 0/1
    if hasattr(sequences[0], "to_array"):
        X01 = np.vstack([s.to_array().astype(int) for s in sequences])
    else:
        X01 = np.asarray(sequences, dtype=int)
    M, N = X01.shape

    # Effect coding
    Z = 1 - 2 * X01  # 0 -> +1, 1 -> -1

    cols = []
    names = []
    index_by_order: Dict[int, List[int]] = {}
    next_col = 0

    # Order 1 (main effects)
    for i in range(N):
        cols.append(Z[:, i][:, None].astype(float))
        names.append(f"pos{i}")
    index_by_order[1] = list(range(next_col, next_col + N))
    next_col += N

    # Higher orders: products of z-columns
    for r in range(2, order + 1):
        start = next_col
        for idxs in combinations(range(N), r):
            col = np.prod(Z[:, idxs], axis=1, dtype=float)[:, None]
            cols.append(col)
            names.append("*".join(f"pos{i}" for i in idxs))
            next_col += 1
        if next_col > start:
            index_by_order[r] = list(range(start, next_col))

    X = np.hstack(cols) if cols else np.zeros((M, 0), dtype=float)
    return X, names, index_by_order


def calculate_epistasis_ensemble(landscape: FitnessLandscape,
                                 order: int,**kwargs) -> Dict:
    """
    Function to compute the background average epistasis of a system.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    order : int
        The order of epistasis to test up to. 
    
    Returns
    -------
    results : Dict
        The results dictionary. 
    """
    # Extract sequences and fitness values
    sequences = [seq.to_array() for seq in landscape.sequences]
    fitness_values = [landscape.get_fitness(seq) for seq in landscape.sequences]
    
    # Calculate mean fitness
    mean_fitness = np.mean(fitness_values)
    
    # Calculate epistasis for each combination of positions
    result = {
        'coefficients': {},
        'by_order': {}
    }
    
    # Add zeroth order (mean)
    result['coefficients']['intercept'] = mean_fitness
    result['by_order'][0] = {'intercept': mean_fitness}
    
    # Calculate first-order effects (main effects)
    seq_length = len(sequences[0])
    result['by_order'][1] = {}
    
    for pos in range(seq_length):
        # Group sequences by value at this position
        by_value = {}
        for seq, fitness in zip(sequences, fitness_values):
            val = seq[pos]
            if val not in by_value:
                by_value[val] = []
            by_value[val].append(fitness)
        
        # Calculate average effect of each value
        for val, fitnesses in by_value.items():
            term = f"{pos}:{val}"
            effect = np.mean(fitnesses) - mean_fitness
            result['coefficients'][term] = effect
            result['by_order'][1][term] = effect
    
    # Calculate higher-order effects if requested
    if order >= 2:
        for o in range(2, order + 1):
            result['by_order'][o] = {}
            
            # Generate all combinations of o positions
            for pos_combo in combinations(range(seq_length), o):
                # Group sequences by values at these positions
                by_values = {}
                for seq, fitness in zip(sequences, fitness_values):
                    vals = tuple(seq[p] for p in pos_combo)
                    if vals not in by_values:
                        by_values[vals] = []
                    by_values[vals].append(fitness)
                
                # Calculate epistasis for each combination of values
                for vals, fitnesses in by_values.items():
                    # Create term string
                    term_parts = [f"{pos}:{val}" for pos, val in zip(pos_combo, vals)]
                    term = ",".join(term_parts)
                    
                    # Calculate expected fitness based on lower-order terms
                    expected = mean_fitness
                    
                    # Add first-order effects
                    for i, (pos, val) in enumerate(zip(pos_combo, vals)):
                        first_order_term = f"{pos}:{val}"
                        if first_order_term in result['coefficients']:
                            expected += result['coefficients'][first_order_term]
                    
                    # Add second-order effects if calculating third or higher order
                    if o >= 3:
                        for i, j in combinations(range(o), 2):
                            pos_i, pos_j = pos_combo[i], pos_combo[j]
                            val_i, val_j = vals[i], vals[j]
                            second_order_term = f"{pos_i}:{val_i},{pos_j}:{val_j}"
                            if second_order_term in result['coefficients']:
                                expected += result['coefficients'][second_order_term]
                    
                    # Calculate epistasis as deviation from expected
                    epistasis = np.mean(fitnesses) - expected
                    result['coefficients'][term] = epistasis
                    result['by_order'][o][term] = epistasis
    
    # Calculate summary statistics
    result['statistics'] = _calculate_epistasis_statistics(result['coefficients'])
    
    return result


def calculate_epistasis_reference_free(landscape: FitnessLandscape,
                                       order: int,
                                       **kwargs) -> Dict:
    """
    Function to calculate the referece-free epistasis (i.e., mutational
    effects are measured relative to the global population and not a
    reference sequence).

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    order : int
        The order of interaction to test up to. 
    
    Returns
    -------
    results : Dict
        Dictionary of results. 
    """
    sequences = np.array([seq.to_array() for seq in landscape.sequences])
    fitness_values = np.array([landscape.get_fitness(seq) for seq in landscape.sequences])
    
    # Get the unique alleles in the landscape
    alphabet = sorted(list(set(allele for seq in sequences for allele in seq)))
    seq_length = sequences.shape[1]
    
    results = {'coefficients': {}, 'by_order': {}}

    # Global mean fitness
    global_mean = np.mean(fitness_values)
    results['coefficients']['intercept'] = global_mean
    results['by_order'][0] = {'intercept': global_mean}
    
    # Calculate effects for orders 1 onwards
    for o in range(1, order + 1):
        results['by_order'][o] = {}
        
        # Iterate over all combinations of positions for the current order
        for pos_combo in combinations(range(seq_length), o):
            
            # Iterate over all combinations of alleles for those positions.
            for allele_combo in product(alphabet, repeat=o):
                
                # Find all sequences that match this specific combination of alleles at these positions.
                mask = np.all(sequences[:, pos_combo] == allele_combo, axis=1)
                
                if not np.any(mask):
                    continue #
                
                # Observed fitness is the average fitness of the matching sequences.
                observed_fitness = np.mean(fitness_values[mask])
                
                # Recursive calculation of expected fitness.
                expected_fitness = results['coefficients']['intercept']
                
                # Sum all lower-order effects.
                for k in range(1, o):

                    # Iterate through all subsets of the current interaction.
                    for lower_order_pos_indices in combinations(range(o), k):
                        
                        lower_order_pos = tuple(pos_combo[i] for i in lower_order_pos_indices)
                        lower_order_alleles = tuple(allele_combo[i] for i in lower_order_pos_indices)
                        
                        # Construct the term name and look up its pre-calculated coefficient
                        term_parts = [f"{p}:{a}" for p, a in zip(lower_order_pos, lower_order_alleles)]
                        lower_order_term = ",".join(term_parts)
                        
                        expected_fitness += results['coefficients'].get(lower_order_term, 0)
                
                # The nth-order epistasis is the deviation from the expected fitness
                epistasis = observed_fitness - expected_fitness
                
                current_term_parts = [f"{p}:{a}" for p, a in zip(pos_combo, allele_combo)]
                current_term = ",".join(current_term_parts)
                results['coefficients'][current_term] = epistasis
                results['by_order'][o][current_term] = epistasis

    results['statistics'] = _calculate_epistasis_statistics(results['coefficients'])
    return results


def _calculate_epistasis_statistics(coefficients: Dict) -> Dict:
    """
    Function to calcaulte epistasis summary statistics. 

    Parameters
    ----------
    coefficients : Dict
        The coefficients dictionary output from an epistasis
        decomposition function. 
    
    Returns
    -------
    Dict
        Dictionary of summary statistics on the epistatic coefficients.
    """
    # Remove intercept for statistics
    coef_values = [v for k, v in coefficients.items() if k != 'intercept']
    
    if not coef_values:
        return {
            'mean': 0,
            'std': 0,
            'max': 0,
            'min': 0,
            'abs_mean': 0
        }
    
    return {
        'mean': np.mean(coef_values),
        'std': np.std(coef_values),
        'max': np.max(coef_values),
        'min': np.min(coef_values),
        'abs_mean': np.mean(np.abs(coef_values))
    }
