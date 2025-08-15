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
    sequences = np.array([seq.to_array() for seq in landscape.sequences])
    fitness_values = np.array([landscape.get_fitness(seq) for seq in landscape.sequences])
    
    # Determine the alphabet from the landscape's sequences
    alphabet = sorted(list(set(allele for seq in sequences for allele in seq)))
    
    # Create the design matrix and feature names
    X, feature_names = _create_design_matrix_one_hot(sequences, order, alphabet)
    
    # Choose and fit the regression model
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
        
    model.fit(X, fitness_values)
    
    # Extract coefficients
    coeffs = {'intercept': model.intercept_}
    for i, name in enumerate(feature_names):
        coeffs[name] = model.coef_[i]
    
    # Organize coefficients by order and calculate statistics
    result = {
        'coefficients': coeffs,
        'by_order': {},
        'model': {
            'r2_score': model.score(X, fitness_values),
            'model_type': model.__class__.__name__
        }
    }
    
    for term, value in coeffs.items():
        order_key = 0 if term == 'intercept' else len(term.split('*'))
        if order_key not in result['by_order']:
            result['by_order'][order_key] = {}
        result['by_order'][order_key][term] = value
        
    result['statistics'] = _calculate_epistasis_statistics(coeffs)
    
    return result


def _create_design_matrix_one_hot(sequences: np.ndarray,
                                  order: int,
                                  alphabet: List) -> Tuple[np.ndarray, List[str]]:
    """
    Creates a design matrix for regression using one-hot encoding for sequences
    and their interactions.
    """
    n_sequences, seq_length = sequences.shape
    alphabet_map = {val: i for i, val in enumerate(alphabet)}
    n_alphabet = len(alphabet)

    # One-hot encode all sequences
    one_hot_sequences = np.zeros((n_sequences, seq_length, n_alphabet))
    for i, seq in enumerate(sequences):
        for j, allele in enumerate(seq):
            if allele in alphabet_map:
                one_hot_sequences[i, j, alphabet_map[allele]] = 1

    features = []
    feature_names = []

    # Add bias term 
    # Order 1: Main effects of each allele at each position
    if order >= 1:

        #(n_sequences, seq_length * n_alphabet)
        order_1_features = one_hot_sequences.reshape(n_sequences, -1)
        features.append(order_1_features)
        for i in range(seq_length):
            for allele in alphabet:
                feature_names.append(f"pos{i}_{allele}")
    
    # Higher Orders of interactions
    for o in range(2, order + 1):
        for pos_indices in combinations(range(seq_length), o):

            # Start with the one-hot features of the first position in the combination
            interaction_features = one_hot_sequences[:, pos_indices[0], :]
            
            # Iteratively compute the outer product with the other positions
            for i in range(1, o):

                # Einsum computes the batch-wise outer product
                interaction_features = np.einsum('...i,...j->...ij', interaction_features, one_hot_sequences[:, pos_indices[i], :])
                
                # Flatten the last dimensions to keep the feature matrix 2D
                interaction_features = interaction_features.reshape(n_sequences, -1)
            
            features.append(interaction_features)
            
            # Generate feature names for this interaction
            allele_combos = product(alphabet, repeat=o)
            base_names = [f"pos{p}" for p in pos_indices]
            for alleles in allele_combos:
                name = "*".join([f"{base}_{a}" for base, a in zip(base_names, alleles)])
                feature_names.append(name)
    
    # Combine all features into the final design matrix
    X = np.concatenate(features, axis=1) if features else np.empty((n_sequences, 0))
    
    return X, feature_names


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
