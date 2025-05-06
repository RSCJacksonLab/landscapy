import numpy as np
import torch
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..transforms.walsh_hadamard import walsh_transform, walsh_coefficients
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures


def calculate_epistasis_walsh(landscape: FitnessLandscape,
                               order: int,
                               backend: Literal['numpy', 'torch'] = 'numpy',
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
    
    backend : str, default=`numpy`
        The backend to use.
    
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
                                    order=order,
                                    backend=backend)
        
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
        
        # Calculate summary statistics
        result['statistics'] = _calculate_epistasis_statistics(coeffs)
        
        return result
    

def calculate_epistasis_regression(landscape: FitnessLandscape,
                                    order: int,
                                    regularization: Literal['l1', 'l2', 'elastic_net'] = None,
                                    alpha: float = 1.0,
                                    **kwargs) -> Dict: 
    """
    Function to measure epistasis with linear modelling.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    order : int
        The order of interaction to test up to. 
    
    regularization : str, default=`None`
        The regularization method to use during linear modelling. If 
        `L1`, LASSO regression is used. If `L2`, RIDGE regression is
        used. If `elastic_net`, `ElasticNet` regression is used. if
        `None`, model is simple linear regression. 
    
    alpha : float
        The regularisation alpha parameter. 
    
    Returns
    -------
    results : Dict
        The results dictionary. 
    """
    
    # Extract sequences and fitness values
    sequences = np.array([seq.to_array() for seq in landscape.sequences])
    fitness_values = np.array([landscape.get_fitness(seq) for seq in landscape.sequences])
    
    # Create design matrix with interaction terms
    X = _create_design_matrix(sequences, order)
    
    # Choose regression model based on regularization
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
    
    # Fit model
    model.fit(X, fitness_values)
    
    # Extract coefficients
    coeffs = {}
    feature_names = _get_feature_names(sequences.shape[1], order)
    
    for i, name in enumerate(feature_names):
        coeffs[name] = model.coef_[i] if i < len(model.coef_) else model.intercept_
    
    # Organize coefficients by order
    result = {
        'coefficients': coeffs,
        'by_order': {}
    }
    
    for term, value in coeffs.items():
        if term == 'intercept':
            order_key = 0
        else:
            order_key = len(term.split('*'))
        
        if order_key not in result['by_order']:
            result['by_order'][order_key] = {}
        
        result['by_order'][order_key][term] = value
    
    # Calculate summary statistics
    result['statistics'] = _calculate_epistasis_statistics(coeffs)
    
    # Add model metrics
    result['model'] = {
        'r2_score': model.score(X, fitness_values),
        'model_type': model.__class__.__name__
    }
    
    return result


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
    # This is a simplified implementation of reference-free analysis
    # For a complete implementation, see the paper by Poelwijk et al. (2019)
    
    # Extract sequences and fitness values
    sequences = [seq.to_array() for seq in landscape.sequences]

    fitness_values = [landscape.get_fitness(seq) for seq in landscape.sequences]
    
    # Calculate global mean fitness
    global_mean = np.mean(fitness_values)
    
    # Calculate epistasis
    result = {
        'coefficients': {},
        'by_order': {0: {'intercept': global_mean}}
    }
    
    # Add zeroth order (global mean)
    result['coefficients']['intercept'] = global_mean
    
    # Calculate first-order effects
    seq_length = len(sequences[0])
    result['by_order'][1] = {}
    
    # For each position, calculate the average effect of each state
    for pos in range(seq_length):
        # Get unique values at this position
        unique_values = set(seq[pos] for seq in sequences)
        
        # For each value, calculate average fitness
        for val in unique_values:
            # Get sequences with this value at this position
            matching_fitnesses = [f for s, f in zip(sequences, fitness_values) if s[pos] == val]
            
            # Calculate average fitness
            avg_fitness = np.mean(matching_fitnesses)
            
            # Calculate effect as deviation from global mean
            effect = avg_fitness - global_mean
            
            # Store effect
            term = f"{pos}:{val}"
            result['coefficients'][term] = effect
            result['by_order'][1][term] = effect
    
    # Calculate higher-order effects
    for o in range(2, order + 1):
        result['by_order'][o] = {}
        
        # Generate all combinations of o positions
        from itertools import combinations
        for pos_combo in combinations(range(seq_length), o):
            # Get all combinations of values at these positions
            value_combos = set()
            for seq in sequences:
                value_combos.add(tuple(seq[p] for p in pos_combo))
            
            # For each combination of values, calculate epistasis
            for vals in value_combos:
                # Get sequences with these values at these positions
                matching_indices = [i for i, seq in enumerate(sequences) 
                                   if all(seq[p] == v for p, v in zip(pos_combo, vals))]
                
                if not matching_indices:
                    continue
                
                matching_fitnesses = [fitness_values[i] for i in matching_indices]
                
                # Calculate observed fitness
                observed = np.mean(matching_fitnesses)
                
                # Calculate expected fitness based on lower-order terms
                expected = global_mean
                
                # Add first-order effects
                for i, (pos, val) in enumerate(zip(pos_combo, vals)):
                    first_order_term = f"{pos}:{val}"
                    expected += result['coefficients'].get(first_order_term, 0)
                
                # Subtract global mean (o-1) times to avoid double counting
                expected -= global_mean * (o - 1)
                
                # Calculate epistasis
                epistasis = observed - expected
                
                # Store epistasis
                term_parts = [f"{pos}:{val}" for pos, val in zip(pos_combo, vals)]
                term = ",".join(term_parts)
                result['coefficients'][term] = epistasis
                result['by_order'][o][term] = epistasis
    
    # Calculate summary statistics
    result['statistics'] = _calculate_epistasis_statistics(result['coefficients'])
    
    return result


def _create_design_matrix(sequences: np.ndarray,
                          order: int) -> np.ndarray:
    """
    Function to create a design matrix of sequences up to a specified
    order.

    Parameters
    ----------
    sequences : np.ndarray
        The sequences to include in the design matrix.
    
    order : int
        The order of interaction to test up to. 
    """
    # Create polynomial features
    poly = PolynomialFeatures(degree=order, include_bias=True)
    X = poly.fit_transform(sequences)
    
    return X

def _get_feature_names(n_features: int,
                       order: int) -> List:
    """
    Function to get feature names for the Design matrix. 

    Parameters
    ----------
    n_features : int
        The number of features. 
    
    order : int
        The order to test epistasis up to in the deign matrix. 
    
    Returns
    -------
    readable_names : List
        A list of human readable feature names that correspond to the
        polynomial epistasis contributions. 
    """
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=order, include_bias=True)
    # Fit on dummy data to get feature names
    poly.fit_transform(np.zeros((1, n_features)))
    
    # Get feature names
    feature_names = poly.get_feature_names_out()
    
    # Convert to more readable format
    readable_names = []
    for name in feature_names:
        if name == '1':
            readable_names.append('intercept')
        else:
            # Replace x0, x1, etc. with position indices
            parts = name.split(' ')
            readable_name = '*'.join(p.replace('x', '') for p in parts if p != '1')
            readable_names.append(readable_name)
    
    return readable_names


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
