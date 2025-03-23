"""
Epistasis analysis for fitness landscapes.

This module provides functions for analyzing epistasis (genetic interactions)
in fitness landscapes.
"""

import numpy as np
import torch
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..transforms.walsh_hadamard import walsh_transform, walsh_coefficients, MultialleleWalshTransform


def calculate_epistasis(landscape, order=2, method='walsh', **kwargs):
    """
    Calculate epistasis up to specified order.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    order : int, optional
        Maximum order of epistasis to calculate.
    method : str, optional
        Method for calculating epistasis:
        - 'walsh': Use Walsh-Hadamard transform
        - 'regression': Use linear regression
        - 'ensemble': Use background-averaged (ensemble) epistasis
        - 'reference_free': Use reference-free analysis
    **kwargs
        Additional parameters for the method.
        
    Returns
    -------
    dict
        Epistasis values and statistics.
    """
    if method == 'walsh':
        return _calculate_epistasis_walsh(landscape, order, **kwargs)
    elif method == 'regression':
        return _calculate_epistasis_regression(landscape, order, **kwargs)
    elif method == 'ensemble':
        return _calculate_epistasis_ensemble(landscape, order, **kwargs)
    elif method == 'reference_free':
        return _calculate_epistasis_reference_free(landscape, order, **kwargs)
    else:
        raise ValueError(f"Unsupported epistasis calculation method: {method}")


def _calculate_epistasis_walsh(landscape, order=2, backend='numpy', **kwargs):
    """Calculate epistasis using Walsh-Hadamard transform."""
    # Check if sequences are binary
    is_binary = True
    for seq in landscape.sequences:
        if not set(seq.sequence).issubset({0, 1}):
            is_binary = False
            break
    
    if is_binary:
        # Use standard Walsh transform for binary sequences
        coeffs = walsh_coefficients(landscape, order=order, backend=backend)
        
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
    else:
        # Use multiallelic Walsh transform
        # Determine alphabet sizes
        alphabet_sizes = []
        for i in range(len(landscape.sequences[0])):
            unique_values = set(seq[i] for seq in landscape.sequences)
            alphabet_sizes.append(len(unique_values))
        
        # Create multiallelic transform
        transform = MultialleleWalshTransform(alphabet_sizes, backend=backend)
        
        # Compute transform
        coefficients = transform.transform(landscape)
        
        # Organize coefficients
        result = {
            'coefficients': coefficients,
            'alphabet_sizes': alphabet_sizes
        }
        
        # Calculate summary statistics
        result['statistics'] = {
            'mean': np.mean(np.abs(coefficients)),
            'std': np.std(coefficients),
            'max': np.max(np.abs(coefficients)),
            'min': np.min(np.abs(coefficients))
        }
        
        return result


def _calculate_epistasis_regression(landscape, order=2, regularization=None, alpha=1.0, **kwargs):
    """Calculate epistasis using linear regression."""
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    
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


def _calculate_epistasis_ensemble(landscape, order=2, **kwargs):
    """Calculate background-averaged (ensemble) epistasis."""
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
            from itertools import combinations
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


def _calculate_epistasis_reference_free(landscape, order=2, **kwargs):
    """Calculate reference-free epistasis."""
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


def _create_design_matrix(sequences, order):
    """Create design matrix with interaction terms up to specified order."""
    from sklearn.preprocessing import PolynomialFeatures
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=order, include_bias=True)
    X = poly.fit_transform(sequences)
    
    return X


def _get_feature_names(n_features, order):
    """Get feature names for design matrix."""
    from sklearn.preprocessing import PolynomialFeatures
    
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


def _calculate_epistasis_statistics(coefficients):
    """Calculate summary statistics for epistasis coefficients."""
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


def epistasis_decomposition(landscape, method='walsh', order=None, **kwargs):
    """
    Decompose fitness into epistatic components.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    method : str, optional
        Method for calculating epistasis.
    order : int or None, optional
        Maximum order of epistasis to include.
    **kwargs
        Additional parameters for the method.
        
    Returns
    -------
    dict
        Decomposition results.
    """
    # Calculate epistasis
    epistasis = calculate_epistasis(landscape, order=order, method=method, **kwargs)
    
    # Extract sequences and fitness values
    sequences = [seq.to_array() for seq in landscape.sequences]
    fitness_values = [landscape.get_fitness(seq) for seq in landscape.sequences]
    
    # Initialize decomposition
    decomposition = {
        'observed': fitness_values,
        'components': {},
        'residuals': np.zeros_like(fitness_values)
    }
    
    # Add components for each order
    for o in range(order + 1 if order is not None else max(epistasis['by_order'].keys()) + 1):
        if o not in epistasis['by_order']:
            continue
        
        # Initialize component for this order
        component = np.zeros_like(fitness_values)
        
        # Add contributions from this order
        for term, value in epistasis['by_order'][o].items():
            if term == 'intercept':
                # Add intercept to all sequences
                component += value
            else:
                # Add contribution only to matching sequences
                if method in ['walsh', 'regression']:
                    # Parse term format from these methods
                    if '*' in term:
                        # Regression format: "0*1*2"
                        positions = [int(p) for p in term.split('*')]
                        for i, seq in enumerate(sequences):
                            if all(seq[p] == 1 for p in positions):
                                component[i] += value
                    else:
                        # Walsh format: "0,1,2"
                        positions = [int(p) for p in term.split(',')]
                        for i, seq in enumerate(sequences):
                            if all(seq[p] == 1 for p in positions):
                                component[i] += value
                else:
                    # Parse term format from ensemble/reference_free methods
                    # Format: "0:1,1:0,2:1" (position:value pairs)
                    conditions = []
                    for part in term.split(','):
                        if ':' in part:
                            pos, val = part.split(':')
                            conditions.append((int(pos), int(val)))
                    
                    for i, seq in enumerate(sequences):
                        if all(seq[pos] == val for pos, val in conditions):
                            component[i] += value
        
        # Add component to decomposition
        decomposition['components'][o] = component
        
        # Update residuals
        decomposition['residuals'] = fitness_values - sum(decomposition['components'].values())
    
    # Calculate variance explained by each order
    total_variance = np.var(fitness_values)
    variance_explained = {}
    
    for o, component in decomposition['components'].items():
        variance_explained[o] = np.var(component) / total_variance
    
    decomposition['variance_explained'] = variance_explained
    decomposition['total_variance'] = total_variance
    decomposition['residual_variance'] = np.var(decomposition['residuals'])
    
    return decomposition


def epistasis_statistics(landscape, method='walsh', order=None, **kwargs):
    """
    Calculate statistics of epistatic effects.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    method : str, optional
        Method for calculating epistasis.
    order : int or None, optional
        Maximum order of epistasis to include.
    **kwargs
        Additional parameters for the method.
        
    Returns
    -------
    dict
        Statistics of epistatic effects.
    """
    # Calculate epistasis
    epistasis = calculate_epistasis(landscape, order=order, method=method, **kwargs)
    
    # Initialize statistics
    statistics = {
        'by_order': {},
        'overall': epistasis['statistics']
    }
    
    # Calculate statistics for each order
    for o, coeffs in epistasis['by_order'].items():
        if o == 0:
            # Skip intercept
            continue
        
        values = list(coeffs.values())
        
        if not values:
            continue
        
        statistics['by_order'][o] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'max': np.max(values),
            'min': np.min(values),
            'abs_mean': np.mean(np.abs(values)),
            'count': len(values)
        }
    
    # Calculate proportion of significant interactions
    if 'threshold' in kwargs:
        threshold = kwargs['threshold']
        
        significant_counts = {}
        total_counts = {}
        
        for o, coeffs in epistasis['by_order'].items():
            if o == 0:
                continue
            
            values = list(coeffs.values())
            
            if not values:
                continue
            
            significant = sum(1 for v in values if abs(v) > threshold)
            significant_counts[o] = significant
            total_counts[o] = len(values)
        
        statistics['significant'] = {
            'counts': significant_counts,
            'totals': total_counts,
            'proportions': {o: significant_counts[o] / total_counts[o] for o in significant_counts}
        }
    
    return statistics
