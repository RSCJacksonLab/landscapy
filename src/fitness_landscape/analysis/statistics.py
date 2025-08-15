import numpy as np
import scipy.stats as stats
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

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

def hypothesis_testing(landscape: FitnessLandscape,
                       groups: Dict,
                       **kwargs) -> Dict:
    """
    Perform hypothesis tests to compare fitness between groups.
    Performs battery of statistical tets on the fitnesses of
    provided groups. 
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    groups : dict
        Dictionary mapping group names to lists of sequence indices.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    dict
        Hypothesis testing results.
    """
    # Extract fitness values
    all_fitness = np.array([landscape.get_fitness(seq) for seq in landscape.sequences])
    
    # Extract fitness values for each group
    group_fitness = {}
    
    for group_name, indices in groups.items():
        group_fitness[group_name] = all_fitness[indices]
    
    # Initialize results
    results = {
        'group_stats': {},
        'pairwise_tests': {}
    }
    
    # Calculate statistics for each group
    for group_name, fitness in group_fitness.items():
        results['group_stats'][group_name] = {
            'mean': np.mean(fitness),
            'median': np.median(fitness),
            'std': np.std(fitness),
            'min': np.min(fitness),
            'max': np.max(fitness),
            'n': len(fitness)
        }
    
    # Perform pairwise tests
    group_names = list(groups.keys())
    
    for i, name1 in enumerate(group_names):
        results['pairwise_tests'][name1] = {}
        
        for j, name2 in enumerate(group_names):
            if i >= j:
                continue
            
            # Get fitness values
            fitness1 = group_fitness[name1]
            fitness2 = group_fitness[name2]
            
            # Perform t-test
            t_stat, t_p = stats.ttest_ind(fitness1, fitness2, equal_var=False)
            
            # Perform Mann-Whitney U test
            u_stat, u_p = stats.mannwhitneyu(fitness1, fitness2)
            
            # Perform Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(fitness1, fitness2)
            
            # Store results
            results['pairwise_tests'][name1][name2] = {
                't_test': {
                    'statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < 0.05
                },
                'mann_whitney': {
                    'statistic': u_stat,
                    'p_value': u_p,
                    'significant': u_p < 0.05
                },
                'ks_test': {
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'significant': ks_p < 0.05
                }
            }
    
    # Perform ANOVA if there are more than 2 groups
    if len(groups) > 2:
        # Create list of groups for ANOVA
        anova_groups = [fitness for fitness in group_fitness.values()]
        
        # Perform one-way ANOVA
        f_stat, f_p = stats.f_oneway(*anova_groups)
        
        results['anova'] = {
            'statistic': f_stat,
            'p_value': f_p,
            'significant': f_p < 0.05
        }
    
    return results


def bootstrap_analysis(landscape: FitnessLandscape,
                       statistic_func: Any,
                       n_bootstrap: int = 1000,
                       **kwargs) -> Dict:
    """
    Perform bootstrap analysis to estimate confidence intervals using a
    statistic function.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    statistic_func : callable
        Function that calculates a statistic from fitness values from a
        single group.
    n_bootstrap : int, optional
        Number of bootstrap samples.
    **kwargs
        Additional parameters for statistic_func.
        
    Returns
    -------
    dict
        Bootstrap analysis results.
    """
    # Extract fitness values
    fitness_values = np.array([landscape.get_fitness(seq) for seq in landscape.sequences])
    
    # Calculate observed statistic
    observed = statistic_func(fitness_values, **kwargs)
    
    # Perform bootstrap
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(fitness_values, size=len(fitness_values), replace=True)
        
        # Calculate statistic using the `statistic_func`.
        stat = statistic_func(sample, **kwargs)
        bootstrap_samples.append(stat)
    
    # Calculate confidence intervals
    alpha = kwargs.get('alpha', 0.05)
    lower = np.percentile(bootstrap_samples, alpha * 100 / 2)
    upper = np.percentile(bootstrap_samples, 100 - alpha * 100 / 2)
    
    return {
        'observed': observed,
        'bootstrap_mean': np.mean(bootstrap_samples),
        'bootstrap_std': np.std(bootstrap_samples),
        'confidence_interval': [lower, upper],
        'confidence_level': 1 - alpha,
        'n_bootstrap': n_bootstrap
    }


def permutation_test(landscape: FitnessLandscape,
                     groups: Dict,
                     statistic_func: Any,
                     n_permutations: int = 1000,
                     **kwargs) -> Dict:
    """
    Perform permutation test to assess significance between groups of
    sequences provided as indices.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    groups : dict
        Dictionary mapping group names to lists of sequence indices.
    statistic_func : callable
        Function that calculates a test statistic from two groups.
    n_permutations : int, optional
        Number of permutations.
    **kwargs
        Additional parameters for statistic_func.
        
    Returns
    -------
    dict
        Permutation test results.
    """
    # Extract fitness values
    all_fitness = np.array([landscape.get_fitness(seq) for seq in landscape.sequences])
    
    # Extract fitness values for each group
    group_names = list(groups.keys())
    
    if len(group_names) != 2:
        raise ValueError("Permutation test requires exactly 2 groups")
    
    group1_name, group2_name = group_names
    group1_fitness = all_fitness[groups[group1_name]]
    group2_fitness = all_fitness[groups[group2_name]]
    
    # Calculate observed statistic
    observed = statistic_func(group1_fitness, group2_fitness, **kwargs)
    
    # Combine groups
    combined = np.concatenate([group1_fitness, group2_fitness])
    n1, n2 = len(group1_fitness), len(group2_fitness)
    
    # Perform permutation test
    permutation_samples = []
    
    for _ in range(n_permutations):
        # Shuffle combined data
        np.random.shuffle(combined)
        
        # Split into two groups
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        
        # Calculate statistic
        stat = statistic_func(perm_group1, perm_group2, **kwargs)
        permutation_samples.append(stat)
    
    # Calculate p-value
    if kwargs.get('alternative', 'two-sided') == 'two-sided':
        p_value = np.mean(np.abs(permutation_samples) >= np.abs(observed))
    elif kwargs.get('alternative', 'two-sided') == 'greater':
        p_value = np.mean(permutation_samples >= observed)
    else:  # 'less'
        p_value = np.mean(permutation_samples <= observed)
    
    return {
        'observed': observed,
        'p_value': p_value,
        'significant': p_value < kwargs.get('alpha', 0.05),
        'n_permutations': n_permutations,
        'group1_name': group1_name,
        'group2_name': group2_name,
        'group1_size': n1,
        'group2_size': n2,
        'alternative': kwargs.get('alternative', 'two-sided')
    }