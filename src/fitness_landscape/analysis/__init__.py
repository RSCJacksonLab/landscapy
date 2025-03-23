"""
Analysis module for fitness landscape package.

This module provides functions for analyzing fitness landscapes, including
epistasis analysis, ruggedness analysis, path analysis, and statistical analysis.
"""

from .epistasis import (
    calculate_epistasis,
    epistasis_decomposition,
    epistasis_statistics
)

from .ruggedness import (
    calculate_ruggedness,
    adaptive_walk,
    neutral_network_analysis,
    landscape_correlation
)

from .path import (
    find_accessible_paths,
    find_shortest_paths,
    analyze_path_accessibility,
    calculate_path_metrics,
    find_evolutionary_trajectories,
    calculate_basin_of_attraction
)

from .statistics import (
    analyze_fitness_distribution,
    correlation_analysis,
    regression_analysis,
    hypothesis_testing,
    bootstrap_analysis,
    permutation_test
)

__all__ = [
    'calculate_epistasis',
    'epistasis_decomposition',
    'epistasis_statistics',
    'calculate_ruggedness',
    'adaptive_walk',
    'neutral_network_analysis',
    'landscape_correlation',
    'find_accessible_paths',
    'find_shortest_paths',
    'analyze_path_accessibility',
    'calculate_path_metrics',
    'find_evolutionary_trajectories',
    'calculate_basin_of_attraction',
    'analyze_fitness_distribution',
    'correlation_analysis',
    'regression_analysis',
    'hypothesis_testing',
    'bootstrap_analysis',
    'permutation_test'
]
