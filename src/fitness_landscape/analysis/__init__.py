"""Analysis API with optional backends loaded on first use."""

from importlib import import_module


_EXPORTS = {
    "find_greedy_accessible_paths": "adaptive_walk",
    "analyze_path_accessibility": "adaptive_walk",
    "calculate_basin_of_attraction_greedy": "adaptive_walk",
    "calculate_basin_of_attraction_stochastic": "adaptive_walk",
    "adaptive_walk_stochastic": "adaptive_walk",
    "neutral_network_analysis": "adaptive_walk",
    "calculate_ruggedness_dirichlet_energy": "dirichlet_energy",
    "local_dirichlet_energy_contribution": "dirichlet_energy",
    "graph_spectral_analysis": "graph",
    "calculate_epistasis_walsh": "epistasis",
    "calculate_epistasis_regression": "epistasis",
    "calculate_epistasis_ensemble": "epistasis",
    "calculate_epistasis_reference_free": "epistasis",
    "graph_properties": "graph",
    "calculate_ruggedness_local_optima": "graph",
    "resistance_distance_matrix": "graph",
    "category_diffusion_hierarchy": "graph",
    "calculate_ruggedness_autocorrelation_analytical": "random_walk",
    "calculate_ruggedness_autocorrelation_stochastic": "random_walk",
    "category_boundary_crossing_times": "random_walk",
    "compute_ruggedness_diffusion_scale": "diffusion_scale",
    "compute_ruggedness_variance_energy": "diffusion_scale",
    "fit_t_bayesian_laplace": "diffusion_scale",
    "fit_t_grid_posterior": "diffusion_scale",
    "fit_t_profile_likelihood": "diffusion_scale",
    "fit_t_bootstrap": "diffusion_scale",
    "procrustes": "graph_induction_alignment",
    "edge_prf_on_observed": "graph_induction_alignment",
    "sp_rmse": "graph_induction_alignment",
    "spectral_rmse": "graph_induction_alignment",
    "edge_length_stats": "graph_induction_alignment",
    "leaf_spanning_tree": "graph_induction_alignment",
    "leaf_splits": "graph_induction_alignment",
    "rf_distance": "graph_induction_alignment",
    "tree_rf_dissimilarity": "graph_induction_alignment",
    "evaluate_reconstruction": "graph_induction_alignment",
    "evaluate_isorank_alignment": "graph_induction_alignment",
    "analyze_fitness_distribution": "statistics",
    "hypothesis_testing": "statistics",
    "permutation_test": "statistics",
    "subsample_analysis": "statistics",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
