# Public API for landscapy 0.9

The supported 0.9 API consists of the names below. They are re-exported from
their package namespaces through `__all__`; public methods and properties
defined by the exported classes are supported as well. The release CI validates
this exact surface with `numpydoc`.

The package root re-exports the core sequence, fitness, graph-construction, and
portable-bundle names for convenience. Importing from the namespaces listed
below is also supported.

| Namespace | Public names |
| --- | --- |
| `fitness_landscape.core` | `BaseNumpySequence`, `BinarySequence`, `MultialleleSequence`, `generate_sequences`, `sequence_distance`, `FitnessLandscape`, `AnnotationQueryResult`, `NumericFitness`, `CategoricalFitness`, `ProbabilisticCategoricalFitness`, `AnnotationLayer`, `create_hamming_graph`, `create_knn_graph`, `create_tda_graph`, `create_diffusion_emb_graph`, `create_evol_diffusion_graph`, `create_phylo_graph` |
| `fitness_landscape.analysis` | `find_greedy_accessible_paths`, `analyze_path_accessibility`, `calculate_basin_of_attraction_greedy`, `calculate_basin_of_attraction_stochastic`, `adaptive_walk_stochastic`, `neutral_network_analysis`, `calculate_ruggedness_dirichlet_energy`, `local_dirichlet_energy_contribution`, `graph_spectral_analysis`, `calculate_epistasis_walsh`, `calculate_epistasis_regression`, `calculate_epistasis_ensemble`, `calculate_epistasis_reference_free`, `graph_properties`, `calculate_ruggedness_local_optima`, `resistance_distance_matrix`, `category_diffusion_hierarchy`, `calculate_ruggedness_autocorrelation_analytical`, `calculate_ruggedness_autocorrelation_stochastic`, `category_boundary_crossing_times`, `compute_ruggedness_diffusion_scale`, `compute_ruggedness_variance_energy`, `fit_t_bayesian_laplace`, `fit_t_grid_posterior`, `fit_t_profile_likelihood`, `fit_t_bootstrap`, `procrustes`, `edge_prf_on_observed`, `sp_rmse`, `spectral_rmse`, `edge_length_stats`, `leaf_spanning_tree`, `leaf_splits`, `rf_distance`, `tree_rf_dissimilarity`, `evaluate_reconstruction`, `evaluate_isorank_alignment`, `analyze_fitness_distribution`, `hypothesis_testing`, `permutation_test`, `subsample_analysis` |
| `fitness_landscape.models` | `create_gnk_landscape`, `create_nk_multi_landscape`, `create_nk_binary_landscape`, `create_rmf_landscape`, `create_elementary_landscape` |
| `fitness_landscape.io` | `save_bundle_dir`, `load_bundle_dir`, `export_lsbundle`, `BundleIOError`, `BundleValidationError`, `ChecksumMismatchError` |
| `fitness_landscape.transforms` | `walsh_transform`, `walsh_coefficients`, `graph_fourier_transform`, `eigenmode_decomposition` |
| `fitness_landscape.phylo` | `ASRConstructor`, `build_Q`, `normalise_Q` |
| `fitness_landscape.embedding` | `ESMEmbedder` |
| `fitness_landscape.graph_matching` | `graph_to_length_matrix`, `landmark_mds`, `detect_gap_pairs_kdtree`, `self_tuned_graph`, `reconstruct_latent_graph_with_steiner`, `reconstruct_latent_graph_midpoint`, `normalize_adj_matrix`, `cosine_similarity_matrix`, `isorank_with_features` |

Implementation modules and names beginning with an underscore are private.
RJMCMC aligners have moved to `phylo-landscapy`; landscapy may expose a
compatibility import when that package is installed, but those objects are not
part of the landscapy 0.9 API contract.

The distribution and pairwise testing functions in `fitness_landscape.analysis`
follow the [statistical inference contract](statistical_inference.md), including
finite-data validation, explicit missing-value policies, multiplicity control,
and reproducible Monte Carlo p-values.
