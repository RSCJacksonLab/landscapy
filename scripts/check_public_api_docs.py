"""Validate NumPy docstring contracts for the landscapy 0.9 public API."""

from __future__ import annotations

import inspect
import warnings
from importlib import import_module

from numpydoc.validate import validate


# These are the supported import namespaces for 0.9. The names must agree with
# each module's __all__; public methods defined directly on exported classes are
# discovered below and validated as part of the same contract.
PUBLIC_EXPORTS: dict[str, tuple[str, ...]] = {
    "fitness_landscape.core": (
        "BaseNumpySequence",
        "BinarySequence",
        "MultialleleSequence",
        "generate_sequences",
        "sequence_distance",
        "FitnessLandscape",
        "AnnotationQueryResult",
        "NumericFitness",
        "CategoricalFitness",
        "ProbabilisticCategoricalFitness",
        "AnnotationLayer",
        "create_hamming_graph",
        "create_knn_graph",
        "create_tda_graph",
        "create_diffusion_emb_graph",
        "create_evol_diffusion_graph",
        "create_phylo_graph",
    ),
    "fitness_landscape.analysis": (
        "find_greedy_accessible_paths",
        "analyze_path_accessibility",
        "calculate_basin_of_attraction_greedy",
        "calculate_basin_of_attraction_stochastic",
        "adaptive_walk_stochastic",
        "neutral_network_analysis",
        "calculate_ruggedness_dirichlet_energy",
        "local_dirichlet_energy_contribution",
        "graph_spectral_analysis",
        "calculate_epistasis_walsh",
        "calculate_epistasis_regression",
        "calculate_epistasis_ensemble",
        "calculate_epistasis_reference_free",
        "graph_properties",
        "calculate_ruggedness_local_optima",
        "resistance_distance_matrix",
        "category_diffusion_hierarchy",
        "calculate_ruggedness_autocorrelation_analytical",
        "time_continuous_autocorrelation",
        "calculate_ruggedness_autocorrelation_stochastic",
        "category_boundary_crossing_times",
        "compute_ruggedness_diffusion_scale",
        "compute_ruggedness_variance_energy",
        "fit_t_bayesian_laplace",
        "fit_t_grid_posterior",
        "fit_t_profile_likelihood",
        "fit_t_bootstrap",
        "procrustes",
        "edge_prf_on_observed",
        "sp_rmse",
        "spectral_rmse",
        "edge_length_stats",
        "leaf_spanning_tree",
        "leaf_splits",
        "rf_distance",
        "tree_rf_dissimilarity",
        "evaluate_reconstruction",
        "evaluate_isorank_alignment",
        "analyze_fitness_distribution",
        "hypothesis_testing",
        "permutation_test",
        "subsample_analysis",
    ),
    "fitness_landscape.models": (
        "create_gnk_landscape",
        "create_nk_multi_landscape",
        "create_nk_binary_landscape",
        "create_rmf_landscape",
        "create_elementary_landscape",
    ),
    "fitness_landscape.io": (
        "save_bundle_dir",
        "load_bundle_dir",
        "export_lsbundle",
        "BundleIOError",
        "BundleValidationError",
        "ChecksumMismatchError",
    ),
    "fitness_landscape.transforms": (
        "walsh_transform",
        "walsh_coefficients",
        "graph_fourier_transform",
        "eigenmode_decomposition",
    ),
    "fitness_landscape.phylo": (
        "ASRConstructor",
        "build_Q",
        "normalise_Q",
    ),
    "fitness_landscape.embedding": ("ESMEmbedder",),
    "fitness_landscape.graph_matching": (
        "graph_to_length_matrix",
        "landmark_mds",
        "detect_gap_pairs_kdtree",
        "self_tuned_graph",
        "reconstruct_latent_graph_with_steiner",
        "reconstruct_latent_graph_midpoint",
        "normalize_adj_matrix",
        "cosine_similarity_matrix",
        "isorank_with_features",
    ),
}

# These checks protect correctness and machine usability. Numpydoc's layout,
# prose-punctuation, mandatory example, and mandatory See Also rules are
# intentionally not release blockers; see pyproject.toml for that rationale.
BLOCKING_CODES = {
    "GL06",  # malformed/unknown section
    "GL08",  # missing docstring
    "PR01",  # signature parameter missing from docs
    "PR02",  # documented parameter absent from signature
    "PR03",  # parameter order differs
    "PR04",  # parameter type missing
    "RT01",  # return value undocumented
    "RT04",  # multiple returns are unnamed
    "YD01",  # yielded value undocumented
}


def _public_paths() -> list[str]:
    paths: list[str] = []
    seen_objects: set[int] = set()

    for module_name, expected_names in PUBLIC_EXPORTS.items():
        module = import_module(module_name)
        exported = tuple(module.__all__)
        missing = sorted(set(expected_names) - set(exported))
        local_extra = sorted(
            name
            for name in set(exported) - set(expected_names)
            if getattr(getattr(module, name, None), "__module__", "").startswith(
                "fitness_landscape"
            )
        )
        if missing or local_extra:
            raise RuntimeError(
                f"{module_name} public export drift: missing={missing}, extra={local_extra}"
            )

        for name in expected_names:
            obj = getattr(module, name)
            if id(obj) in seen_objects:
                continue
            seen_objects.add(id(obj))
            path = f"{module_name}.{name}"
            paths.append(path)

            if not inspect.isclass(obj):
                continue
            for attribute_name, attribute in obj.__dict__.items():
                if attribute_name.startswith("_"):
                    continue
                if isinstance(attribute, (classmethod, staticmethod)):
                    candidate = attribute.__func__
                elif isinstance(attribute, property):
                    candidate = attribute.fget
                else:
                    candidate = attribute
                if inspect.isfunction(candidate):
                    paths.append(f"{path}.{attribute_name}")

    return paths


def main() -> int:
    failures: list[str] = []
    paths = _public_paths()

    with warnings.catch_warnings(record=True) as parser_warnings:
        warnings.simplefilter("always")
        for path in paths:
            result = validate(path)
            for code, message in result["errors"]:
                if code in BLOCKING_CODES:
                    failures.append(f"{path}: {code}: {message}")

    for warning in parser_warnings:
        failures.append(f"numpydoc parser warning: {warning.message}")

    if failures:
        print("Public API documentation validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Validated {len(paths)} public API objects and methods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
