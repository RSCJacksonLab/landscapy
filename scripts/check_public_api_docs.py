"""Validate NumPy docstrings for the 0.9 API and non-private source tree."""

from __future__ import annotations

import ast
import inspect
import warnings
from importlib import import_module
from pathlib import Path

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

# Click replaces decorated callbacks with Command objects whose runtime
# signatures are ``*args, **kwargs``. Their option contracts and help rendering
# are covered by tests/test_public_api.py, so only these four wrappers are
# exempt from signature-based numpydoc validation.
CLICK_WRAPPER_EXEMPTIONS = {
    "fitness_landscape.__main__.cli",
    "fitness_landscape.__main__.evol_diffusion_landscape",
    "fitness_landscape.__main__.knn_landscape",
    "fitness_landscape.__main__.phylo_landscape",
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


def _source_paths() -> tuple[list[str], list[str]]:
    source_root = Path(__file__).resolve().parents[1] / "src" / "fitness_landscape"
    missing_module_docstrings: list[str] = []
    paths: list[str] = []

    for source_path in sorted(source_root.rglob("*.py")):
        relative = source_path.relative_to(source_root.parent).with_suffix("")
        if any(
            part.startswith("_") and part not in {"__init__", "__main__"}
            for part in relative.parts
        ):
            continue

        module_parts = (
            relative.parts[:-1] if relative.name == "__init__" else relative.parts
        )
        module_name = ".".join(module_parts)
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
        if ast.get_docstring(tree) is None:
            missing_module_docstrings.append(str(source_path))

        for node in tree.body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.name.startswith("_"):
                continue

            object_path = f"{module_name}.{node.name}"
            paths.append(object_path)
            if not isinstance(node, ast.ClassDef):
                continue

            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if child.name.startswith("_"):
                    continue
                paths.append(f"{object_path}.{child.name}")

    return paths, missing_module_docstrings


def main() -> int:
    failures: list[str] = []
    public_paths = _public_paths()
    source_paths, missing_module_docstrings = _source_paths()

    for source_path in missing_module_docstrings:
        failures.append(f"{source_path}: missing module docstring")

    missing_exemptions = CLICK_WRAPPER_EXEMPTIONS - set(source_paths)
    if missing_exemptions:
        failures.append(
            "Click wrapper exemption drift: missing=" + repr(sorted(missing_exemptions))
        )

    with warnings.catch_warnings(record=True) as parser_warnings:
        warnings.simplefilter("always")
        for path in public_paths:
            result = validate(path)
            for code, message in result["errors"]:
                if code in BLOCKING_CODES:
                    failures.append(f"{path}: {code}: {message}")

        for path in source_paths:
            if path in CLICK_WRAPPER_EXEMPTIONS:
                continue
            result = validate(path)
            for code, message in result["errors"]:
                if code in BLOCKING_CODES:
                    failures.append(f"{path}: {code}: {message}")

    for warning in parser_warnings:
        failures.append(f"numpydoc parser warning: {warning.message}")

    if failures:
        print("Documentation validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Validated {len(public_paths)} public API objects and methods.")
    print(
        f"Validated {len(source_paths) - len(CLICK_WRAPPER_EXEMPTIONS)} "
        "non-private source objects and methods; "
        f"exempted {len(CLICK_WRAPPER_EXEMPTIONS)} Click wrappers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
