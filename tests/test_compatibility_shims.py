"""Contract tests for APIs moved to optional companion packages."""

from __future__ import annotations

import pytest

from fitness_landscape.core import superscape
from fitness_landscape.graph_matching import hierarchical_alignment
from fitness_landscape.graph_matching import latent_alignment
import fitness_landscape.graph_matching as graph_matching


def _assert_moved_dependency_error(callable_, expected_api: str) -> None:
    with pytest.raises(ModuleNotFoundError) as caught:
        callable_()

    message = str(caught.value)
    assert "phylo-landscapy" in message
    assert expected_api in message
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == "phylo_landscapy"


@pytest.mark.parametrize(
    ("placeholder", "expected_api"),
    [
        (superscape.FitnessSuperscape, "FitnessSuperscape"),
        (superscape.NullAligner, "FitnessSuperscape"),
    ],
)
def test_superscape_placeholders_report_the_moved_dependency(
    placeholder,
    expected_api,
):
    _assert_moved_dependency_error(placeholder, expected_api)


def test_graph_matching_package_reports_moved_aligner_dependency():
    def access_aligner():
        return graph_matching.RJMCMCAligner

    _assert_moved_dependency_error(access_aligner, "RJMCMCAligner")


@pytest.mark.parametrize(
    ("module", "name", "expected_api"),
    [
        (
            hierarchical_alignment,
            "HierarchicalRJMCMCAligner",
            "HierarchicalRJMCMCAligner",
        ),
        (latent_alignment, "RJMCMCAligner", "RJMCMCAligner"),
    ],
)
def test_legacy_modules_report_the_moved_dependency(module, name, expected_api):
    def access_name():
        return getattr(module, name)

    _assert_moved_dependency_error(access_name, expected_api)


def test_graph_matching_unknown_attribute_remains_an_attribute_error():
    with pytest.raises(AttributeError, match="not_a_public_api"):
        graph_matching.not_a_public_api
