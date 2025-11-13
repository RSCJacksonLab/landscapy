import pandas as pd
import pytest

from fitness_landscape.analysis.graph import annotate_louvain_communities


def test_annotate_louvain_communities_creates_layer(binary_3bit_landscape):
    landscape = binary_3bit_landscape
    layer = annotate_louvain_communities(
        landscape,
        annotation_name="auto_comm",
        seed=123,
    )

    assert layer.name == "auto_comm"
    frame = layer.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert {"community_id", "community_label", "community_size", "louvain_community"} <= set(frame.columns)
    assert len(frame) == len(landscape.sequences)
    assert frame["community_id"].notna().any()

    for idx in range(len(landscape.sequences)):
        node = landscape._nodes_by_index[idx]
        annotations = landscape.graph.nodes[node].get("annotations", {})
        assert "auto_comm" in annotations
        assert "community_id" in annotations["auto_comm"]


def test_annotate_louvain_requires_overwrite_flag(binary_3bit_landscape):
    landscape = binary_3bit_landscape
    annotate_louvain_communities(landscape, annotation_name="auto_comm", seed=123)

    with pytest.raises(ValueError):
        annotate_louvain_communities(landscape, annotation_name="auto_comm", seed=123)

    layer = annotate_louvain_communities(
        landscape,
        annotation_name="auto_comm",
        seed=321,
        overwrite=True,
    )
    assert landscape.get_annotation_layer("auto_comm") is layer
