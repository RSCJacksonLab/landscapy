from __future__ import annotations

import pandas as pd
import networkx as nx

from fitness_landscape.core.digraph import create_trajectory_digraph
from fitness_landscape.core.landscape import DirectedFitnessLandscape


def test_create_trajectory_digraph_aggregates_ordered_transitions():
    trajectories = pd.DataFrame(
        [
            {
                "trajectory_id": "t0",
                "step": 0,
                "current_node": "AAAA",
                "next_node": "AAAT",
                "sequence_current": "AAAA",
                "sequence_next": "AAAT",
            },
            {
                "trajectory_id": "t0",
                "step": 1,
                "current_node": "AAAT",
                "next_node": "AATT",
                "sequence_current": "AAAT",
                "sequence_next": "AATT",
            },
            {
                "trajectory_id": "t1",
                "step": 0,
                "current_node": "AAAA",
                "next_node": "AAAT",
                "sequence_current": "AAAA",
                "sequence_next": "AAAT",
            },
        ]
    )

    digraph = create_trajectory_digraph(trajectories)

    assert isinstance(digraph, nx.DiGraph)
    assert digraph.number_of_nodes() == 3
    assert digraph.number_of_edges() == 2
    assert digraph["AAAA"]["AAAT"]["observed_count"] == 2
    assert digraph["AAAA"]["AAAT"]["trajectory_count"] == 2
    assert digraph.nodes["AAAA"]["current_count"] == 2
    assert digraph.nodes["AAAT"]["visit_count"] == 3
    assert "trajectory_stats" in digraph.graph["_auto_annotations"]
    assert "trajectory_role" in digraph.graph["_auto_annotations"]


def test_directed_landscape_build_supports_trajectory_mode():
    trajectories = pd.DataFrame(
        [
            {"trajectory_id": "t0", "step": 0, "current_node": "AAAA", "next_node": "AAAT"},
            {"trajectory_id": "t0", "step": 1, "current_node": "AAAT", "next_node": "AATT"},
        ]
    )

    landscape = DirectedFitnessLandscape.build(trajectories, digraph="trajectory")

    assert isinstance(landscape.graph, nx.DiGraph)
    assert landscape.graph.number_of_nodes() == 3
    assert landscape.graph.number_of_edges() == 2
    assert "trajectory_stats" in landscape.annotation_layers
    assert "trajectory_role" in landscape.annotation_layers
