# Construct phylogenetic topology from an alignment

This is an introductory topology recipe for `create_phylo_graph` and
`FitnessLandscape.from_alignment`. The full phylogenetics cookbook, including
ancestral reconstruction and known-tree evaluation, is tracked in [issue
#228](https://github.com/RSCJacksonLab/landscapy/issues/228).

## Install and input

```bash
python -m pip install "landscapy[phylogeny]"
```

The input is a named aligned FASTA with unique tip IDs and common sequence
length. This smoke example uses Cogent3 neighbour joining, disables model
fitting, and leaves ancestral states as placeholders to keep the documentation
run small.

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import numpy as np

from fitness_landscape import FitnessLandscape
from fitness_landscape.core import create_phylo_graph

alignment = Path("docs/cookbook/data/toy_proteins.fasta")
lines = [line.strip() for line in alignment.read_text().splitlines() if line.strip()]
tip_names = [line[1:] for line in lines if line.startswith(">")]
tip_sequences = [line for line in lines if not line.startswith(">")]
assert len(tip_names) == len(set(tip_names)) == 6
assert len({len(sequence) for sequence in tip_sequences}) == 1

options = {
    "replacement_matrix": ["LG"],
    "model_fitting": False,
    "reconstruct_ancestral_states": False,
    "phylo_backend": "cogent_nj",
}
graph = create_phylo_graph(alignment, **options)
landscape = FitnessLandscape.from_alignment(alignment, **options)

assert set(tip_names) <= set(graph.nodes)
assert nx.is_tree(graph)
assert graph.number_of_nodes() == 10
assert graph.number_of_edges() == 9
assert len(landscape) == landscape.graph.number_of_nodes()
assert set(landscape.node_to_sequence_index) == set(landscape.graph.nodes)

extant = {node for node in graph if node in tip_names}
ancestral = set(graph) - extant
assert len(extant) == 6 and len(ancestral) == 4
assert all(graph.nodes[node].get("asr_placeholder") is False for node in extant)
assert all(graph.nodes[node].get("asr_placeholder") is True for node in ancestral)

schema = graph.graph["landscapy_edge_schema"]
assert schema["distance"] == {
    "key": "branch_length", "units": "expected_substitutions_per_site"
}
missing_branch_lengths = [
    (u, v) for u, v, data in graph.edges(data=True)
    if "branch_length" not in data or not np.isfinite(data["branch_length"])
]
print({"tips": sorted(extant), "ancestors": sorted(ancestral), "options": options})
print("missing_branch_lengths", missing_branch_lengths)
```

The fixture produces a connected ten-node tree: six observed tips and four
model-derived ancestral placeholders. Internal-node names and exact tied NJ
resolution are not biological identifiers.

## Branch-length gate and interpretation

`branch_length`, when present, is a distance in expected substitutions per site
and must not be passed directly as conductance. The current Cogent3 topology
route can return edges without a finite `branch_length`; the example audits and
reports those edges. Stop any branch-length-weighted analysis when that list is
non-empty. Do not invent unit lengths or invert missing values silently. This is
also why the fuller route remains gated on the truncation/branch audit in #142
and the regression criteria in #228.

Tree inference, ancestral reconstruction, evolutionary-diffusion affinity, and
fitness-landscape analysis are distinct steps. A successfully constructed tree
does not establish a phylogeny without model adequacy, backend/version records,
alignment QC, and appropriate comparison.

## Common failures

- FASTA tip names are duplicated, changed, or unmatched to downstream rows.
- Alignment columns or terminal gaps are silently truncated.
- Placeholder ancestors are treated as observed or fully reconstructed states.
- Missing branch lengths are replaced without a declared model.
- A rooted display is interpreted as a justified evolutionary root.
