# Construct an exact Hamming graph

A Hamming edge joins two observed, aligned sequences that differ at exactly one
site. It does not add unobserved single mutants.

## Install and input

```bash
python -m pip install landscapy
```

All sequences must have the same aligned length and compatible symbol
semantics. This recipe removes `111` from the complete binary fixture to make
the missing-neighbour audit visible.

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import pandas as pd

from fitness_landscape import BinarySequence, FitnessLandscape
from fitness_landscape.analysis import graph_properties
from fitness_landscape.core import create_hamming_graph

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"),
    dtype={"sequence": "string"},
)
observed_text = table.loc[table.sequence != "111", "sequence"].tolist()
sequences = [BinarySequence(text, sequence_id=f"observed-{text}") for text in observed_text]
assert len({len(sequence) for sequence in sequences}) == 1

graph = create_hamming_graph(sequences)
landscape = FitnessLandscape.build(sequences, graph="hamming", fitness_layers={})
assert set(graph.edges) == set(landscape.graph.edges)
assert all(
    sum(a != b for a, b in zip(sequences[u].to_array(), sequences[v].to_array())) == 1
    for u, v in graph.edges
)

observed = set(observed_text)
missing_by_sequence = {}
for text in observed_text:
    theoretical = {
        text[:site] + ("1" if text[site] == "0" else "0") + text[site + 1 :]
        for site in range(len(text))
    }
    missing_by_sequence[text] = sorted(theoretical - observed)

summary = graph_properties(graph)
isolates = list(nx.isolates(graph))
assert sorted({item for values in missing_by_sequence.values() for item in values}) == ["111"]
assert summary["components"]["count"] == 1
assert isolates == []
assert graph.graph["landscapy_edge_schema"]["distance"]["units"] == "hamming_count"

print(graph.number_of_nodes(), graph.number_of_edges(), summary["components"])
print(missing_by_sequence, isolates)
```

The seven observed nodes remain connected and have nine edges. Three observed
sequences report `111` as a missing single-mutant neighbour. That absence is a
property of the sampled table; the constructor does not infer whether `111` is
non-viable, unmeasured, or filtered upstream.

## Interpretation

The graph exactly represents single-site adjacency among observed aligned rows.
Report isolates, components, and missing theoretical neighbours before using
fragmentation, local optima, or accessibility as biological evidence. The
canonical `distance` is a Hamming count; `weight` is conductance under the
[edge-semantics contract](../../edge_semantics.md).

## Common failures

- Unaligned or unequal-length strings invalidate site-wise Hamming distance.
- Gaps are treated as symbols without a declared alignment/gap policy.
- Duplicate sequence rows are mistaken for distinct genotypes.
- Missing neighbours are interpreted as lethal mutations without a sampling
  audit.
- A sparse empirical Hamming graph is compared with a denser representation
  without reporting node support and components.
