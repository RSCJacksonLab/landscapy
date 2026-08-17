# Identify local optima

A local optimum has no represented neighbour with greater fitness. The package
uses a non-strict convention: nodes tied with all neighbours qualify.

## Input

Use a finite scalar active layer and report graph construction, component
support, and tie convention. This example compares two graphs over the same six
versioned rows.

## Worked example

```python
# cookbook: test
from pathlib import Path

import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import calculate_ruggedness_local_optima
from fitness_landscape.core import FitnessLandscape, NumericFitness, create_hamming_graph, create_knn_graph

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
).iloc[:6]
sequences = [
    BinarySequence(text, sequence_id=f"toy-{index}")
    for index, text in enumerate(table["sequence"])
]
fitness = NumericFitness.from_scalars("assay", table["fitness"])
graphs = {
    "hamming": create_hamming_graph(sequences),
    "ohe_knn_k3": create_knn_graph(
        sequences, k=3, embedding_domain="ohe", backend="balltree", tie_policy="all"
    ),
}

report = {}
for name, graph in graphs.items():
    landscape = FitnessLandscape(sequences, graph, fitness_layers={"assay": fitness})
    landscape.view("assay")
    result = calculate_ruggedness_local_optima(landscape)
    report[name] = {
        "edges": graph.number_of_edges(),
        "components": 1,
        "count": result["local_optima_count"],
        "nodes": result["local_optima"],
        "sequence_ids": [
            landscape.sequences[index].id for index in result["local_optima_indices"]
        ],
        "fitness": [
            float(landscape.get_signal()[index]) for index in result["local_optima_indices"]
        ],
        "tie_rule": "fitness >= every represented neighbour",
    }

assert report["hamming"]["count"] == 2
assert report["ohe_knn_k3"]["count"] == 1
assert report["hamming"]["sequence_ids"] == ["toy-3", "toy-5"]
print(report)
```

The same measurements have two Hamming optima but one kNN optimum because the
represented neighbour sets differ. Sparse observation can inflate the count by
omitting fitter single-mutant neighbours.

## Common failures

- A tied plateau is described as one strict optimum.
- Internal node integers are reported without biological sequence identifiers.
- Optima from disconnected components are compared without component support.
- Graph views with different nodes or fitness scales are treated as matched.
- An observed local optimum is called evolutionarily unbeatable.
