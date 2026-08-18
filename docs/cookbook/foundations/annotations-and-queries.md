# Attach and query annotations

Annotations store per-sequence context such as taxonomy, assay background, and
split membership without pretending those labels are measured fitness.

## Input

Annotation tables may be aligned by sequence index, sequence value, or unique
sequence ID. Keys must cover all rows unless `allow_missing=True` is chosen.

## Worked example

```python
# cookbook: test
from pathlib import Path

import pandas as pd

from fitness_landscape import BinarySequence, FitnessLandscape
from fitness_landscape.core import AnnotationLayer

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"),
    dtype={"sequence": "string"},
)
sequences = [BinarySequence(text, sequence_id=f"toy-{i:03d}") for i, text in enumerate(table.sequence)]
landscape = FitnessLandscape.build(sequences, graph="hamming", fitness_layers={})

# Ready-made layer: positional alignment.
context = AnnotationLayer(
    "context",
    table[["taxonomy", "background", "split", "activity_class"]],
    metadata={"source": "cookbook toy data v1.0"},
)
landscape.attach_annotation(context)

# Inline layers: alignment by ID and by sequence value.
landscape.attach_annotation(
    name="plate",
    data={sequence.id: {"plate": f"P{index // 4 + 1}"} for index, sequence in enumerate(sequences)},
    map_by="name",
)
landscape.attach_annotation(
    name="quality",
    data={text: {"passed_qc": True} for text in table.sequence},
    map_by="sequence",
)

query = landscape.query_annotations(
    "context", {"background": "alternate", "split": "test"}
)
assert query.sequence_indices == [5, 6, 7]
assert query.dataframe["taxonomy"].tolist() == ["clade-B"] * 3
assert [sequences[i].id for i in query.sequence_indices] == [
    "toy-005", "toy-006", "toy-007"
]

induced = landscape.graph.subgraph(query.node_ids).copy()
assert set(induced.nodes) == set(query.node_ids)
assert set(induced.edges) == set(query.edges)

# This conversion is intentional: activity_class is a derived assay phenotype,
# not taxonomy/background/split metadata.
derived_activity = landscape.annotation_to_fitness(
    "context",
    field="activity_class",
    name="derived_activity",
    dtype="categorical",
    categories=["low", "mid", "high"],
    metadata={"role": "derived assay phenotype"},
    attach=True,
)
assert derived_activity.get_value(7) == "high"
assert landscape.get_layer("derived_activity") is derived_activity
assert landscape.get_annotation_layer("context").to_dataframe().iloc[7]["split"] == "test"

print(query.sequence_indices)
print(list(induced.nodes), list(induced.edges))
```

The query returns sequence indices, graph node labels, induced edges, sequence
objects, and a filtered DataFrame. The expected match is rows 5–7. The explicit
index/ID assertions verify that annotations remain aligned after filtering.

## Interpretation

The induced subgraph answers which observed edges occur among the matching
rows. It does not show that the annotation caused the topology. `taxonomy`,
`background`, and `split` remain metadata. The example converts only a stated
assay-derived category.

## Common failures

- Non-unique IDs make `map_by="name"` ambiguous.
- Sequence-keyed mapping inherits duplicate-sequence ambiguity.
- `allow_missing=True` creates null annotation rows that later queries may
  exclude.
- Converting taxonomy or split labels to fitness can create an artificial
  numerical signal.
