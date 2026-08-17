# Export sequence tensors

`FitnessLandscape.to_sequence_tensors` exports OHE, active embeddings, or
token IDs together with aligned fitness tensors. Use an explicit feature view
in recorded pipelines; `auto` prefers an attached embedding, then tokens, then
OHE.

## Input

This recipe covers OHE and token views. Token views need a compatible
Transformers tokenizer or tokenizer object. Variable token lengths are padded
and returned with an attention mask.

## Worked example

```python
# cookbook: test
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fitness_landscape import BinarySequence
from fitness_landscape.core import FitnessLandscape, NumericFitness

class AuditTokenizer:
    def __call__(self, text, add_special_tokens=True, return_tensors="pt"):
        ids = [101]
        for symbol in text.split():
            ids.extend([11, 12] if symbol == "1" else [10])
        ids.append(102)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
)
sequences = [BinarySequence(text, sequence_id=f"toy-{i}") for i, text in enumerate(table.sequence)]
embedding = np.arange(len(sequences) * 4, dtype=float).reshape(len(sequences), 4) / 10
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"measured": NumericFitness.from_scalars("measured", table.fitness)},
    embeddings={"cached_plm": embedding},
    embedding_domain="cached_plm",
)

auto = landscape.to_sequence_tensors(as_batch=True, feature_view="auto")
embedded = landscape.to_sequence_tensors(as_batch=True, feature_view="embedding")
ohe = landscape.to_sequence_tensors(as_batch=True, feature_view="ohe", include_embeddings=True)
selected_indices = [1, 6]
tokens = landscape.to_sequence_tensors(
    sequence_idx=selected_indices,
    tokenizer=AuditTokenizer(),
    feature_view="tokens",
    include_embeddings=True,
    as_batch=True,
)

assert auto["sequence_tensor"].shape == (8, 4)
assert torch.equal(auto["sequence_tensor"], embedded["sequence_tensor"])
assert ohe["sequence_tensor"].shape == (8, 3, 2)
assert ohe["embedding"].shape == (8, 4)
assert tokens["sequence_tensor"].shape == tokens["attention_mask"].shape
assert tokens["sequence_tensor"].shape[0] == len(selected_indices)
assert tokens["fitness_tensors"]["measured"].shape[0] == len(selected_indices)
np.testing.assert_allclose(
    tokens["fitness_tensors"]["measured"].squeeze().numpy(),
    table.fitness.to_numpy()[selected_indices],
)
assert torch.equal(tokens["embedding"], embedded["sequence_tensor"][selected_indices])
print({"auto": auto["sequence_tensor"].shape, "tokens": tokens["sequence_tensor"].shape})
```

The export does not add sequence IDs or selected indices to the returned
dictionary. Retain `selected_indices` and the corresponding sequence IDs beside
the tensors and verify targets against them, as above.

## Common failures

- `auto` silently changes from OHE to embeddings after an embedding is attached.
- Padded token IDs are used without their attention mask.
- Tokenizer name and revision are not recorded with exported tensors.
- Fitness tensors are squeezed or shuffled independently of features.
- `include_embeddings` is mistaken for changing the selected feature view.
