# Generate generalized and multiallelic NK landscapes

`create_gnk_landscape` supports a uniform alphabet, per-site alphabets, a full
template with selected variable sites, and an explicit interaction adjacency
matrix. Here `N` is the number of variable sites, not necessarily full sequence
length.

## Install and input

```bash
python -m pip install landscapy
```

An explicit interaction matrix is an `N x N` symmetric binary matrix with zero
diagonal. Its edges describe epistatic interactions, whereas the returned
landscape graph describes single-allele sequence neighbours.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.models import create_gnk_landscape

template = list("ACGTA")
variable_sites = [1, 3, 4]
site_alphabets = {
    1: ["C", "T"],
    3: ["A", "G", "T"],
    4: ["A", "C"],
}
interaction_adjacency = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
)

landscape = create_gnk_landscape(
    N=3,
    alphabet=site_alphabets,
    seed=27,
    adj_mat=interaction_adjacency,
    base_sequence=template,
    variable_sites=variable_sites,
)
layer_name = next(iter(landscape.fitness_layers))
metadata = landscape.fitness_layers[layer_name].metadata

assert len(landscape.sequences) == 2 * 3 * 2
for sequence in landscape.sequences:
    values = list(sequence.to_array())
    assert values[0] == template[0] and values[2] == template[2]
    for site in variable_sites:
        assert values[site] in site_alphabets[site]
assert metadata["N"] == 3
assert metadata["K"] is None
assert metadata["interaction_type"] == "adjacency"
assert metadata["interaction_degrees"] == [1, 2, 1]
assert metadata["variable_sites"] == variable_sites
assert metadata["alphabet_sizes"] == {1: 2, 3: 3, 4: 2}
assert landscape.graph.number_of_nodes() == 12
print(layer_name, metadata)
```

For a uniform alphabet, pass a list such as `alphabet=["A", "C", "G"]`.
`create_nk_multi_landscape` is a deprecated compatibility alias; new work
should call `create_gnk_landscape` directly.

## Common failures

- `N` is set to template length while only a subset of sites is variable.
- Global template coordinates and local adjacency-matrix indices are mixed.
- The interaction adjacency is mistaken for the landscape's Hamming graph.
- Expected states are calculated as `alphabet_size**template_length`.
- Variable-site alphabets or seed metadata are not retained with results.
