# Compare empirical occupancy with a reference panel

`calculate_sequence_occupancy` and `calculate_graph_occupancy` compare two
sequence panels in a shared `FitnessLandscape`. The source lists define the
probability masses; the graph defines the regions or smoothing geometry.
Both functions are exported from `fitness_landscape.analysis` and implemented
in `fitness_landscape.analysis.bottleneck`.

Selections are **zero-based positions in `landscape.sequences`**, not graph-node
labels. Each selection must be nonempty and contain unique indices. A sequence
row may belong to both panels. Distinct rows with duplicate sequence strings
remain separate observations. Unselected nodes participate in graph geometry
and smoothing but carry no initial probability. Neither function mutates the
landscape or applies its absorbing-boundary annotations.

## Sequence counts in shared Louvain regions

NetworkX Louvain communities are computed on the entire supplied graph, once
per call, without using panel or family labels. The function counts each
panel's rows in these common regions and normalizes each panel separately.
`resolution`, `threshold` and `seed` are public Louvain controls; the default
seed is zero. Higher resolution generally produces smaller regions. Returned
community IDs are ordered by their first canonical sequence row; compare
memberships, not the IDs, across different graphs.

`weighting="global"` gives every selected sequence equal weight within its
panel, regardless of the other panel's sample size. `weighting="matched_family"`
gives family `f` the same total mass `alpha[f]` in both panels, assigning each
sequence weight `alpha[f] / count_in_its_panel_and_family`. The default is equal
mass per family; `family_weights` supplies positive relative target masses.
Every selected family must occur in both panels. Missing families are rejected,
not silently removed or imputed. Families are independent supplied labels, not
necessarily the Louvain regions. Choosing the counting regions themselves as
families can force the matched count divergence to zero by construction.
Matched results compare the pooled distributions after family weighting;
they are not the average of separate within-family divergences, and opposing
shifts in different families can cancel in the pooled comparison.

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.analysis import (
    calculate_sequence_occupancy,
    calculate_graph_occupancy,
)

sequences = [BaseNumpySequence([i], sequence_id=f"s{i}") for i in range(6)]
graph = nx.Graph()
for i, sequence in enumerate(sequences):
    graph.add_node(f"node-{i}", sequence=sequence)
graph.add_edges_from([
    ("node-0", "node-1"), ("node-1", "node-2"), ("node-2", "node-0"),
    ("node-3", "node-4"), ("node-4", "node-5"), ("node-5", "node-3"),
])
landscape = FitnessLandscape.build(sequences, graph=graph)
landscape.attach_annotation(
    name="families",
    data={"family": ["x", "x", "y", "x", "y", "y"]},
    map_by="index",
)
labels = landscape.get_annotation_layer("families").to_dataframe()["family"]
reference, observed = [0, 2, 3], [1, 4, 5]

counts = calculate_sequence_occupancy(
    landscape, reference, observed, resolution=1.0, seed=17, weight_key=None,
)
matched = calculate_sequence_occupancy(
    landscape, reference, observed,
    weighting="matched_family", family_labels=labels,
    family_weights={"x": 1, "y": 1},
    resolution=1.0, seed=17, weight_key=None,
)
assert counts["metadata"]["community_count"] == 2
np.testing.assert_allclose(counts["distribution"].reference_probability, [2/3, 1/3])
np.testing.assert_allclose(counts["distribution"].observed_probability, [1/3, 2/3])
np.testing.assert_allclose(matched["family_summary"].target_weight, [0.5, 0.5])
print(counts["distribution"])
print(matched["divergence"])
```

`weight_key="auto"` follows the Landscapy edge-semantics contract. For an
external graph, supply an explicit conductance key or `None` for unweighted
geometry. Conductance weights must be finite and non-negative. Zero-weight
edges are absent from Louvain geometry; edgeless graphs yield singleton
communities. Self-loops retain NetworkX's community-detection semantics.

## Mass-conserving graph smoothing

For weighted adjacency `W`, let `P = D^-1 W` where `D` contains adjacency row
sums. Zero-degree nodes have `P[i,i] = 1`. Self-loop weights count once in each
row sum. The initial row distributions `u_reference` and `u_observed` are
constructed from the same global or matched-family weights as above.

- **Random walk:** `u @ (laziness*I + (1-laziness)*P)**steps`. `steps` is a
  non-negative integer (default one); `laziness` is in `[0,1]` (default 0.5).
  Laziness prevents bipartite oscillation. Zero steps or laziness one leaves
  the source mass unchanged.
- **Heat:** `u @ exp(diffusion_time*(P-I))`. `diffusion_time` is finite and
  non-negative (default one), with zero giving the original distribution.
  This is the continuous-time random-walk heat kernel, not the unnormalized
  combinatorial-Laplacian heat kernel. Time is measured in graph jump-rate
  units and is not calibrated biological time. Laziness does not rescale it.

The two source distributions use exactly the same kernel. Random walks use
sparse matrix-vector products; heat uses SciPy `expm_multiply` on the two mass
vectors together. No dense node-by-node kernel is built. Unused parameters
(`steps`/`laziness` with heat, or `diffusion_time` with random walks) are rejected.
Disconnected components retain their own mass and all components are included.

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.analysis import calculate_graph_occupancy

sequences = [BaseNumpySequence([i], sequence_id=f"s{i}") for i in range(2)]
graph = nx.path_graph(2)
for i, sequence in enumerate(sequences):
    graph.nodes[i]["sequence"] = sequence
landscape = FitnessLandscape.build(sequences, graph=graph)

walk = calculate_graph_occupancy(
    landscape, [0], [1], kernel="random_walk", steps=3, laziness=0.75,
)
heat = calculate_graph_occupancy(
    landscape, [0], [1], kernel="heat", diffusion_time=0.5,
    weighting="matched_family", family_labels=["same-family", "same-family"],
)
np.testing.assert_allclose(walk["distribution"].reference_probability, [0.5625, 0.4375])
np.testing.assert_allclose(
    heat["distribution"].reference_probability,
    [(1 + np.exp(-1))/2, (1 - np.exp(-1))/2],
)
assert heat["metadata"]["absorbing_boundary_applied"] is False
```

## Returned quantities and interpretation

Both methods return:

- `distribution`: community or canonical-node probabilities, enrichment
  `p_observed / p_reference`, and signed local contributions
  `p_observed * log(p_observed / p_reference)`. Community tables also report raw
  source counts and community size. A local KL contribution can be negative;
  the total divergence is non-negative.
- `divergence`: `kl_observed_reference`, `kl_reference_observed`,
  `jensen_shannon` (divergence, not distance), `total_variation`, and
  `observed_mass_without_reference`.
- `sequence_weights`: canonical sequence indices, graph-node labels and both
  initial masses; also family labels in matched mode and community IDs for
  sequence counts.
- `family_summary`: family source counts and normalized target masses, empty
  for global weighting.
- `metadata`: exact source selections, effective weight key, weighting, log
  base, NetworkX version, and partition or smoothing parameters.

Logarithms are natural (nats). No pseudocount is added. Positive observed mass
where reference mass is zero gives infinite forward KL; Jensen-Shannon remains
finite and is bounded by `log(2)`. Enrichment is zero for reference-only support,
infinite for observed-only support, and NaN when both probabilities are zero.
For heat, tiny probabilities can underflow to zero: an infinite computed KL
does not necessarily imply disconnected mathematical support. Numerical
roundoff is cleaned only after checking positivity and probability conservation.

These are descriptive estimators at an explicit resolution, not tests of
selection, equilibrium or physical foldability. Equality of distributions
means equal relative mass, not just overlapping support. Increasing smoothing
erases within-component differences; it must not be tuned solely to obtain a
preferred divergence. Family weighting changes the estimand. Graph geometry
still depends on sampling density even when each source's mass is normalized.
Finite-sample bias, phylogenetic dependence, sampler dependence and uncertainty
require a separate study-specific analysis; no independent-sample P value is
invented here.

References: [Blondel et al. (2008), Louvain communities](https://doi.org/10.1088/1742-5468/2008/10/P10008),
[Coifman and Lafon (2006), diffusion maps](https://doi.org/10.1016/j.acha.2006.04.006),
and [SciPy's sparse exponential action](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html).
