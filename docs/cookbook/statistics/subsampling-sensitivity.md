# Evaluate subsampling sensitivity

Subsampling asks whether a descriptive result is stable to reduced observed
node and edge support.

## Input

Record node/edge retention, seed, sample count, active layer, serial or Ray
execution, component policy, undefined outputs, and failures.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape.analysis import calculate_ruggedness_dirichlet_energy, subsample_analysis
from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=4, K=1, seed=53)
layer_name = landscape.active_layer_name

def analysis(sample):
    try:
        energy = calculate_ruggedness_dirichlet_energy(sample, weight_key=None)
        value = float(energy["global_dirichlet_energy"])
        status = "ok" if np.isfinite(value) else "undefined"
    except (ValueError, FloatingPointError):
        value, status = np.nan, "failed"
    return {
        "global_energy": value,
        "nodes": len(sample),
        "edges": sample.graph.number_of_edges(),
        "components": nx.number_connected_components(sample.graph),
        "status_code": {"ok": 0, "undefined": 1, "failed": 2}[status],
    }

result = subsample_analysis(
    landscape,
    analysis,
    n_samples=30,
    subsample_node_prop=0.75,
    subsample_edge_prop=0.8,
    seed=107,
    layer_name=layer_name,
    use_ray=False,
)
raw = result["results"]
status_counts = {
    "ok": sum(item["status_code"] == 0 for item in raw),
    "undefined": sum(item["status_code"] == 1 for item in raw),
    "failed": sum(item["status_code"] == 2 for item in raw),
}
assert len(raw) == 30
assert status_counts == {"ok": 30, "undefined": 0, "failed": 0}
assert all(item["components"] == 1 for item in raw)
assert {item["nodes"] for item in raw} == {12}
summary = result["per_key"]["global_energy"]
assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]

report = {
    "source": {"nodes": 16, "edges": 32, "components": 1},
    "settings": {"samples": 30, "node_keep": 0.75, "edge_keep": 0.8, "seed": 107, "use_ray": False},
    "statuses": status_counts,
    "sample_component_counts": sorted({item["components"] for item in raw}),
    "energy_summary": summary,
}
print(report)
```

The sampler returns connected observed subgraphs, so the component count is an
explicit conditioning rule. The percentile interval describes robustness over
these correlated support. It is not a confidence interval over
independent biological replicates and supplies no confirmatory p-value.

## Common failures

- Subsamples are called independent replicates.
- Failed or undefined analyses disappear from the denominator.
- Component conditioning and retained node/edge counts are not reported.
- Parallel worker count or seeds are omitted.
- A robustness interval is interpreted as population uncertainty.
- Technical failures in Ray workers are interpreted as part of the subsampled distribution.
