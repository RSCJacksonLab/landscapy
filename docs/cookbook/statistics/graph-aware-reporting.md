# Report graph-aware analyses

Graph-aware reporting separates dataset-level independent units from dependent
nodes/edges and preserves component, eligibility, and parameter denominators.

## Install and input

```bash
python -m pip install landscapy
```

This worked example summarizes paired graph-view contrasts across six
independent datasets. The millions of edges contributing to each dataset-level
metric are not the inferential sample size.

## Worked example

```python
# cookbook: test
import numpy as np
import pandas as pd

table = pd.DataFrame(
    {
        "dataset": ["A", "B", "C", "D", "E", "F"],
        "hamming_metric": [0.42, 0.31, np.nan, 0.55, 0.27, 0.48],
        "knn_metric": [0.46, 0.29, 0.40, 0.60, 0.33, 0.49],
        "hamming_components_analyzed": [1, 2, 0, 3, 1, 2],
        "hamming_components_total": [1, 4, 5, 3, 2, 2],
        "knn_components_analyzed": [1, 1, 1, 1, 1, 1],
        "knn_components_total": [1, 1, 1, 1, 1, 1],
        "edge_count_hamming": [120, 450, 900, 1800, 75, 640],
        "edge_count_knn": [500, 900, 1600, 2600, 300, 1200],
    }
)
eligible = table.dropna(subset=["hamming_metric", "knn_metric"]).copy()
eligible["paired_delta"] = eligible["knn_metric"] - eligible["hamming_metric"]

rng = np.random.default_rng(113)
deltas = eligible["paired_delta"].to_numpy()
bootstrap = np.array([
    rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(5000)
])
report = {
    "estimand": "mean paired kNN-minus-Hamming metric across eligible datasets",
    "independent_unit": "dataset",
    "datasets_total": len(table),
    "datasets_eligible": len(eligible),
    "datasets_skipped": table.loc[table["hamming_metric"].isna(), "dataset"].tolist(),
    "skip_reason": "Hamming result undefined after component eligibility rules",
    "mean_delta": float(deltas.mean()),
    "bootstrap_95_interval": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
    "bootstrap_seed": 113,
    "bootstrap_replicates": 5000,
    "edge_count_is_not_n": int(table[["edge_count_hamming", "edge_count_knn"]].sum().sum()),
    "graph_parameter_sensitivity": "report predeclared k/threshold variants separately",
    "analysis_status": "exploratory robustness summary",
}

assert report["datasets_total"] == 6
assert report["datasets_eligible"] == 5
assert report["datasets_skipped"] == ["C"]
assert len(deltas) == 5
print(report)
```

## Reporting checklist

- Name the graph constructor, parameters, weights, node set, and active layer.
- Report component eligibility, skips, isolates, undefined metrics, and honest denominators.
- State the independent unit and dependence among nodes, edges, splits, and repeats.
- Preserve graph-parameter and node-support sensitivity rather than selecting a preferred view.
- Separate exploratory summaries, robustness intervals, and confirmatory inference.
- Record seeds, software/data versions, missing-value rules, and multiplicity families.

## Common failures

- Edge count is presented as the inferential sample size.
- Undefined datasets or components are silently dropped.
- Largest-component-only results are generalized to all observed nodes.
- An exploratory bootstrap interval is labeled a preregistered test.
- Graph parameters are tuned after inspecting the desired contrast.
