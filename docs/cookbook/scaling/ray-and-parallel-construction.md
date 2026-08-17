# Configure Ray and parallel construction

Use one declared parallelism layer. Landscapy's public Ray workflows start a
runtime only when needed and shut down only a runtime they own. Avoid launching
multithreaded BLAS or nested Ray work inside every worker.

## Input

`create_evol_diffusion_graph(..., cpus=N)` assigns one CPU per alignment task.
`subsample_analysis(..., use_ray=True, num_workers=N)` bounds analysis workers.
Choose `N` from scheduler allocation rather than total host CPUs.

## Worked serial known answer

```python
# cookbook: test
from fitness_landscape.analysis import calculate_ruggedness_local_optima, subsample_analysis
from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=4, K=1, seed=31)
serial = subsample_analysis(
    landscape,
    calculate_ruggedness_local_optima,
    n_samples=4,
    subsample_node_prop=0.75,
    subsample_edge_prop=0.8,
    seed=90210,
    layer_name=landscape.active_layer_name,
    use_ray=False,
)
repeat = subsample_analysis(
    landscape,
    calculate_ruggedness_local_optima,
    n_samples=4,
    subsample_node_prop=0.75,
    subsample_edge_prop=0.8,
    seed=90210,
    layer_name=landscape.active_layer_name,
    use_ray=False,
)
parallel_config = {
    "ray_extra": "parallel",
    "workers": 2,
    "subsample_call": {"use_ray": True, "num_workers": 2},
    "evolutionary_diffusion_call": {"cpus": 2},
    "task_order": "input sample order",
    "seed": 90210,
    "nested_parallelism": False,
    "resume_stage": "persist graph/sample inputs before distributed analysis",
}

assert serial["results"] == repeat["results"]
assert len(serial["results"]) == 4
assert parallel_config["workers"] >= 1
print(parallel_config, serial["per_key"]["local_optima_count"])
```

Run the serial result first, then compare the parallel result with the same
seeds and stable input order on the deployment system. Collect every task
failure and persist completed stage identifiers; Ray execution is not itself a
resumable scientific pipeline.

## Common failures

- Each worker receives all allocated CPUs, causing nested oversubscription.
- A caller-owned Ray runtime is shut down by unrelated cleanup code.
- Parallel completion order is confused with sample identity.
- Failed tasks are silently omitted from the denominator.
- Retrying a stage draws new seeds or overwrites completed artifacts.
