# Audit installed feature availability

An environment should expose the modules required by the preregistered
workflow without silently changing backends. This audit reports import
availability; it does not modify the environment or prove that an optional
backend is scientifically appropriate.

## Input

Run the audit with the same Python executable that will run the analysis.
Record Python, operating system, architecture, and missing modules because
upstream wheels can differ across environments.

## Worked example

```python
# cookbook: test
import importlib.util
import platform

features = {
    "core": ["numpy", "networkx", "pandas", "scipy"],
    "knn": ["sklearn"],
    "tda": ["gudhi", "sklearn"],
    "analysis": ["gudhi", "sklearn"],
    "faiss": ["faiss", "sklearn"],
    "alignment": ["softalign"],
    "phylogeny": ["cogent3", "piqtree", "softalign"],
    "parallel": ["ray"],
    "embeddings": ["torch", "transformers", "tqdm"],
    "ml": ["torch", "torch_geometric", "transformers", "tqdm"],
    "cli": ["click"],
    "parquet": ["pyarrow"],
}

report = {}
for feature, modules in features.items():
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    report[feature] = {
        "available": not missing,
        "missing": missing,
    }

environment = {
    "python": platform.python_version(),
    "os": platform.system(),
    "architecture": platform.machine(),
}
assert report["core"]["available"]
print(environment, report)
```

The result distinguishes import availability from workflow validation. For
example, an available FAISS module does not establish exact-neighbour recall,
and an available CUDA runtime does not establish adequate device memory or
numerical equivalence with CPU execution.

## Common failures

- The audit is run with a different Python executable from the analysis.
- A module import is treated as proof that the backend is scientifically valid.
- A missing FAISS module leads to an undocumented backend change instead of an
  explicit BallTree fallback.
- Python, OS, and architecture are omitted from the environment record.
- A model or reference database download is mistaken for a package dependency.
