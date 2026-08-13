# Choose optional dependencies and portable fallbacks

Install only the extras required by a workflow, or use `all` for supported
non-ML functionality. Quote the requirement in zsh and other shells so square
brackets are not expanded.

## Install and input

```bash
python -m pip install "landscapy[all]"
python -m pip install "landscapy[ml]"  # separate, explicit ML install
```

The one-command non-ML install was defined in [issue #210](https://github.com/RSCJacksonLab/landscapy/issues/210).
FAISS wheels depend on OS and architecture; BallTree is the portable exact kNN
fallback. Parquet is optional because portable bundles also support the
checksummed JSON-table backend.

## Worked example

```python
# cookbook: test
import importlib.util
import platform

workflows = {
    "core": (None, ["numpy", "networkx", "pandas", "scipy"]),
    "knn": ("knn", ["sklearn"]),
    "tda": ("tda", ["gudhi", "sklearn"]),
    "analysis": ("analysis", ["gudhi", "sklearn"]),
    "faiss": ("faiss", ["faiss", "sklearn"]),
    "alignment": ("alignment", ["softalign"]),
    "phylogeny": ("phylogeny", ["cogent3", "piqtree", "softalign"]),
    "parallel": ("parallel", ["ray"]),
    "embeddings": ("embeddings", ["torch", "transformers", "tqdm"]),
    "ml": ("ml", ["torch", "torch_geometric", "transformers", "tqdm"]),
    "cli": ("cli", ["click"]),
    "parquet": ("parquet", ["pyarrow"]),
}

report = {}
for workflow, (extra, modules) in workflows.items():
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    report[workflow] = {
        "available": not missing,
        "missing": missing,
        "install": "python -m pip install landscapy" if extra is None else f'python -m pip install "landscapy[{extra}]"',
    }

report["all"] = {
    "includes": [name for name in workflows if name != "ml"],
    "install": 'python -m pip install "landscapy[all]"',
    "note": "ML remains explicitly opt-in",
}
environment = {
    "python": platform.python_version(),
    "os": platform.system(),
    "architecture": platform.machine(),
}
assert report["core"]["available"]
assert "ml" not in report["all"]["includes"]
assert "balltree" in "balltree is the fallback when a FAISS wheel is unavailable"
print(environment, report)
```

Missing optional imports raise an actionable command naming the extra. Record
Python, OS, and architecture because upstream wheels can differ even when the
Landscapy version is unchanged.

## Common failures

- An unquoted `landscapy[extra]` requirement is passed to zsh.
- `all` is assumed to include the deliberately separate `ml` dependencies.
- A FAISS index is required where no compatible upstream wheel exists.
- BallTree and FAISS outputs are assumed identical without a topology check.
- JSON-table bundles are rejected merely because `pyarrow` is absent.
