# Installation and system requirements

Complete installation once before using the rest of the cookbook. Recipe pages
assume that Landscapy and the features needed for the chosen workflow are
already available; they do not repeat package-installation commands.

## Supported environment

Use CPython 3.11 or 3.12. The package metadata accepts Python 3.11 and later,
but the release CI matrix currently tests only 3.11 and 3.12. Python 3.13 and
later should therefore be treated as unverified rather than supported by CI.

The lightweight core depends on NetworkX, NumPy, Pandas, and SciPy. It does not
require a GPU. Optional graph construction, analysis, export, parallel,
phylogenetic, embedding, and machine-learning features add the dependencies
listed below. Upstream wheel availability can depend on operating system,
architecture, and Python version.

Use a fresh virtual environment for a reproducible installation. The commands
below call the environment's Python directly, so shell activation is optional.

## Linux and macOS

Select an installed Python 3.11 or 3.12 interpreter. Replace `python3.12` with
`python3.11` when that is the supported interpreter on the system.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "landscapy[all]"
```

Quote extras expressions in POSIX shells. In particular, zsh otherwise treats
square brackets as a filename pattern.

## Windows PowerShell

The Python launcher can select the interpreter explicitly. Calling the virtual
environment's executable directly avoids PowerShell activation-policy issues.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "landscapy[all]"
```

Use `-3.11` instead when Python 3.11 is the supported local interpreter.

## Choose the installed features

`landscapy[all]` is the recommended user installation. It installs every
supported non-ML feature for which the current platform is eligible, including
the CLI, portable Parquet export, kNN and TDA constructors, analysis helpers,
CPU parallelism, alignment, phylogeny, CPU FAISS where eligible, and PLM
embeddings. It deliberately excludes the PyTorch Geometric export dependency.

For a smaller or specialized environment, install only the required feature
set. Extras may be combined inside one quoted expression.

| Requirement | Included capability |
| --- | --- |
| `landscapy` | Core sequences, fitness layers, Hamming graphs, analyses that use only core dependencies, and portable JSON-table bundles |
| `landscapy[all]` | Recommended comprehensive non-ML installation |
| `landscapy[cli]` | Command-line entry points |
| `landscapy[parquet]` | Parquet payloads in portable bundles |
| `landscapy[knn]` | BallTree kNN and embedding-diffusion construction |
| `landscapy[tda]` | TDA graph construction |
| `landscapy[analysis]` | Analysis dependencies shared with TDA workflows |
| `landscapy[faiss]` | CPU FAISS where an eligible upstream wheel is available, plus the BallTree fallback |
| `landscapy[alignment]` | Soft sequence alignment |
| `landscapy[phylogeny]` | Tree inference and ancestral reconstruction |
| `landscapy[parallel]` | Ray-backed CPU parallelism |
| `landscapy[embeddings]` | ESM/Transformers protein embeddings |
| `landscapy[ml]` | Embeddings plus PyTorch Geometric export |

For the comprehensive installation plus ML export, install both feature sets:

```bash
python -m pip install "landscapy[all,ml]"
```

## Platform-specific limits

- `faiss-cpu` is selected only on eligible Linux, macOS, and Windows
  architecture markers. BallTree remains the portable exact kNN backend when a
  compatible FAISS wheel is unavailable.
- The package does not supply GPU FAISS. CPU FAISS and BallTree are the
  supported fallbacks.
- The embedding and ML extras use the PyTorch build resolved by pip. GPU and
  accelerator-specific PyTorch installations are upstream environment choices
  and are not exercised by Landscapy CI.
- The `all` extra includes PLM embedding dependencies, which are comparatively
  large, but it does not include `torch-geometric`; that remains in `ml`.
- Conda may be used to create the Python environment, but the release CI tests
  installation with pip rather than a Conda package.

## Verify the installation

Run these checks with the Python executable from the environment:

```bash
python -c "from importlib.metadata import version; from fitness_landscape import FitnessLandscape; print(version('landscapy'), FitnessLandscape.__name__)"
```

When `all` or `cli` is installed, also verify the command-line module:

```bash
python -m fitness_landscape --help
```

The first command confirms the installed distribution and core import. The
second confirms that the optional command-line module exposes its help text. A
successful import verifies software availability; it does not validate a graph
representation or scientific analysis.

Use the [installed feature audit](feature-availability.md) to record which
optional workflow modules are importable in the same environment.

## Development installation

From a source checkout, install an editable environment with the same feature
scope used by the full Linux test job:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,all,ml]"
python -m pytest
```

## What CI tests

The authoritative configuration is
[`.github/workflows/ci.yaml`](../../../.github/workflows/ci.yaml). Its current
jobs are:

| Job | Environment | Scope |
| --- | --- | --- |
| Documentation | Ubuntu, Python 3.12 | Installs `dev,all`, validates the documented public API, and executes every cookbook recipe block |
| Lint | Ubuntu, Python 3.12 | Checks duplicate definitions, undefined names, and undefined exports with Ruff |
| Test | Ubuntu, Python 3.11 and 3.12 | Installs `dev,all,ml`, runs the full pytest suite with branch coverage, enforces 56% project coverage and publication-module floors |
| Build | Ubuntu, Python 3.12 | Builds source and wheel distributions and validates them with Twine |
| Minimal install | Ubuntu, Python 3.12 | Installs a built wheel with only required dependencies in a clean virtual environment and exercises sequences, fitness, Hamming construction, and portable export |
| Comprehensive install | `ubuntu-latest`, `macos-14`, and `windows-latest`; Python 3.11 and 3.12 | Installs `all`, checks its expected distributions, exercises CLI help, and builds and reloads an OHE/BallTree kNN landscape |

The full pytest suite is not run on macOS or Windows; those systems receive the
comprehensive installation smoke test. CI has no GPU job, does not validate
Python 3.13 or later, and does not enable the opt-in live ESM model-download
test. The workflow currently runs for pushes and pull requests on `main`,
`dev`, and `release`; changes targeting the `docs` branch do not trigger it.
