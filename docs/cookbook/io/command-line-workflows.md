# Run Landscapy from the command line

The CLI is useful for reproducible graph construction from FASTA/alignment
files. Its current graph commands write trusted local compatibility pickles;
convert them immediately to a portable bundle for sharing.

## Input

OHE kNN requires one aligned FASTA with unique IDs and equal aligned length.
BallTree is the portable exact backend.

## Worked example

```python
# cookbook: test
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fitness_landscape.core import FitnessLandscape

fasta_text = ">s0\nMKT\n>s1\nMKS\n>s2\nMRT\n>s3\nAKT\n"
with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    fasta = tmp / "aligned.fasta"
    pickle_path = tmp / "knn.pkl"
    bundle = tmp / "knn_bundle"
    log = tmp / "knn.log"
    fasta.write_text(fasta_text)

    command = [
        sys.executable,
        "-m",
        "fitness_landscape",
        "knn-landscape",
        "--sequences", str(fasta),
        "--output", str(pickle_path),
        "--k", "2",
        "--backend", "balltree",
        "--embedding-domain", "ohe",
        "--tie-policy", "all",
        "--seed", "7",
        "--log-file", str(log),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert pickle_path.exists() and log.exists()

    # This pickle was created locally by the command above. Do not load an
    # untrusted pickle received from another person or repository.
    landscape = FitnessLandscape.load(pickle_path)
    landscape.save_bundle_dir(
        bundle,
        metadata={"command": command, "input": fasta.name},
    )
    restored = FitnessLandscape.load_bundle_dir(bundle)
    assert restored.graph.number_of_nodes() == 4
    assert restored.graph.graph["landscapy_edge_schema"]["constructor"] == "knn-balltree"

    for entry_point in ["landscapy-evol", "landscapy-phylo"]:
        executable = Path(sys.executable).with_name(entry_point)
        result = subprocess.run(
            [str(executable), "--help"], check=True, capture_output=True, text=True
        )
        assert "--help" in result.stdout

print(restored.graph.number_of_nodes(), restored.graph.number_of_edges())
```

`landscapy-evol` additionally needs `knn,alignment,parallel`; `landscapy-phylo`
needs `phylogeny`. Their inputs are aligned FASTA files, and phylogenetic runs
must also record the replacement model, backend, tip-name audit, and tree/ASR
assumptions. Use `--help` from the installed version as the option authority.

## Common failures

- Running a FAISS command on a platform without a compatible wheel instead of
  selecting `--backend balltree`.
- Passing unequal raw sequences to OHE kNN without a biological alignment.
- Sharing the CLI pickle instead of converting to a portable bundle.
- Omitting the full command, log, input hash, seed, and backend metadata.
- Treating CLI completion as validation of the graph's biological meaning.
