"""Benchmark sparse embedding-diffusion construction over representative cases.

Run from a development checkout with, for example::

    python benchmarks/benchmark_embedding_diffusion.py
    python benchmarks/benchmark_embedding_diffusion.py --case 20000,128,16,1

The output is JSON so release benchmarking can retain the exact dimensions,
constructor budgets, sparse nonzero counts, edge count, and elapsed time.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from fitness_landscape.core.graph import create_diffusion_emb_graph
from fitness_landscape.core.sequence import BinarySequence


DEFAULT_CASES = (
    (1_000, 32, 8, 1),
    (1_000, 32, 8, 2),
    (5_000, 64, 16, 1),
)


def _parse_case(value: str) -> tuple[int, int, int, int]:
    try:
        n, dimensions, k, power = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("case must have form n,d,k,t") from error
    if min(n, dimensions, k, power) < 1:
        raise argparse.ArgumentTypeError("n, d, k, and t must all be positive")
    return n, dimensions, k, power


def run_case(
    case: tuple[int, int, int, int],
    *,
    seed: int,
    max_diffusion_nnz: int,
    max_diffusion_work: int,
) -> dict[str, object]:
    n, dimensions, k, power = case
    embeddings = np.random.default_rng(seed).normal(size=(n, dimensions)).astype(
        np.float32
    )
    sequences = [BinarySequence([index % 2]) for index in range(n)]
    started = time.perf_counter()
    graph = create_diffusion_emb_graph(
        sequences,
        embeddings,
        embedding_domain="plm",
        backend="balltree",
        k=k,
        t=power,
        max_diffusion_nnz=max_diffusion_nnz,
        max_diffusion_work=max_diffusion_work,
    )
    elapsed = time.perf_counter() - started
    return {
        "n": n,
        "dimensions": dimensions,
        "k": k,
        "t": power,
        "elapsed_seconds": elapsed,
        "edges": graph.number_of_edges(),
        **graph.graph["diffusion_construction"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help="repeatable n,d,k,t case; defaults to the release benchmark suite",
    )
    parser.add_argument("--seed", type=int, default=185)
    parser.add_argument("--max-diffusion-nnz", type=int, default=50_000_000)
    parser.add_argument("--max-diffusion-work", type=int, default=1_000_000_000)
    args = parser.parse_args()
    cases = tuple(args.case) if args.case else DEFAULT_CASES
    results = [
        run_case(
            case,
            seed=args.seed + index,
            max_diffusion_nnz=args.max_diffusion_nnz,
            max_diffusion_work=args.max_diffusion_work,
        )
        for index, case in enumerate(cases)
    ]
    print(json.dumps({"cases": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
