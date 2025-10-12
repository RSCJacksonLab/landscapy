"""CLI entry points for landscapy."""

from __future__ import annotations

import importlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import click
import numpy as np

from ._const import PROT_20
from .core.graph import _encode_multiallele, create_evol_diffusion_graph
from .core.landscape import FitnessLandscape
from .utils import _compute_embeddings_from_sequences, fasta_to_prot20_sequences


def _load_phylo_module() -> Any:
    """Attempt to import optional legacy CLI module."""

    try:
        return importlib.import_module("phylo_landscapy.__main__")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise click.UsageError(
            "Superscape and phylogenetic CLI commands now live in 'phylo-landscapy'. "
            "Install phylo-landscapy to use those subcommands."
        ) from exc


@click.group()
def cli() -> None:
    """Entry point maintained for backwards compatibility."""


@cli.command()
@click.pass_context
def diffusion_evol_superscape(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    module = _load_phylo_module()
    ctx.forward(module.diffusion_evol_superscape)


def _configure_logger(
    log_file: str | None,
    log_level: str,
    log_prefix: str | None,
    sequences: Path,
    output: Path,
) -> tuple[logging.Logger, str | None]:
    logger = logging.getLogger("fitness_landscape")
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    resolved_log_file = log_file
    if not resolved_log_file and log_prefix:
        ts = time.strftime("%Y%m%d-%H%M%S")
        seq_base = sequences.name or "sequences"
        log_name = f"{log_prefix}_{seq_base}_{ts}_{os.getpid()}.log"
        resolved_log_file = str(output.parent / log_name)

    if resolved_log_file:
        handler_path = Path(resolved_log_file)
        handler_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(handler_path)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == fh.baseFilename
            for h in logger.handlers
        ):
            logger.addHandler(fh)
    else:
        if not logger.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(sh)

    return logger, resolved_log_file


@cli.command()
# Reading / writing
@click.option("--sequences", required=True, type=click.Path(exists=True), help="Path to the input FASTA file or directory.")
@click.option("--output", required=True, type=click.Path(), help="Destination for the serialized FitnessLandscape (.pkl).")
# Diffusion graph parameters
@click.option("--k", type=int, default=50, show_default=True, help="kNN neighbours for pre-filtering.")
@click.option("--t", type=int, default=5, show_default=True, help="Diffusion power (steps).")
@click.option("--tau", type=float, default=1.0, show_default=True, help="Score temperature for kernel conversion.")
@click.option("--connectivity-threshold", type=float, default=1e-4, show_default=True, help="Connectivity threshold for the diffused matrix.")
@click.option("--backend", type=click.Choice(["auto", "faiss", "balltree"]), default="auto", show_default=True, help="kNN backend.")
@click.option("--index-type", type=click.Choice(["hnsw", "flat", "ivf"]), default="hnsw", show_default=True, help="FAISS index type.")
@click.option("--faiss-metric", type=click.Choice(["ip", "l2"]), default="ip", show_default=True, help="FAISS metric.")
@click.option("--include-self", is_flag=True, default=False, help="Include self edges in the kNN graph.")
@click.option("--use-gpu", is_flag=True, default=False, help="Use GPU for FAISS (when available for selected index).")
@click.option("--hnsw-M", "hnsw_m", type=int, default=32, show_default=True, help="HNSW M parameter.")
@click.option("--cpus", type=int, default=1, show_default=True, help="Total Ray CPU slots to allocate for alignment tasks.")
@click.option("--compute-hamming-edges/--no-compute-hamming-edges", default=True, show_default=True, help="Attach expected Hamming edge weights when possible.")
# Embeddings
@click.option("--compute-embeddings/--no-compute-embeddings", default=True, show_default=True, help="Compute node embeddings (OHE or PLM).")
@click.option("--embedding-domain", type=click.Choice(["ohe", "plm"]), default="ohe", show_default=True, help="Embedding domain for node attributes.")
@click.option("--plm-model-name", type=str, default="facebook/esm2_t6_8M_UR50D", show_default=True, help="Protein language model to use when embedding-domain=plm.")
@click.option("--plm-batch-size", type=int, default=64, show_default=True, help="Batch size for PLM embeddings.")
@click.option("--plm-device", type=str, default=None, help="Device for PLM embeddings (e.g. 'cpu' or 'cuda').")
# Embedding checkpoints
@click.option("--embeddings-in", type=click.Path(exists=True), default=None, help="Optional .npy file with precomputed embeddings.")
@click.option("--embeddings-out", type=click.Path(), default=None, help="Optional path to save computed/loaded embeddings (.npy).")
@click.option("--only-embeddings", is_flag=True, default=False, help="Only produce embeddings (skip graph build). Requires --embeddings-out.")
# Logging
@click.option("--log-file", type=click.Path(), default=None, help="Optional log file path.")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO", show_default=True)
@click.option("--log-progress", is_flag=True, default=False, help="Enable verbose progress logging.")
@click.option("--log-prefix", type=str, default=None, help="Derive a log filename using this prefix when --log-file is not provided.")
def evol_diffusion_landscape(
    sequences: str,
    output: str,
    k: int,
    t: int,
    tau: float,
    connectivity_threshold: float,
    backend: str,
    index_type: str,
    faiss_metric: str,
    include_self: bool,
    use_gpu: bool,
    hnsw_m: int,
    cpus: int,
    compute_hamming_edges: bool,
    compute_embeddings: bool,
    embedding_domain: str,
    plm_model_name: str,
    plm_batch_size: int,
    plm_device: str | None,
    embeddings_in: str | None,
    embeddings_out: str | None,
    only_embeddings: bool,
    log_file: str | None,
    log_level: str,
    log_progress: bool,
    log_prefix: str | None,
) -> None:
    """Construct an evolutionary diffusion FitnessLandscape and save it."""

    seq_path = Path(sequences)
    out_path = Path(output)
    logger, resolved_log = _configure_logger(log_file, log_level, log_prefix, seq_path, out_path)

    if log_progress:
        logger.info("Progress logging enabled.")

    t0 = time.perf_counter()
    c0 = time.process_time()
    logger.info("evol-diffusion-landscape: start")

    # Read sequences
    try:
        _t_read0 = time.perf_counter()
        _c_read0 = time.process_time()
        if seq_path.is_dir():
            fasta_files = sorted(
                p for p in seq_path.iterdir() if p.suffix.lower() in {".fasta", ".fa", ".fas"}
            )
            if not fasta_files:
                raise click.UsageError(f"The directory '{sequences}' contains no FASTA files.")
            seqs = []
            for fp in fasta_files:
                logger.info("Reading FASTA: %s", fp)
                seqs.extend(fasta_to_prot20_sequences(fp, strict=False))
            logger.info(
                "Combined sequences from %d FASTA files (total=%d)",
                len(fasta_files),
                len(seqs),
            )
        else:
            seqs = fasta_to_prot20_sequences(seq_path, strict=False)
        logger.info(
            "Read sequences: n=%d wall=%.2fs cpu=%.2fs",
            len(seqs),
            time.perf_counter() - _t_read0,
            time.process_time() - _c_read0,
        )
    except Exception as exc:
        raise click.UsageError(str(exc)) from exc

    if not seqs:
        raise click.UsageError("No sequences parsed from the provided input.")

    # Embeddings
    E = None
    if embeddings_in is not None:
        _t_emb0 = time.perf_counter()
        _c_emb0 = time.process_time()
        logger.info("Loading embeddings from %s", embeddings_in)
        try:
            E = np.load(embeddings_in)
        except Exception as exc:
            raise click.UsageError(f"Failed to load embeddings from {embeddings_in}: {exc}") from exc
        if getattr(E, "shape", (None,))[0] != len(seqs):
            raise click.UsageError(
                f"Embeddings count ({getattr(E, 'shape', (None,))[0]}) does not match sequences ({len(seqs)})."
            )
        logger.info(
            "Embeddings loaded: shape=%s wall=%.2fs cpu=%.2fs",
            getattr(E, "shape", None),
            time.perf_counter() - _t_emb0,
            time.process_time() - _c_emb0,
        )
    elif not compute_embeddings:
        raise click.UsageError(
            "--no-compute-embeddings provided but no --embeddings-in; supply precomputed embeddings or enable computation."
        )
    elif embedding_domain == "ohe":
        _t_emb0 = time.perf_counter()
        _c_emb0 = time.process_time()
        lengths = {len(s) for s in seqs}
        if len(lengths) == 1:
            E, _ = _encode_multiallele(seqs)
        else:
            logger.warning(
                "Sequences have non-uniform lengths (%s). Falling back to composition embeddings for kNN prefilter.",
                sorted(lengths),
            )
            amap = {aa: i for i, aa in enumerate(PROT_20)}
            E = np.zeros((len(seqs), len(PROT_20)), dtype=np.float32)
            for idx, seq in enumerate(seqs):
                arr = getattr(seq, "to_array")()
                counts = np.zeros(len(PROT_20), dtype=np.float32)
                total = 0.0
                for sym in arr:
                    j = amap.get(str(sym).upper())
                    if j is not None:
                        counts[j] += 1.0
                        total += 1.0
                if total > 0:
                    counts /= total
                else:
                    counts[:] = 1.0 / len(PROT_20)
                E[idx] = counts
        logger.info(
            "Embeddings (ohe/composition) built: shape=%s wall=%.2fs cpu=%.2fs",
            getattr(E, "shape", None),
            time.perf_counter() - _t_emb0,
            time.process_time() - _c_emb0,
        )
    else:
        _t_emb0 = time.perf_counter()
        _c_emb0 = time.process_time()
        E = _compute_embeddings_from_sequences(
            seqs,
            model_name=plm_model_name,
            batch_size=plm_batch_size,
            device=plm_device,
        )
        logger.info(
            "Embeddings (PLM) built: shape=%s wall=%.2fs cpu=%.2fs",
            getattr(E, "shape", None),
            time.perf_counter() - _t_emb0,
            time.process_time() - _c_emb0,
        )

    logger.info("Embeddings shape=%s", getattr(E, "shape", None))

    if embeddings_out is not None:
        try:
            np.save(embeddings_out, E)
            logger.info("Saved embeddings to %s", embeddings_out)
        except Exception as exc:
            raise click.UsageError(f"Failed to save embeddings to {embeddings_out}: {exc}") from exc

    if only_embeddings:
        if embeddings_out is None:
            raise click.UsageError("--only-embeddings requires --embeddings-out.")
        logger.info("only-embeddings set; skipping graph construction.")
        logger.info(
            "evol-diffusion-landscape: end wall=%.2fs cpu=%.2fs",
            time.perf_counter() - t0,
            time.process_time() - c0,
        )
        return

    # Build graph
    if cpus < 1:
        raise click.UsageError("--cpus must be at least 1.")
    approx_pairs = 0
    max_pairs = len(seqs) * (len(seqs) - 1) // 2
    if len(seqs) and k > 0:
        approx_pairs = min(max_pairs, (len(seqs) * k) // 2)
    if approx_pairs:
        logger.info(
            "Estimated pairwise alignment tasks ≈ %d (upper bound %d). With cpus=%d, Ray can execute up to %d tasks concurrently.",
            approx_pairs,
            max_pairs,
            cpus,
            cpus,
        )
        if backend != "faiss" and len(seqs) >= 5000:
            logger.warning(
                "Large dataset detected (n=%d). Consider --backend=faiss for faster neighbour search.",
                len(seqs),
            )
    _t_graph0 = time.perf_counter()
    _c_graph0 = time.process_time()
    G = create_evol_diffusion_graph(
        sequences=seqs,
        embeddings=E,
        k=k,
        t=t,
        tau=tau,
        connectivity_threshold=connectivity_threshold,
        backend=backend,
        index_type=index_type,
        faiss_metric=faiss_metric,
        include_self=include_self,
        use_gpu=use_gpu,
        hnsw_M=hnsw_m,
        cpus=cpus,
        _compute_hamming_edges=compute_hamming_edges,
    )
    logger.info(
        "Graph constructed: nodes=%d edges=%d wall=%.2fs cpu=%.2fs",
        G.number_of_nodes(),
        G.number_of_edges(),
        time.perf_counter() - _t_graph0,
        time.process_time() - _c_graph0,
    )

    landscape = FitnessLandscape.from_graph(G)

    _t_save0 = time.perf_counter()
    _c_save0 = time.process_time()
    landscape.save(out_path)
    logger.info(
        "Saved landscape: wall=%.2fs cpu=%.2fs",
        time.perf_counter() - _t_save0,
        time.process_time() - _c_save0,
    )
    logger.info("Landscape saved to %s", output)
    logger.info(
        "evol-diffusion-landscape: end wall=%.2fs cpu=%.2fs",
        time.perf_counter() - t0,
        time.process_time() - c0,
    )

    if resolved_log:
        click.echo(f"Log written to {resolved_log}")


@cli.command()
@click.pass_context
def phylo_landscape(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    module = _load_phylo_module()
    ctx.forward(module.phylo_landscape)


@cli.command()
@click.pass_context
def phylo_superscape(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    module = _load_phylo_module()
    ctx.forward(module.phylo_superscape)


if __name__ == "__main__":  # pragma: no cover
    cli()
