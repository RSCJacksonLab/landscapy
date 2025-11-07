"""CLI entry points for landscapy."""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import click
import numpy as np
from cogent3 import load_aligned_seqs

from ._const import PROT_20
from .core.graph import _encode_multiallele, create_evol_diffusion_graph, create_knn_graph, create_phylo_graph
from .core.landscape import FitnessLandscape
from .utils import _compute_embeddings_from_sequences, fasta_to_prot20_sequences, sanitize_alignment


@click.group()
def cli() -> None:
    """Entry point maintained for backwards compatibility."""


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


def _parse_diffusion_power(
    ctx: click.Context, param: click.Parameter, value: Any
) -> Optional[Union[int, float]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value

    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"inf", "+inf", "infinity"}:
        return math.inf
    if text == "-inf":
        raise click.BadParameter("`t` must be non-negative when finite.", param=param)
    try:
        numeric = float(text)
    except ValueError as exc:
        raise click.BadParameter(f"Unable to parse diffusion power '{value}'.") from exc

    if numeric < 0:
        raise click.BadParameter("`t` must be >= 0 or one of {None, inf}.", param=param)

    if numeric.is_integer():
        return int(numeric)
    return numeric


def _composition_embeddings(seqs: Sequence[Any]) -> np.ndarray:
    amap = {aa: i for i, aa in enumerate(PROT_20)}
    emb = np.zeros((len(seqs), len(PROT_20)), dtype=np.float32)
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
        emb[idx] = counts
    return emb


@cli.command()
# Reading / writing
@click.option("--sequences", required=True, type=click.Path(exists=True), help="Path to the input FASTA file or directory.")
@click.option("--output", required=True, type=click.Path(), help="Destination for the serialized FitnessLandscape (.pkl).")
# Diffusion graph parameters
@click.option("--k", type=int, default=50, show_default=True, help="kNN neighbours for pre-filtering.")
@click.option("--t", callback=_parse_diffusion_power, type=str, default="5", show_default=True, help="Diffusion power; accepts integers, 'inf', or 'none'.")
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
@click.option("--compute-embeddings/--no-compute-embeddings", default=True, show_default=True, help="Compute node embeddings (OHE, composition, or PLM).")
@click.option("--embedding-domain", type=click.Choice(["ohe", "composition", "plm"]), default="ohe", show_default=True, help="Embedding domain for node attributes.")
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
    t: Optional[Union[int, float]],
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

    need_gapped_embeddings = embedding_domain == "ohe"
    seqs_gapped_alignment: list | None = None

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
            use_gapped = need_gapped_embeddings and len(fasta_files) == 1
            for fp in fasta_files:
                logger.info("Reading FASTA: %s", fp)
                if use_gapped:
                    seq_list, gapped = fasta_to_prot20_sequences(fp, strict=False, return_gapped=True)
                    seqs.extend(seq_list)
                    seqs_gapped_alignment = gapped
                else:
                    seqs.extend(fasta_to_prot20_sequences(fp, strict=False))
            logger.info(
                "Combined sequences from %d FASTA files (total=%d)",
                len(fasta_files),
                len(seqs),
            )
        else:
            if need_gapped_embeddings:
                seqs, seqs_gapped_alignment = fasta_to_prot20_sequences(
                    seq_path, strict=False, return_gapped=True
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
        source_for_embeddings = seqs_gapped_alignment if seqs_gapped_alignment is not None else seqs
        lengths = {len(s) for s in source_for_embeddings}
        if len(lengths) != 1:
            raise click.UsageError(
                "Sequences have non-uniform lengths under the provided alignment. "
                "Provide a single aligned FASTA or rerun with --embedding-domain composition."
            )
        E, _ = _encode_multiallele(source_for_embeddings)
        if seqs_gapped_alignment is not None:
            logger.info("Using gapped alignment (length=%d) for OHE embeddings.", len(source_for_embeddings[0]))
        logger.info(
            "Embeddings (ohe) built: shape=%s wall=%.2fs cpu=%.2fs",
            getattr(E, "shape", None),
            time.perf_counter() - _t_emb0,
            time.process_time() - _c_emb0,
        )
    elif embedding_domain == "composition":
        _t_emb0 = time.perf_counter()
        _c_emb0 = time.process_time()
        E = _composition_embeddings(seqs)
        logger.info(
            "Embeddings (composition) built: shape=%s wall=%.2fs cpu=%.2fs",
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
# Reading / writing
@click.option("--sequences", required=True, type=click.Path(exists=True), help="Path to the input FASTA file or directory.")
@click.option("--output", required=True, type=click.Path(), help="Destination for the serialized FitnessLandscape (.pkl).")
# kNN graph parameters
@click.option("--k", type=int, default=50, show_default=True, help="kNN neighbours for graph construction.")
@click.option("--backend", type=click.Choice(["auto", "faiss", "balltree"]), default="auto", show_default=True, help="kNN backend.")
@click.option("--index-type", type=click.Choice(["hnsw", "flat", "ivf"]), default="hnsw", show_default=True, help="FAISS index type.")
@click.option("--faiss-metric", type=click.Choice(["ip", "l2"]), default="ip", show_default=True, help="FAISS metric.")
@click.option("--include-self", is_flag=True, default=False, help="Include self edges in the kNN graph.")
@click.option("--use-gpu", is_flag=True, default=False, help="Use GPU for FAISS (when available for selected index).")
@click.option("--hnsw-M", "hnsw_m", type=int, default=32, show_default=True, help="HNSW M parameter.")
@click.option("--cpus", type=int, default=1, show_default=True, help="Reserved CPU slots (informational; kNN graph is single-threaded).")
@click.option("--tiebuffer", type=int, default=128, show_default=True, help="Extra neighbours to inspect when breaking distance ties.")
@click.option("--tie-policy", type=click.Choice(["all", "min_index", "random"]), default="all", show_default=True, help="How to break kNN distance ties.")
@click.option("--seed", type=int, default=None, help="Random seed used when tie-policy=random.")
@click.option("--compute-hamming-edges/--no-compute-hamming-edges", default=True, show_default=True, help="Attach expected Hamming edge weights when possible.")
# Embeddings
@click.option("--compute-embeddings/--no-compute-embeddings", default=True, show_default=True, help="Compute node embeddings (OHE, composition, or PLM).")
@click.option("--embedding-domain", type=click.Choice(["ohe", "composition", "plm"]), default="ohe", show_default=True, help="Embedding domain for node attributes.")
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
def knn_landscape(
    sequences: str,
    output: str,
    k: int,
    backend: str,
    index_type: str,
    faiss_metric: str,
    include_self: bool,
    use_gpu: bool,
    hnsw_m: int,
    cpus: int,
    tiebuffer: int,
    tie_policy: str,
    seed: int | None,
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
    """Construct a k-nearest neighbour FitnessLandscape and save it."""

    seq_path = Path(sequences)
    out_path = Path(output)
    logger, resolved_log = _configure_logger(log_file, log_level, log_prefix, seq_path, out_path)

    if log_progress:
        logger.info("Progress logging enabled.")

    t0 = time.perf_counter()
    c0 = time.process_time()
    logger.info("knn-landscape: start")

    need_gapped_embeddings = embedding_domain == "ohe"
    seqs_gapped_alignment: list | None = None

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
            use_gapped = need_gapped_embeddings and len(fasta_files) == 1
            for fp in fasta_files:
                logger.info("Reading FASTA: %s", fp)
                if use_gapped:
                    seq_list, gapped = fasta_to_prot20_sequences(fp, strict=False, return_gapped=True)
                    seqs.extend(seq_list)
                    seqs_gapped_alignment = gapped
                else:
                    seqs.extend(fasta_to_prot20_sequences(fp, strict=False))
            logger.info(
                "Combined sequences from %d FASTA files (total=%d)",
                len(fasta_files),
                len(seqs),
            )
        else:
            if need_gapped_embeddings:
                seqs, seqs_gapped_alignment = fasta_to_prot20_sequences(
                    seq_path, strict=False, return_gapped=True
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
        logger.info("Skipping embedding computation (--no-compute-embeddings).")
    elif embedding_domain == "ohe":
        _t_emb0 = time.perf_counter()
        _c_emb0 = time.process_time()
        source_for_embeddings = seqs_gapped_alignment if seqs_gapped_alignment is not None else seqs
        lengths = {len(s) for s in source_for_embeddings}
        if len(lengths) != 1:
            raise click.UsageError(
                "Sequences have non-uniform lengths under the provided alignment. "
                "Provide a single aligned FASTA or rerun with --embedding-domain composition."
            )
        E, _ = _encode_multiallele(source_for_embeddings)
        if seqs_gapped_alignment is not None:
            logger.info("Using gapped alignment (length=%d) for OHE embeddings.", len(source_for_embeddings[0]))
        logger.info(
            "Embeddings (ohe) built: shape=%s wall=%.2fs cpu=%.2fs",
            getattr(E, "shape", None),
            time.perf_counter() - _t_emb0,
            time.process_time() - _c_emb0,
        )
    elif embedding_domain == "composition":
        _t_emb0 = time.perf_counter()
        _c_emb0 = time.process_time()
        E = _composition_embeddings(seqs)
        logger.info(
            "Embeddings (composition) built: shape=%s wall=%.2fs cpu=%.2fs",
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

    if E is not None:
        logger.info("Embeddings shape=%s", getattr(E, "shape", None))

    if embeddings_out is not None:
        if E is None:
            raise click.UsageError("--embeddings-out requires embeddings to be computed or loaded.")
        try:
            np.save(embeddings_out, E)
            logger.info("Saved embeddings to %s", embeddings_out)
        except Exception as exc:
            raise click.UsageError(f"Failed to save embeddings to {embeddings_out}: {exc}") from exc

    if only_embeddings:
        if embeddings_out is None:
            raise click.UsageError("--only-embeddings requires --embeddings-out.")
        if E is None:
            raise click.UsageError("--only-embeddings requires embeddings to be computed or loaded.")
        logger.info("only-embeddings set; skipping graph construction.")
        logger.info(
            "knn-landscape: end wall=%.2fs cpu=%.2fs",
            time.perf_counter() - t0,
            time.process_time() - c0,
        )
        return

    if cpus < 1:
        raise click.UsageError("--cpus must be at least 1.")
    if cpus != 1:
        logger.info("cpus option set to %d (kNN graph construction currently runs single-threaded).", cpus)

    if k < 1:
        raise click.UsageError("--k must be at least 1.")

    if len(seqs) <= 1:
        logger.warning("Only %d sequence(s) provided; resulting graph may be trivial.", len(seqs))

    approx_edges = max(0, len(seqs) * min(k, max(len(seqs) - 1, 0)) // 2)
    if approx_edges:
        logger.info(
            "Estimated undirected edges ≈ %d (k=%d, n=%d).",
            approx_edges,
            k,
            len(seqs),
        )

    _t_graph0 = time.perf_counter()
    _c_graph0 = time.process_time()
    G = create_knn_graph(
        sequences=seqs,
        k=k,
        backend=backend,
        index_type=index_type,
        faiss_metric=faiss_metric,
        include_self=include_self,
        use_gpu=use_gpu,
        hnsw_M=hnsw_m,
        tiebuffer=tiebuffer,
        tie_policy=tie_policy,
        seed=seed,
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
        "knn-landscape: end wall=%.2fs cpu=%.2fs",
        time.perf_counter() - t0,
        time.process_time() - c0,
    )

    if resolved_log:
        click.echo(f"Log written to {resolved_log}")


@cli.command()
@click.option("--alignment", required=True, type=click.Path(exists=True), help="Path to an aligned FASTA file (.fa/.fasta/.fas).")
@click.option("--output", required=True, type=click.Path(), help="Destination for the serialized FitnessLandscape (.pkl).")
@click.option("--replacement-matrix", "replacement_matrix", multiple=True, default=("LG",), show_default=True,
              help="Replacement matrix (or matrices) for phylogenetic reconstruction. Repeat to provide multiple.")
@click.option("--model-fitting/--no-model-fitting", default=True, show_default=True,
              help="Enable model selection / fitting when building the tree.")
@click.option("--phylo-backend", type=click.Choice(["cogent_nj", "iqtree"]), default="iqtree", show_default=True,
              help="Backend used for phylogenetic inference.")
@click.option("--distance", "dist_calc", type=click.Choice(["pdist", "paralinear", "hamming"]), default="pdist", show_default=True,
              help="Distance measure supplied to the phylogenetic backend.")
@click.option("--tree", "tree_path", type=click.Path(exists=True), default=None,
              help="Optional Newick tree to reuse instead of inferring a new one.")
@click.option("--sanitize/--no-sanitize", default=True, show_default=True,
              help="Sanitize the alignment (uppercase, canonical residues, unique IDs).")
@click.option("--compute-hamming-edges/--no-compute-hamming-edges", default=True, show_default=False,
              help="Attach expected Hamming edge annotations when possible.")
@click.option("--lightweight-nodes", is_flag=True, default=False, help="Drop heavy posterior arrays from nodes to save memory.")
@click.option("--hard-ancestors", is_flag=True, default=False,
              help="Collapse ancestral posterior sequences to hard consensus strings.")
@click.option("--nested-parallel", is_flag=True, default=False,
              help="Allow nested parallelism during edge annotation (use with care).")
@click.option("--log-file", type=click.Path(), default=None, help="Optional log file path.")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="INFO", show_default=True)
@click.option("--log-progress", is_flag=True, default=False, help="Enable verbose progress logging during ASR.")
@click.option("--log-prefix", type=str, default=None, help="Derive a log filename using this prefix when --log-file is omitted.")
def phylo_landscape(
    alignment: str,
    output: str,
    replacement_matrix: tuple[str, ...],
    model_fitting: bool,
    phylo_backend: str,
    dist_calc: str,
    tree_path: str | None,
    sanitize: bool,
    compute_hamming_edges: bool,
    lightweight_nodes: bool,
    hard_ancestors: bool,
    nested_parallel: bool,
    log_file: str | None,
    log_level: str,
    log_progress: bool,
    log_prefix: str | None,
) -> None:
    """Construct a phylogenetic FitnessLandscape from an aligned FASTA."""

    aln_path = Path(alignment)
    out_path = Path(output)
    logger, resolved_log = _configure_logger(log_file, log_level, log_prefix, aln_path, out_path)

    if log_progress:
        logger.info("Progress logging enabled.")

    t0 = time.perf_counter()
    c0 = time.process_time()
    logger.info("phylo-landscape: start")

    # Load alignment
    try:
        _t_aln0 = time.perf_counter()
        _c_aln0 = time.process_time()
        aln = load_aligned_seqs(str(aln_path), moltype="protein")
        n_tips = len(aln.names)
        aln_len = getattr(aln, "seq_len", None)
        if aln_len is None:
            try:
                aln_len = aln.shape[1]  # type: ignore[index]
            except Exception:
                aln_len = len(str(aln.get_gapped_seq(aln.names[0]))) if n_tips else 0
        logger.info(
            "Alignment loaded: tips=%d length=%s wall=%.2fs cpu=%.2fs",
            n_tips,
            aln_len,
            time.perf_counter() - _t_aln0,
            time.process_time() - _c_aln0,
        )
    except Exception as exc:
        raise click.UsageError(f"Failed to load alignment {alignment!r}: {exc}") from exc

    if sanitize:
        _t_san0 = time.perf_counter()
        _c_san0 = time.process_time()
        aln = sanitize_alignment(aln)
        logger.info(
            "Alignment sanitised: tips=%d length=%d wall=%.2fs cpu=%.2fs",
            len(aln.names),
            getattr(aln, "seq_len", aln.shape[1] if hasattr(aln, "shape") else 0),
            time.perf_counter() - _t_san0,
            time.process_time() - _c_san0,
        )

    matrices = [str(m) for m in (replacement_matrix or ("LG",))]
    if not matrices:
        matrices = ["LG"]

    kwargs: dict[str, Any] = {}
    if tree_path is not None:
        kwargs["phylogenetic_tree"] = Path(tree_path)

    _t_graph0 = time.perf_counter()
    _c_graph0 = time.process_time()
    try:
        graph = create_phylo_graph(
            sequences=aln,
            replacement_matrix=matrices,
            model_fitting=model_fitting,
            _log_progress=log_progress,
            _nested_parallel=nested_parallel,
            phylo_backend=phylo_backend,
            _dist_calc=dist_calc,
            _compute_hamming_edges=compute_hamming_edges,
            _lightweight_nodes=lightweight_nodes,
            _hard_ancestors=hard_ancestors,
            **kwargs,
        )
    except Exception as exc:
        raise click.UsageError(f"Failed to construct phylogenetic graph: {exc}") from exc
    logger.info(
        "Phylogenetic graph constructed: nodes=%d edges=%d wall=%.2fs cpu=%.2fs",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        time.perf_counter() - _t_graph0,
        time.process_time() - _c_graph0,
    )

    landscape = FitnessLandscape.from_graph(graph)

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
        "phylo-landscape: end wall=%.2fs cpu=%.2fs",
        time.perf_counter() - t0,
        time.process_time() - c0,
    )

    if resolved_log:
        click.echo(f"Log written to {resolved_log}")
if __name__ == "__main__":  # pragma: no cover
    cli()
