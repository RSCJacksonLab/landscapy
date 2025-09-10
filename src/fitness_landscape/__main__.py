from networkx.generators import directed
import click
import os
import json
from pathlib import Path
import logging
import time
from fitness_landscape.core.superscape import FitnessSuperscape
from fitness_landscape.core.landscape import DirectedFitnessLandscape
from fitness_landscape.utils import moving_window_alignment, sanitize_alignment
from fitness_landscape.graph_matching.latent_alignment import BernoulliBeta
from cogent3 import load_aligned_seqs
import pickle
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.graph import create_evol_diffusion_graph
from fitness_landscape.core.sequence import read_from_fasta
from fitness_landscape.utils import _compute_embeddings_from_sequences, fasta_to_prot20_sequences
from fitness_landscape._const import PROT_20

@click.group()
def cli():
    """A python tool for fitness landscape analysis."""
    pass

@cli.command()
# Reading and writing results
@click.option('--sequences', required=True, type=click.Path(exists=True), help='Path to the input FASTA file or a directory of FASTA files.')
@click.option('--output', required=True, type=click.Path(), help='Path to save the serialized FitnessSuperscape object.')

# Diffusion graph parameters
@click.option('--k', required=False, type=int, default=50, help='kNN neighbors for pre-filtering.')
@click.option('--t', required=False, type=int, default=5, help='Diffusion power (steps).')
@click.option('--tau', required=False, type=float, default=1.0, help='Score temperature for kernel conversion.')
@click.option('--connectivity-threshold', required=False, type=float, default=1e-4, help='Default connectivity threshold for diffused matrix (used if no --thresholds/--threshold-grid).')
@click.option('--thresholds', required=False, type=float, multiple=True, help='Explicit list of connectivity thresholds to sample (posterior over cutoffs). Provide multiple times: --thresholds 1e-4 --thresholds 1e-3 ...')
@click.option('--threshold-grid', required=False, type=str, default=None, help='Generate thresholds as start:end:count (linear space). Example: 1e-4:1e-1:5')
@click.option('--backend', required=False, type=click.Choice(['auto','faiss','balltree']), default='auto', help='kNN backend.')
@click.option('--index-type', required=False, type=click.Choice(['hnsw','flat','ivf']), default='hnsw', help='FAISS index type.')
@click.option('--faiss-metric', required=False, type=click.Choice(['ip','l2']), default='ip', help='FAISS metric.')
@click.option('--include-self', is_flag=True, default=False, help='Include self edges in kNN.')
@click.option('--use-gpu', is_flag=True, default=False, help='Use GPU for FAISS if available.')
@click.option('--hnsw-M', required=False, type=int, default=32, help='HNSW M parameter.')
@click.option('--cpus', required=False, type=int, default=1, help='CPUs per scoring task (used internally).')
        
@click.option('--compute-embeddings', is_flag=True, default=True, help='Compute embeddings for sequences (ohe or plm).')
@click.option('--embedding-domain', required=False, type=click.Choice(['ohe','plm']), default='ohe', help='Embedding domain for diffusion graph.')
@click.option('--plm-model-name', required=False, type=str, default='facebook/esm2_t6_8M_UR50D', help='PLM model if embedding-domain=plm.')
@click.option('--plm-batch-size', required=False, type=int, default=64, help='Batch size for PLM embeddings.')
@click.option('--plm-device', required=False, type=str, default=None, help='Device for PLM embeddings.')

# RJMCMC sampling
@click.option('--bernoulli-beta-alpha0', required=False, type=float, default=1, help='Bernoulli-Beta prior alpha0 parameter.')
@click.option('--bernoulli-beta-alpha1', required=False, type=float, default=3, help='Bernoulli-Beta prior alpha1 parameter.')
@click.option('--rjmcmc-alpha', required=False, type=float, default=0.9, help='Alpha parameter for the RJMCMC cosine and topological similarity trade-off.')
@click.option('--birth-gamma-prior', required=False, type=float, default=0.01, help='Prior put on the birth rate of new latent slots in alignment. Applies symmetrically to deaths.')
@click.option('--birth-step-prob', required=False, type=float, default=0.5, help='Probability of sampling a birth/death step in the RJMCMC sampler.')
@click.option('--burn-in-samples', required=False, type=int, default=1000, help='The number of burn-in sample steps for the RJMCMC sampler.')
@click.option('--total-samples', required=False, type=int, default=10000, help='The total number of sample steps for the RJMCMC sampler.')
@click.option('--sample-thin', required=False, type=int, default=10, help='Thinning interval for the RJMCMC sampler.')
@click.option('--auto-anchor', required=False, is_flag=True, default=False, help='Boolean flag to auto-anchor nodes to latent slots by cosine similarity.')
@click.option('--anchor-cosine-threshold', required=False, type=float, default=1, help='Cosine similarity threshold for auto-anchoring nodes to latent slots.')
@click.option('--posterior-threshold', required=False, type=float, default=0.25, help='Posterior probability threshold to binarize the latent graph (used in stitching with sliding windows).')
@click.option('--seed', required=False, type=int, default=None, help='Seed for the random number generator to make results reproducible.')
@click.option('--local-window-shifts', required=False, type=int, default=None, help='Number of interleaved shifts for overlapping local windows. Set 0 to disable sliding windows.')
@click.option('--local-window-size', required=False, type=int, default=None, help='Window size (number of nodes) for local sliding windows. Heuristic default if omitted.')
@click.option('--local-window-stride', required=False, type=int, default=None, help='Stride between local windows. Heuristic default if omitted.')

# Logging / checkpointing (same as phylo)
@click.option('--log-file', required=False, type=click.Path(), help='Path to write a detailed log file for this run.')
@click.option('--log-level', required=False, type=click.Choice(['DEBUG','INFO','WARNING','ERROR']), default='INFO', help='Log level for the detailed log file.')
@click.option('--log-progress', is_flag=True, default=False, help='Enable verbose progress logging within constructors.')
@click.option('--log-prefix', required=False, type=str, help='Basename for auto log filename when --log-file is not provided.')
@click.option('--checkpoint-dir', required=False, type=click.Path(), help='Directory to write construction/alignment checkpoints. Defaults to <output_dir>/<output_stem>_ckpt.')
@click.option('--checkpoint-interval', required=False, type=int, default=300, help='Checkpoint interval in seconds during parallel construction.')
@click.option('--resume-checkpoint', required=False, type=click.Path(), help='Resume from a previous construction checkpoint file.')
@click.option('--sequential-construction', is_flag=True, default=False, help='Construct each landscape sequentially (avoids Ray during construction); the diffusion graph internally parallelizes scoring.')
@click.option('--meta-cpu-chains', required=False, type=int, default=os.cpu_count(), help='Number of CPU chains to use for the meta-alignment step in hierarchical alignment.')
@click.option('--local-cpu-chains', required=False, type=int, default=(os.cpu_count()//10 if os.cpu_count()//10 > 1 else 1), help='Number of CPU chains to use for each parallel local alignment chain.')
def diffusion_evol_superscape(sequences,
                                  output,
                                  k,
                                  t,
                                  tau,
                                  connectivity_threshold,
                                  thresholds,
                                  threshold_grid,
                                  backend,
                                  index_type,
                                  faiss_metric,
                                  include_self,
                                  use_gpu,
                                  hnsw_m,
                                  cpus,
                                  compute_embeddings,
                                  embedding_domain,
                                  plm_model_name,
                                  plm_batch_size,
                                  plm_device,
                                  bernoulli_beta_alpha0,
                                  bernoulli_beta_alpha1,
                                  rjmcmc_alpha,
                                  birth_gamma_prior,
                                  birth_step_prob,
                                  burn_in_samples,
                                  total_samples,
                                  sample_thin,
                                  auto_anchor,
                                  anchor_cosine_threshold,
                                  posterior_threshold,
                                  seed,
                                  log_file,
                                  log_level,
                                  log_progress,
                                  log_prefix,
                                  checkpoint_dir,
                                  checkpoint_interval,
                                  resume_checkpoint,
                                  sequential_construction,
                                  meta_cpu_chains,
                                  local_cpu_chains,
                                  local_window_shifts,
                                  local_window_size,
                                  local_window_stride):
    """
    HPC interface to build a Superscape based on the evolutionary diffusion graph.
    Uses create_evol_diffusion_graph under the hood for each input FASTA.
    """
    # Setup logging (reuse same scheme as phylo_superscape)
    logger = logging.getLogger('fitness_landscape')
    if not log_file:
        ts = time.strftime('%Y%m%d-%H%M%S')
        seq_base = os.path.basename(sequences.rstrip('/'))
        out_p = Path(output)
        base_dir = out_p.parent
        default_prefix = log_prefix or 'diffusion_evol_superscape'
        log_name = f"{default_prefix}_{seq_base}_{ts}_{os.getpid()}.log"
        log_file = str(base_dir / log_name)
    if log_file:
        logger.setLevel(getattr(logging, log_level))
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == fh.baseFilename for h in logger.handlers):
            logger.addHandler(fh)
    t0 = time.perf_counter(); c0 = time.process_time()
    logger.info('diffusion-evol-superscape: start')

    # Ingest input
    fasta_paths = []
    if os.path.isdir(sequences):
        fasta_paths = [os.path.join(sequences, f) for f in os.listdir(sequences) if f.endswith(('.fasta','.fa','.fas'))]
    else:
        fasta_paths = [sequences]
    if not fasta_paths:
        raise click.UsageError(f"No FASTA files found in {sequences}")
    logger.info('Found %d FASTA files', len(fasta_paths))

    # Checkpoint dir
    if checkpoint_dir:
        ckpt_dir = Path(checkpoint_dir)
    else:
        out_p = Path(output)
        ckpt_dir = out_p.parent / f"{out_p.stem}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Sampler kwargs (passed to hierarchical aligner/RJMCMC)
    bernoulli_beta = BernoulliBeta(alpha0=bernoulli_beta_alpha0, alpha1=bernoulli_beta_alpha1)
    sampler_kwargs = dict(
        bernoulli_beta=bernoulli_beta,
        alpha=rjmcmc_alpha,
        birth_prior_gamma=birth_gamma_prior,
        birth_death_prob=birth_step_prob,
        burn_in=burn_in_samples,
        samples=total_samples,
        thin=sample_thin,
        auto_anchor=auto_anchor,
        cosine_anchor_threshold=anchor_cosine_threshold,
        seed=seed,
        local_cpu_chains=local_cpu_chains,
        # Optional sliding-window params for hierarchical aligner
        **({"local_window_shifts": local_window_shifts} if local_window_shifts is not None else {}),
        **({"local_window_size": local_window_size} if local_window_size is not None else {}),
        **({"local_window_stride": local_window_stride} if local_window_stride is not None else {}),
        _checkpoint_dir=str(ckpt_dir),
        _checkpoint_interval=checkpoint_interval,
        _resume_checkpoint=resume_checkpoint,
    )

    # Build threshold list (posterior over cutoff samples)
    thrs: list[float] = []
    if thresholds:
        thrs = list(thresholds)
    elif threshold_grid:
        try:
            start_s, end_s, cnt_s = threshold_grid.split(':')
            start_v = float(start_s); end_v = float(end_s); cnt = int(cnt_s)
            if cnt <= 0:
                raise ValueError
            import numpy as _np
            thrs = list(_np.linspace(start_v, end_v, cnt, dtype=float))
        except Exception:
            raise click.UsageError('--threshold-grid must be of the form start:end:count, e.g., 1e-4:1e-1:5')
    else:
        thrs = [connectivity_threshold]

    logger.info('Using %d connectivity thresholds: %s', len(thrs), ', '.join(f'{v:.3g}' for v in thrs))
    logger.info('RJMCMC: alpha=%.3f burn-in=%d samples=%d thin=%d auto_anchor=%s', rjmcmc_alpha, burn_in_samples, total_samples, sample_thin, str(auto_anchor))
    logger.info('Parallelism: meta_cpu_chains=%s local_cpu_chains=%s sequential_construction=%s', str(meta_cpu_chains), str(local_cpu_chains), str(sequential_construction))

    # If requested, construct sequentially here; otherwise, fan out Ray jobs.
    if sequential_construction:
        landscapes = []
        for i, fp in enumerate(fasta_paths, 1):
            logger.info('[dataset %d/%d] reading %s', i, len(fasta_paths), fp)
            try:
                seqs = fasta_to_prot20_sequences(Path(fp), strict=False)
            except Exception as e:
                raise click.UsageError(str(e))
            # Embeddings
            if not compute_embeddings:
                raise click.UsageError('--compute-embeddings must be enabled for diffusion graph')
            if embedding_domain == 'ohe':
                from fitness_landscape.core.graph import _encode_multiallele
                E, _ = _encode_multiallele(seqs)
            else:
                E = _compute_embeddings_from_sequences(seqs, model_name=plm_model_name, batch_size=plm_batch_size, device=plm_device)
            logger.info('[dataset %d/%d] embeddings shape=%s', i, len(fasta_paths), getattr(E,'shape', None))

            # Build one landscape per threshold (posterior over cutoffs)
            for thr in thrs:
                logger.info('[dataset %d/%d] building diffusion graph @ threshold=%g', i, len(fasta_paths), thr)
                G = create_evol_diffusion_graph(
                    sequences=seqs,
                    embeddings=E,
                    k=k,
                    t=t,
                    tau=tau,
                    connectivity_threshold=thr,
                    backend=backend,
                    index_type=index_type,
                    faiss_metric=faiss_metric,
                    include_self=include_self,
                    use_gpu=use_gpu,
                    hnsw_M=hnsw_m,
                    cpus=cpus,
                )
                landscapes.append(FitnessLandscape.from_graph(G))

            # Construction checkpoint
            ckpt_file = ckpt_dir / 'superscape_construction.ckpt.pkl'
            try:
                with open(ckpt_file, 'wb') as f:
                    pickle.dump({'landscapes': landscapes, 'datasets': fasta_paths, 'ts': time.time()}, f)
                if log_progress:
                    logger.info('construction checkpoint written: %s', ckpt_file)
            except Exception:
                pass

        superscape = FitnessSuperscape(landscapes=landscapes, posterior_prob_cutoff=posterior_threshold, **sampler_kwargs)
    else:
        # Build parallel construction jobs: cartesian product of datasets x thresholds
        total_jobs = len(fasta_paths) * len(thrs)
        construction_jobs = []
        job_id = 0
        for fp in fasta_paths:
            logger.info('Queuing jobs for %s across %d thresholds', fp, len(thrs))
            try:
                # Sanitize input: allow non-canonical chars (e.g., X) to be converted to gaps then removed
                seqs = fasta_to_prot20_sequences(Path(fp), strict=False)
            except Exception as e:
                raise click.UsageError(str(e))
            # Precompute embeddings once to avoid recomputation in workers
            if not compute_embeddings:
                raise click.UsageError('--compute-embeddings must be enabled for diffusion graph')
            if embedding_domain == 'ohe':
                from fitness_landscape.core.graph import _encode_multiallele
                E, _ = _encode_multiallele(seqs)
            else:
                E = _compute_embeddings_from_sequences(seqs, model_name=plm_model_name, batch_size=plm_batch_size, device=plm_device)
            for thr in thrs:
                job_id += 1
                construction_jobs.append({
                    'sequences': seqs,
                    'embeddings': E,
                    'graph_type': 'evol_diffusion',
                    # diffusion/evol params
                    'k': k,
                    't': t,
                    'tau': tau,
                    'connectivity_threshold': thr,
                    'backend': backend,
                    'index_type': index_type,
                    'faiss_metric': faiss_metric,
                    'include_self': include_self,
                    'use_gpu': use_gpu,
                    'hnsw_M': hnsw_m,
                    'cpus': cpus,
                    # embeddings provided; skip internal compute
                    'embedding_domain': embedding_domain,
                    '_compute_embeddings': False,
                    # bookkeeping
                    '_log_progress': log_progress,
                    '_job_id': job_id,
                    '_total_jobs': total_jobs,
                })

        # Switch to streaming submission to bound memory and avoid
        # materializing all jobs up front (mirrors phylo_superscape).
        def _construction_job_iter():
            job_id = 0
            total_jobs = len(fasta_paths) * len(thrs)
            for fp in fasta_paths:
                logger.info('Queuing jobs for %s across %d thresholds', fp, len(thrs))
                try:
                    # Sanitize input: allow non-canonical chars (e.g., X) to be converted to gaps then removed
                    seqs = fasta_to_prot20_sequences(Path(fp), strict=False)
                except Exception as e:
                    raise click.UsageError(str(e))
                # Precompute embeddings once per dataset
                if not compute_embeddings:
                    raise click.UsageError('--compute-embeddings must be enabled for diffusion graph')
                if embedding_domain == 'ohe':
                    from fitness_landscape.core.graph import _encode_multiallele
                    E, _ = _encode_multiallele(seqs)
                else:
                    E = _compute_embeddings_from_sequences(seqs, model_name=plm_model_name, batch_size=plm_batch_size, device=plm_device)
                ds_name = os.path.basename(str(fp))
                for thr in thrs:
                    job_id += 1
                    job_label = f"dataset={ds_name} thr={thr:.3g} k={k} t={t}"
                    yield {
                        'sequences': seqs,
                        'embeddings': E,
                        'graph_type': 'evol_diffusion',
                        # diffusion/evol params
                        'k': k,
                        't': t,
                        'tau': tau,
                        'connectivity_threshold': thr,
                        'backend': backend,
                        'index_type': index_type,
                        'faiss_metric': faiss_metric,
                        'include_self': include_self,
                        'use_gpu': use_gpu,
                        'hnsw_M': hnsw_m,
                        'cpus': cpus,
                        # embeddings provided; skip internal compute
                        'embedding_domain': embedding_domain,
                        '_compute_embeddings': False,
                        # bookkeeping
                        '_log_progress': log_progress,
                        '_job_id': job_id,
                        '_total_jobs': total_jobs,
                        '_job_label': job_label,
                    }

        logger.info('Checkpointing: dir=%s interval=%ss', str(ckpt_dir), str(checkpoint_interval))
        logger.info('Launching streaming parallel construction (Ray)')
        superscape = FitnessSuperscape.from_streaming_construction(
            constructor_type='undirected',
            construction_job_iter=_construction_job_iter(),
            posterior_prob_cutoff=posterior_threshold,
            _show_progress=log_progress,
            _construct_checkpoint_dir=str(ckpt_dir),
            _meta_cpu_chains=meta_cpu_chains,
            _parent_task_cpus=0.25,
            **sampler_kwargs,
        )
    superscape.save(Path(output))
    logger.info('Superscape saved to %s', output)
    logger.info('diffusion-evol-superscape: end wall=%.2fs cpu=%.2fs', time.perf_counter()-t0, time.process_time()-c0)
@cli.command()
# Reading and writing results
@click.option('--sequences', required=True, type=click.Path(exists=True), help='Path to the input alignment file or a directory of alignment files.')
@click.option('--output', required=True, type=click.Path(), help='Path to save the serialized FitnessSuperscape object.')

# Processing alignment input
@click.option('--fan-alignment', required=False, is_flag=True, default=False, help='Boolean flag to indicate if the input alignment should be fanned into sub-alignments for parallel processing.')
@click.option('--fan-alignment-window', required=False, type=int, help='Moving window size for fanning the input alignment into sub-alignments.')
@click.option('--fan-alignment-overlap', required=False, type=int, help='Overlap size for fanning the input alignment into sub-alignments.')

# Phylogenetic inference
@click.option('--directed-landscape', required=False, is_flag=True, default=False, help='Boolean flag to indicate if a directed phylogenetic fitness landscape should be constructed.')
@click.option('--compute-phylo-embeddings/--no-compute-phylo-embeddings', default=True, help='Compute embeddings for extant and ancestral sequences to attach to nodes.')
@click.option('--embedding-domain', required=False, type=click.Choice(['ohe', 'plm']), default='ohe', help='Embedding domain for node attributes (ohe or plm).')
@click.option('--replacement-matrix', required=False, multiple=True, default=['LG'], help='Replacement matrix/matrices for IQ-TREE model selection (e.g., LG). Can be provided multiple times.')
@click.option('--model-fitting/--no-model-fitting', default=False, help='Whether to perform IQ-TREE model selection across the provided replacement matrices.')

# RJMCMC sampling
@click.option('--bernoulli-beta-alpha0', required=False, type=float, default=1, help='Bernoulli-Beta prior alpha0 parameter.')
@click.option('--bernoulli-beta-alpha1', required=False, type=float, default=1, help='Bernoulli-Beta prior alpha1 parameter.')
@click.option('--rjmcmc-alpha', required=False, type=float, default=0.9, help='Alpha parameter for the RJMCMC cosine and topological similarity trade-off.')
@click.option('--birth-gamma-prior', required=False, type=float, default=0.02, help='Prior put on the birth rate of new latent slots in alignment. Applies symmetrically to deaths.')
@click.option('--birth-step-prob', required=False, type=float, default=0.2, help='Probability of sampling a birth/death step in the RJMCMC sampler.')
@click.option('--burn-in-samples', required=False, type=int, default=1000, help='The number of burn-in sample steps for the RJMCMC sampler.')
@click.option('--total-samples', required=False, type=int, default=5000, help='The total number of sample steps for the RJMCMC sampler.')
@click.option('--sample-thin', required=False, type=int, default=50, help='Thinning interval for the RJMCMC sampler.')
@click.option('--auto-anchor', required=False, is_flag=True, default=True, help='Boolean flag to auto-anchor nodes to latent slots by cosine similarity.')
@click.option('--anchor-cosine-threshold', required=False, type=float, default=0.99, help='Cosine similarity threshold for auto-anchoring nodes to latent slots.')
@click.option('--posterior-threshold', required=False, type=float, default=0.25, help='Posterior probability threshold to binarize the latent graph (used in stitching with sliding windows).')
@click.option('--sequential-construction', is_flag=True, default=False, help='Construct each landscape sequentially (avoids Ray during construction).')
@click.option('--seed', required=False, type=int, default=None, help='Seed for the random number generator to make results reproducible.')

# Logging
@click.option('--log-file', required=False, type=click.Path(), help='Path to write a detailed log file for this run.')
@click.option('--log-level', required=False, type=click.Choice(['DEBUG','INFO','WARNING','ERROR']), default='INFO', help='Log level for the detailed log file.')
@click.option('--log-progress', is_flag=True, default=False, help='Enable verbose progress logging within constructors.')
@click.option('--log-prefix', required=False, type=str, help='Basename for auto log filename when --log-file is not provided. Defaults to an informative, unique name.')

# Checkpointing (enabled by default for CLI runs unless disabled via env)
@click.option('--checkpoint-dir', required=False, type=click.Path(), help='Directory to write construction/alignment checkpoints. Defaults to <output_dir>/<output_stem>_ckpt.')
@click.option('--checkpoint-interval', required=False, type=int, default=300, help='Checkpoint interval in seconds during parallel construction.')
@click.option('--resume-checkpoint', required=False, type=click.Path(), help='Resume from a previous construction checkpoint file (superscape_construction.ckpt.pkl or hier_local.ckpt.pkl).')

# RJMCMC cpu chains
@click.option('--meta-cpu-chains', required=False, type=int, default=os.cpu_count(), help='Number of CPU chains to use for the meta-alignment step in hierarchical alignment.')
@click.option('--local-cpu-chains', required=False, type=int, default=(os.cpu_count()//10 if os.cpu_count()//10 > 1 else 1), help='Number of CPU chains to use for each parallel local alignment chain.')

# Hierarchical sliding windows
@click.option('--local-window-shifts', required=False, type=int, default=None, help='Number of interleaved shifts for overlapping local windows. Set 0 to disable sliding windows.')
@click.option('--local-window-size', required=False, type=int, default=None, help='Window size (number of nodes) for local sliding windows. Heuristic default if omitted.')
@click.option('--local-window-stride', required=False, type=int, default=None, help='Stride between local windows. Heuristic default if omitted.')

# Alignment cleaning
@click.option('--drop-all-gap-columns/--keep-all-gap-columns', default=True, help='Drop columns that are entirely gaps in each alignment.')
@click.option('--max-gap-frac', required=False, type=float, default=None, help='If set in [0,1], drop columns with gap fraction strictly greater than this threshold.')
@click.option('--max-seq-gap-frac', required=False, type=float, default=None, help='If set in [0,1], drop any sequence whose gap fraction exceeds this threshold (e.g., 0.5 drops sequences >50% gaps).')
# Ray worker lifecycle
@click.option('--ray-fresh-worker/--no-ray-fresh-worker', default=False, help='If set, each Ray job uses a fresh worker (max_calls=1) to avoid native library state reuse.')
# Streaming memory control
@click.option('--max-seqs-per-block', required=False, type=int, default=None, help='If set, split each input alignment into blocks of at most this many sequences and process blocks sequentially. Fanning within a block may still use parallel Ray jobs.')

def phylo_superscape(sequences,
                     output,
                     fan_alignment,
                     fan_alignment_window,
                     fan_alignment_overlap,
                     directed_landscape,
                     compute_phylo_embeddings,
                     embedding_domain,
                     replacement_matrix,
                     model_fitting,
                     bernoulli_beta_alpha0,
                     bernoulli_beta_alpha1,
                     rjmcmc_alpha,
                     birth_gamma_prior,
                     birth_step_prob,
                     burn_in_samples,
                     total_samples,
                     sample_thin,
                     auto_anchor,
                     anchor_cosine_threshold,
                     posterior_threshold,
                     seed,
                     meta_cpu_chains,
                     local_cpu_chains,
                     local_window_shifts,
                     local_window_size,
                     local_window_stride,
                     drop_all_gap_columns,
                     max_gap_frac,
                     max_seq_gap_frac,
                     ray_fresh_worker,
                     max_seqs_per_block,
                     sequential_construction,
                     log_file,
                     log_level,
                     log_progress,
                     log_prefix,
                     checkpoint_dir,
                     checkpoint_interval,
                     resume_checkpoint):
    """
    Constructs and aligns phylogenetic fitness landscapes in parallel.
    """
    
    # Configure logging if requested
    logger = logging.getLogger('fitness_landscape')
    # If no explicit log file, derive an informative unique name next to the output path
    if not log_file:
        ts = time.strftime('%Y%m%d-%H%M%S')
        seq_base = os.path.basename(sequences.rstrip('/'))
        out_p = Path(output)
        base_dir = out_p.parent
        default_prefix = log_prefix or 'phylo_superscape'
        # Include directed flag, sequence source, and PID for uniqueness
        flavor = 'directed' if directed_landscape else 'undirected'
        log_name = f"{default_prefix}_{flavor}_{seq_base}_{ts}_{os.getpid()}.log"
        log_file = str(base_dir / log_name)
    if log_file:
        logger.setLevel(getattr(logging, log_level))
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        # Avoid adding multiple handlers on subsequent invocations
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == fh.baseFilename for h in logger.handlers):
            logger.addHandler(fh)
    t0 = time.perf_counter(); c0 = time.process_time()
    logger.info('phylo-superscape: start')
    logger.info('RJMCMC: alpha=%.3f burn-in=%d samples=%d thin=%d auto_anchor=%s', rjmcmc_alpha, burn_in_samples, total_samples, sample_thin, str(auto_anchor))
    logger.info('Parallelism: meta_cpu_chains=%s local_cpu_chains=%s sequential_construction=%s', str(meta_cpu_chains), str(local_cpu_chains), str(sequential_construction))
    if fan_alignment:
        logger.info('Fanning enabled: window=%s overlap=%s', str(fan_alignment_window), str(fan_alignment_overlap))
    if max_seqs_per_block:
        logger.info('Sequence blocking: max_seqs_per_block=%s', str(max_seqs_per_block))

    # Helpers hoisted so both sub-iterators can reuse them
    from cogent3.core.alignment import make_aligned_seqs

    def _trim_alignment(alignment):
        try:
            names = list(alignment.names)
            seqs = [str(alignment.get_gapped_seq(n)) for n in names]
        except Exception:
            return alignment
        if not seqs:
            return alignment
        L = len(seqs[0])
        keep_mask = [True] * L
        for i in range(L):
            col = [s[i] for s in seqs]
            gap_count = sum(1 for c in col if c == '-')
            # Drop all-gap columns
            if drop_all_gap_columns and gap_count == len(col):
                keep_mask[i] = False
                continue
            # Optional high-gap filter
            if max_gap_frac is not None and 0.0 <= float(max_gap_frac) <= 1.0:
                if (gap_count / len(col)) > float(max_gap_frac):
                    keep_mask[i] = False
        if all(keep_mask):
            return alignment
        if not any(keep_mask):
            # Dropped all columns; return None to signal skip
            return None
        new_map = {n: ''.join(ch for ch, k in zip(s, keep_mask) if k) for n, s in zip(names, seqs)}
        return make_aligned_seqs(new_map, moltype='protein')

    def _drop_gappy_sequences(alignment):
        if max_seq_gap_frac is None:
            return alignment
        try:
            thr = float(max_seq_gap_frac)
        except Exception:
            return alignment
        if not (0.0 <= thr <= 1.0):
            return alignment
        try:
            names = list(alignment.names)
            seqs = [str(alignment.get_gapped_seq(n)) for n in names]
        except Exception:
            return alignment
        if not seqs:
            return alignment
        L = len(seqs[0]) or 1
        keep_names = []
        for n, s in zip(names, seqs):
            gap_frac = (s.count('-') / L)
            if gap_frac <= thr:
                keep_names.append(n)
        if len(keep_names) == len(names):
            return alignment
        if not keep_names:
            logger.warning('All sequences exceeded max-seq-gap-frac=%.3f after trimming; skipping this alignment.', thr)
            return None
        new_map = {n: str(alignment.get_gapped_seq(n)) for n in keep_names}
        return make_aligned_seqs(new_map, moltype='protein')

    def _iter_sub_alignments():
        from fitness_landscape.utils import iter_moving_window_alignment
        if os.path.isdir(sequences):
            fasta_files = [f for f in os.listdir(sequences) if f.endswith(('.fasta', '.fa', '.fas'))]
            if not fasta_files:
                raise click.UsageError(f"The directory '{sequences}' contains no FASTA files.")
            for fasta_file in fasta_files:
                alignment_path = os.path.join(sequences, fasta_file)
                alignment = load_aligned_seqs(alignment_path, moltype='protein')
                alignment = sanitize_alignment(alignment)
                alignment = _trim_alignment(alignment)
                if alignment is None:
                    logger.warning('Dropped alignment %s after trimming (no columns remain).', alignment_path)
                    continue
                alignment = _drop_gappy_sequences(alignment)
                if alignment is None:
                    logger.warning('Dropped alignment %s due to excessive sequence gaps.', alignment_path)
                    continue
                logger.info(f'Loaded alignment from {alignment_path}')
                if fan_alignment:
                    if not all([fan_alignment_window, fan_alignment_overlap]):
                        raise click.UsageError("If --fan-alignment is set, both --fan-alignment-window and --fan-alignment-overlap must be provided.")
                    for sub in iter_moving_window_alignment(alignment, fan_alignment_window, fan_alignment_overlap):
                        sub2 = _trim_alignment(sub)
                        sub2 = _drop_gappy_sequences(sub2)
                        if sub2 is None:
                            continue
                        yield sub2
                else:
                    yield alignment
        else:
            alignment = load_aligned_seqs(sequences, moltype='protein')
            alignment = sanitize_alignment(alignment)
            alignment = _trim_alignment(alignment)
            if alignment is None:
                logger.warning('Input alignment dropped after trimming (no columns remain).')
                return
            alignment = _drop_gappy_sequences(alignment)
            if alignment is None:
                logger.warning('Input alignment dropped due to excessive sequence gaps.')
                return
            logger.info(f'Loaded alignment from {sequences}')
            if fan_alignment:
                if not all([fan_alignment_window, fan_alignment_overlap]):
                    raise click.UsageError("If --fan-alignment is set, both --fan-alignment-window and --fan-alignment-overlap must be provided.")
                for sub in iter_moving_window_alignment(alignment, fan_alignment_window, fan_alignment_overlap):
                    sub2 = _trim_alignment(sub)
                    sub2 = _drop_gappy_sequences(sub2)
                    if sub2 is None:
                        continue
                    yield sub2
            else:
                yield alignment

    # Build a streaming job generator for (di)graph construction
    def _construction_job_iter():
        """
        Generate construction jobs with optional sequence blocking.
        If max_seqs_per_block is set, split each alignment into blocks
        of up to that many sequences and emit a barrier between blocks so
        the streaming constructor waits until all jobs in a block finish
        before moving on.
        """
        from cogent3.core.alignment import make_aligned_seqs
        from fitness_landscape.utils import iter_moving_window_alignment

        job_counter = 0
        for alignment in _iter_sub_alignments():
            # Chunk by sequences if requested
            names = list(alignment.names)
            if max_seqs_per_block and max_seqs_per_block > 0 and len(names) > max_seqs_per_block:
                blocks = [names[i:i+max_seqs_per_block] for i in range(0, len(names), max_seqs_per_block)]
            else:
                blocks = [names]

            for b_idx, block_names in enumerate(blocks):
                block_map = {n: str(alignment.get_gapped_seq(n)) for n in block_names}
                block_aln = make_aligned_seqs(block_map, moltype='protein')

        def _emit_job(seq_aln):
            nonlocal job_counter
            job_counter += 1
            try:
                _n = len(list(seq_aln.names))
            except Exception:
                _n = None
            _lbl = f"phylo size={_n if _n is not None else '?'} directed={bool(directed_landscape)}"
            if directed_landscape:
                return {
                    "sequences": seq_aln,
                    "digraph_type": "phylogenetic",
                    "replacement_matrix": list(replacement_matrix),
                    "model_fitting": model_fitting,
                    "_compute_phylo_embeddings": compute_phylo_embeddings,
                    "embedding_domain": embedding_domain,
                    "_log_progress": log_progress,
                    "_job_id": job_counter,
                    "_total_jobs": None,
                    "_job_label": _lbl,
                    "_nested_construction_parallel": False,
                    "_lightweight_nodes": True,
                    "_hard_ancestors": True,
                }
            else:
                return {
                    "sequences": seq_aln,
                    "graph_type": "phylogenetic",
                    "replacement_matrix": list(replacement_matrix),
                    "model_fitting": model_fitting,
                    "_compute_phylo_embeddings": compute_phylo_embeddings,
                    "embedding_domain": embedding_domain,
                    "_log_progress": log_progress,
                    "_job_id": job_counter,
                    "_total_jobs": None,
                    "_job_label": _lbl,
                    "_nested_construction_parallel": False,
                    "_lightweight_nodes": True,
                    "_hard_ancestors": True,
                }

        # Emit fanned or whole-block jobs
        if fan_alignment:
            if not all([fan_alignment_window, fan_alignment_overlap]):
                raise click.UsageError("If --fan-alignment is set, both --fan-alignment-window and --fan-alignment-overlap must be provided.")
            for sub in iter_moving_window_alignment(block_aln, fan_alignment_window, fan_alignment_overlap):
                sub2 = _trim_alignment(sub)
                sub2 = _drop_gappy_sequences(sub2)
                if sub2 is None:
                    continue
                yield _emit_job(sub2)
        else:
            yield _emit_job(block_aln)

        # Insert barrier after each block to force sequential block processing
        if b_idx < len(blocks) - 1:
            yield {"_barrier": True}

    bernoulli_beta = BernoulliBeta(alpha0=bernoulli_beta_alpha0, alpha1=bernoulli_beta_alpha1)
    
    sampler_kwargs = {
        "bernoulli_beta": bernoulli_beta,
        "alpha": rjmcmc_alpha,
        "birth_prior_gamma": birth_gamma_prior,
        "birth_death_prob": birth_step_prob,
        "burn_in": burn_in_samples,
        "samples": total_samples,
        "thin": sample_thin,
        "auto_anchor": auto_anchor,
        "cosine_anchor_threshold": anchor_cosine_threshold,
        "seed": seed,
        "local_cpu_chains": local_cpu_chains,
        # Optional sliding-window controls for hierarchical aligner
        # If omitted, core defaults will enable sliding windows.
        **({"local_window_shifts": local_window_shifts} if local_window_shifts is not None else {}),
        **({"local_window_size": local_window_size} if local_window_size is not None else {}),
        **({"local_window_stride": local_window_stride} if local_window_stride is not None else {}),
        # propagate checkpointing into hierarchical aligner
        "_checkpoint_dir": None,  # filled below
        "_checkpoint_interval": checkpoint_interval,
        "_resume_checkpoint": resume_checkpoint,
    }

    # Default checkpoint directory (for CLI runs) if not explicitly provided
    if checkpoint_dir:
        ckpt_dir = Path(checkpoint_dir)
    else:
        out_p = Path(output)
        ckpt_dir = out_p.parent / f"{out_p.stem}_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sampler_kwargs["_checkpoint_dir"] = str(ckpt_dir)
    logger.info('Checkpointing: dir=%s interval=%ss', str(ckpt_dir), str(checkpoint_interval))

    # Optionally avoid Ray during per-alignment construction
    if sequential_construction:
        landscapes = []
        for j in _construction_job_iter():
            seqs = j.pop('sequences')
            if directed_landscape:
                from fitness_landscape.core.landscape import DirectedFitnessLandscape
                landscapes.append(DirectedFitnessLandscape.from_sequences(sequences=seqs, **j))
            else:
                from fitness_landscape.core.landscape import FitnessLandscape
                landscapes.append(FitnessLandscape.from_sequences(sequences=seqs, **j))
        superscape = FitnessSuperscape(landscapes=landscapes, posterior_prob_cutoff=posterior_threshold, **sampler_kwargs)
    else:
        logger.info('Launching streaming parallel construction (Ray)')
        superscape = FitnessSuperscape.from_streaming_construction(
            constructor_type=('directed' if directed_landscape else 'undirected'),
            construction_job_iter=_construction_job_iter(),
            posterior_prob_cutoff=posterior_threshold,
            _meta_cpu_chains=meta_cpu_chains,
            _fresh_worker_per_job=ray_fresh_worker,
            _show_progress=log_progress,
            _construct_checkpoint_dir=str(ckpt_dir),
            **sampler_kwargs
        )

    superscape.save(Path(output))
    logger.info('Superscape saved to %s', output)
    logger.info('phylo-superscape: end wall=%.2fs cpu=%.2fs', time.perf_counter()-t0, time.process_time()-c0)


@cli.command()
# Reading and writing results
@click.option('--sequences', required=True, type=click.Path(exists=True), help='Path to the input alignment file.')
@click.option('--output', required=True, type=click.Path(), help='Path to save the serialized FitnessLandscape object (.pkl).')

# Phylogenetic inference controls
@click.option('--replacement-matrix', multiple=True, default=['LG'], help='Replacement matrix/matrices for IQ-TREE model selection (e.g., LG). Can be provided multiple times.')
@click.option('--model-fitting/--no-model-fitting', default=True, help='Whether to fit and select the best model (AICc) from the provided set.')

# Embeddings for node attributes
@click.option('--compute-phylo-embeddings/--no-compute-phylo-embeddings', default=True, help='Compute embeddings for extant and ancestral sequences to attach to nodes.')
@click.option('--embedding-domain', type=click.Choice(['ohe', 'plm']), default='ohe', help='Embedding domain for auto-computed embeddings.')
@click.option('--plm-model-name', type=str, default='facebook/esm2_t6_8M_UR50D', help='PLM model to use when embedding-domain=plm.')
@click.option('--plm-batch-size', type=int, default=64, help='Batch size for PLM embeddings.')
@click.option('--plm-device', type=str, default=None, help='Device for PLM embeddings (e.g., cpu or cuda).')
@click.option('--compute-hamming-edges/--no-compute-hamming-edges', default=True, help='Compute expected Hamming edge weights after phylo reconstruction.')
@click.option('--lightweight-nodes/--no-lightweight-nodes', default=False, help='Return lightweight nodes (drop gapped_arr) to reduce memory.')
@click.option('--hard-ancestors/--no-hard-ancestors', default=False, help='Collapse ancestral SoftSequence to hard argmax sequence to reduce memory.')
# Logging (mirror other commands)
@click.option('--log-file', type=click.Path(), default=None, help='Optional log file path.')
@click.option('--log-level', type=click.Choice(['DEBUG','INFO','WARNING','ERROR']), default='INFO', show_default=True)
@click.option('--log-progress', is_flag=True, default=False, help='Enable verbose progress logging.')
@click.option('--log-prefix', type=str, default=None, help='If --log-file not provided, derive a log filename using this prefix next to --output.')
def phylo_landscape(sequences,
                    output,
                    replacement_matrix,
                    model_fitting,
                    compute_phylo_embeddings,
                    embedding_domain,
                    plm_model_name,
                    plm_batch_size,
                    plm_device,
                    compute_hamming_edges,
                    lightweight_nodes,
                    hard_ancestors,
                    log_file,
                    log_level,
                    log_progress,
                    log_prefix):
    """
    Construct a single phylogenetic FitnessLandscape and save it to disk.

    This avoids the parallel Superscape flow to help isolate issues with
    Ray workers and focuses on a single phylogenetic reconstruction + ASR.
    """
    # Logger setup (same scheme as others)
    logger = logging.getLogger('fitness_landscape')
    if not log_file and log_prefix:
        ts = time.strftime('%Y%m%d-%H%M%S')
        seq_base = os.path.basename(str(sequences).rstrip('/'))
        out_p = Path(output)
        base_dir = out_p.parent
        log_name = f"{log_prefix}_{seq_base}_{ts}_{os.getpid()}.log"
        log_file = str(base_dir / log_name)
    if log_file:
        logger.setLevel(getattr(logging, log_level))
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == fh.baseFilename for h in logger.handlers):
            logger.addHandler(fh)

    t0 = time.perf_counter(); c0 = time.process_time()
    logger.info('phylo-landscape: start')

    # Build the phylogenetic landscape (undirected) using the alignment
    logger.info('Loading alignment and constructing phylogenetic landscape')
    landscape = FitnessLandscape.from_sequences(
        sequences=Path(sequences),
        graph_type='phylogenetic',
        _compute_phylo_embeddings=compute_phylo_embeddings,
        embedding_domain=embedding_domain,
        replacement_matrix=list(replacement_matrix),
        model_fitting=model_fitting,
        model_name=plm_model_name,
        batch_size=plm_batch_size,
        device=plm_device,
        _compute_hamming_edges=compute_hamming_edges,
        _lightweight_nodes=lightweight_nodes,
        _hard_ancestors=hard_ancestors,
    )

    # Persist to disk
    logger.info('Saving landscape to %s', output)
    landscape.save(Path(output))
    logger.info('phylo-landscape: end wall=%.2fs cpu=%.2fs', time.perf_counter()-t0, time.process_time()-c0)


@cli.command()
# Reading/writing
@click.option('--sequences', required=True, type=click.Path(exists=True), help='Path to the input FASTA file.')
@click.option('--output', required=True, type=click.Path(), help='Path to save the serialized FitnessLandscape object (.pkl).')

# Diffusion graph parameters
@click.option('--k', type=int, default=50, show_default=True, help='kNN neighbors for pre-filtering.')
@click.option('--t', type=int, default=5, show_default=True, help='Diffusion power (steps).')
@click.option('--tau', type=float, default=1.0, show_default=True, help='Score temperature for kernel conversion.')
@click.option('--connectivity-threshold', type=float, default=1e-4, show_default=True, help='Connectivity threshold for diffused matrix.')
@click.option('--backend', type=click.Choice(['auto','faiss','balltree']), default='auto', show_default=True, help='kNN backend.')
@click.option('--index-type', type=click.Choice(['hnsw','flat','ivf']), default='hnsw', show_default=True, help='FAISS index type.')
@click.option('--faiss-metric', type=click.Choice(['ip','l2']), default='ip', show_default=True, help='FAISS metric (ip recommended).')
@click.option('--include-self', is_flag=True, default=False, help='Include self edges in kNN graph.')
@click.option('--use-gpu', is_flag=True, default=False, help='Use GPU for FAISS (if available for selected index).')
@click.option('--hnsw-M', 'hnsw_m', type=int, default=32, show_default=True, help='HNSW M parameter.')
@click.option('--cpus', type=int, default=1, show_default=True, help='CPUs per scoring task (used internally).')
@click.option('--compute-hamming-edges/--no-compute-hamming-edges', default=True, help='Compute expected Hamming edge weights after phylo reconstruction.')

# Embeddings
@click.option('--compute-embeddings/--no-compute-embeddings', default=True, help='Compute node embeddings (ohe or plm).')
@click.option('--embedding-domain', type=click.Choice(['ohe','plm']), default='ohe', show_default=True, help='Embedding domain for node attributes.')
@click.option('--plm-model-name', type=str, default='facebook/esm2_t6_8M_UR50D', show_default=True, help='PLM model when embedding-domain=plm.')
@click.option('--plm-batch-size', type=int, default=64, show_default=True, help='Batch size for PLM embeddings.')
@click.option('--plm-device', type=str, default=None, help='Device for PLM embeddings (e.g., cpu or cuda).')

# Logging
@click.option('--log-file', type=click.Path(), default=None, help='Optional log file path.')
@click.option('--log-level', type=click.Choice(['DEBUG','INFO','WARNING','ERROR']), default='INFO', show_default=True)
@click.option('--log-progress', is_flag=True, default=False, help='Enable verbose progress logging.')
@click.option('--log-prefix', type=str, default=None, help='If --log-file not provided, derive a log filename using this prefix next to --output.')
def evol_diffusion_landscape(sequences,
                             output,
                             k,
                             t,
                             tau,
                             connectivity_threshold,
                             backend,
                             index_type,
                             faiss_metric,
                             include_self,
                             use_gpu,
                             hnsw_m,
                             cpus,
                             compute_hamming_edges,
                             compute_embeddings,
                             embedding_domain,
                             plm_model_name,
                             plm_batch_size,
                             plm_device,
                             log_file,
                             log_level,
                             log_progress,
                             log_prefix):
    """
    Construct a single evolutionary diffusion FitnessLandscape and save it to disk.

    This is analogous to phylo-landscape but uses the diffusion/evolutionary
    scoring over a provided embedding space (OHE or PLM) and kNN prefiltering.
    """
    # Logger (derive filename from prefix if requested and not explicitly provided)
    logger = logging.getLogger('fitness_landscape')
    if not log_file and log_prefix:
        ts = time.strftime('%Y%m%d-%H%M%S')
        seq_base = os.path.basename(str(sequences).rstrip('/'))
        out_p = Path(output)
        base_dir = out_p.parent
        log_name = f"{log_prefix}_{seq_base}_{ts}_{os.getpid()}.log"
        log_file = str(base_dir / log_name)
    if log_file:
        logger.setLevel(getattr(logging, log_level))
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == fh.baseFilename for h in logger.handlers):
            logger.addHandler(fh)
    t0 = time.perf_counter(); c0 = time.process_time()
    logger.info('evol-diffusion-landscape: start')

    # Read sequences: accept a FASTA file or a directory of FASTA files.
    # When a directory is provided, combine all FASTA files into one dataset.
    try:
        seq_path = Path(sequences)
        if seq_path.is_dir():
            fasta_files = sorted([p for p in seq_path.iterdir() if p.suffix.lower() in {'.fasta', '.fa', '.fas'}])
            if not fasta_files:
                raise click.UsageError(f"The directory '{sequences}' contains no FASTA files.")
            seqs = []
            for fp in fasta_files:
                logger.info('Reading FASTA: %s', fp)
                seqs.extend(fasta_to_prot20_sequences(fp, strict=False))
            logger.info('Combined sequences from %d FASTA files (total=%d)', len(fasta_files), len(seqs))
        else:
            seqs = fasta_to_prot20_sequences(seq_path, strict=False)
    except Exception as e:
        raise click.UsageError(str(e))

    if not seqs:
        raise click.UsageError('No sequences parsed from the provided input.')

    # Embeddings
    if not compute_embeddings:
        raise click.UsageError('--no-compute-embeddings is not supported for this constructor; provide embeddings or enable computation.')

    if embedding_domain == 'ohe':
        # If variable sequence lengths, fall back to a length-invariant
        # composition embedding to avoid OHE stacking errors.
        lengths = {len(s) for s in seqs}
        if len(lengths) == 1:
            from fitness_landscape.core.graph import _encode_multiallele
            E, _ = _encode_multiallele(seqs)
        else:
            logger.warning('Sequences have non-uniform lengths (%s). Falling back to composition embeddings for kNN prefilter.', sorted(lengths))
            A = [str(a).upper() for a in PROT_20]
            amap = {a: i for i, a in enumerate(A)}
            import numpy as _np
            E = _np.zeros((len(seqs), len(A)), dtype=_np.float32)
            for r, s in enumerate(seqs):
                arr = getattr(s, 'to_array', lambda: [])()
                counts = _np.zeros(len(A), dtype=_np.float32)
                tot = 0
                for sym in arr:
                    j = amap.get(str(sym).upper())
                    if j is not None:
                        counts[j] += 1.0
                        tot += 1
                if tot > 0:
                    counts /= float(tot)
                else:
                    counts[:] = 1.0 / len(A)
                E[r] = counts
    else:
        E = _compute_embeddings_from_sequences(seqs, model_name=plm_model_name, batch_size=plm_batch_size, device=plm_device)

    logger.info('embeddings shape=%s', getattr(E, 'shape', None))

    # Build diffusion-evolution graph
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

    landscape = FitnessLandscape.from_graph(G)

    # Save
    
    landscape.save(Path(output))

    logger.info('Landscape saved to %s', output)
    logger.info('evol-diffusion-landscape: end wall=%.2fs cpu=%.2fs', time.perf_counter()-t0, time.process_time()-c0)

if __name__ == '__main__':
    cli()
