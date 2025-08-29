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

@click.group()
def cli():
    """A python tool for fitness landscape analysis."""
    pass

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
@click.option('--compute-phylo-embeddings', required=False, is_flag=True, default=True, help='Boolean flag to indicate whether sequences should be embedded in a latent space.')
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
@click.option('--sequential-construction', is_flag=True, default=False, help='Construct each landscape sequentially (avoids Ray during construction).')
@click.option('--seed', required=False, type=int, default=None, help='Seed for the random number generator to make results reproducible.')

# Logging
@click.option('--log-file', required=False, type=click.Path(), help='Path to write a detailed log file for this run.')
@click.option('--log-level', required=False, type=click.Choice(['DEBUG','INFO','WARNING','ERROR']), default='INFO', help='Log level for the detailed log file.')
@click.option('--log-progress', is_flag=True, default=False, help='Enable verbose progress logging within constructors.')

# RJMCMC cpu chains
@click.option('--meta-cpu-chains', required=False, type=int, default=os.cpu_count(), help='Number of CPU chains to use for the meta-alignment step in hierarchical alignment.')
@click.option('--local-cpu-chains', required=False, type=int, default=(os.cpu_count()//10 if os.cpu_count()//10 > 1 else 1), help='Number of CPU chains to use for each parallel local alignment chain.')

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
                     seed,
                     meta_cpu_chains,
                     local_cpu_chains,
                     sequential_construction,
                     log_file,
                     log_level,
                     log_progress):
    """
    Constructs and aligns phylogenetic fitness landscapes in parallel.
    """
    
    # Configure logging if requested
    logger = logging.getLogger('fitness_landscape')
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

    sub_alignments = []
    
    if os.path.isdir(sequences):
        fasta_files = [f for f in os.listdir(sequences) if f.endswith(('.fasta', '.fa', '.fas'))]
        if not fasta_files:
            raise click.UsageError(f"The directory '{sequences}' contains no FASTA files.")
            
        for fasta_file in fasta_files:
            alignment_path = os.path.join(sequences, fasta_file)
            alignment = load_aligned_seqs(alignment_path, moltype='protein')
            alignment = sanitize_alignment(alignment)
            logger.info(f'Loaded alignment from {alignment_path}')
            
            if fan_alignment:
                if not all([fan_alignment_window, fan_alignment_overlap]):
                    raise click.UsageError("If --fan-alignment is set, both --fan-alignment-window and --fan-alignment-overlap must be provided.")
                sub_alignments.extend(moving_window_alignment(alignment, fan_alignment_window, fan_alignment_overlap))
            else:
                sub_alignments.append(alignment)

    else: 
        alignment = load_aligned_seqs(sequences, moltype='protein')
        alignment = sanitize_alignment(alignment)
        logger.info(f'Loaded alignment from {sequences}')
        if fan_alignment:
            if not all([fan_alignment_window, fan_alignment_overlap]):
                raise click.UsageError("If --fan-alignment is set, both --fan-alignment-window and --fan-alignment-overlap must be provided.")
            sub_alignments = moving_window_alignment(alignment, fan_alignment_window, fan_alignment_overlap)
            logger.info(f'Fanned into {len(sub_alignments)} windows (window={fan_alignment_window}, overlap={fan_alignment_overlap})')
        else:
            sub_alignments.append(alignment)
            logger.info('Single alignment mode (no fanning)')

    # Build job specs for (di)graph construction
    total_jobs = len(sub_alignments)
    if directed_landscape:
        construction_jobs = [{
            "sequences": sub_alignment,
            "digraph_type": "phylogenetic",
            "replacement_matrix": list(replacement_matrix),
            "model_fitting": model_fitting,
            "_compute_phylo_embeddings": compute_phylo_embeddings,
            "embedding_domain": embedding_domain,
            "_log_progress": log_progress,
            "_job_id": i + 1,
            "_total_jobs": total_jobs,
        } for i, sub_alignment in enumerate(sub_alignments)]
    else:
        construction_jobs = [{
            "sequences": sub_alignment,
            "graph_type": "phylogenetic",
            "replacement_matrix": list(replacement_matrix),
            "model_fitting": model_fitting,
            "_compute_phylo_embeddings": compute_phylo_embeddings,
            "embedding_domain": embedding_domain,
            "_log_progress": log_progress,
            "_job_id": i + 1,
            "_total_jobs": total_jobs,
        } for i, sub_alignment in enumerate(sub_alignments)]

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
    }

    # Optionally avoid Ray during per-alignment construction (stability for IQ-TREE)
    if sequential_construction:
        landscapes = []
        if directed_landscape:
            for job in construction_jobs:
                from fitness_landscape.core.landscape import DirectedFitnessLandscape
                j = dict(job)
                sequences = j.pop('sequences')
                logger.info('Sequential directed construction started')
                ts = time.perf_counter(); cs = time.process_time()
                landscapes.append(DirectedFitnessLandscape.from_sequences(sequences=sequences, **j))
                logger.info(f'Seq directed construction finished in wall={time.perf_counter()-ts:.2f}s cpu={time.process_time()-cs:.2f}s')
        else:
            for job in construction_jobs:
                from fitness_landscape.core.landscape import FitnessLandscape
                j = dict(job)
                sequences = j.pop('sequences')
                logger.info('Sequential undirected construction started')
                ts = time.perf_counter(); cs = time.process_time()
                landscapes.append(FitnessLandscape.from_sequences(sequences=sequences, **j))
                logger.info(f'Seq undirected construction finished in wall={time.perf_counter()-ts:.2f}s cpu={time.process_time()-cs:.2f}s')
        superscape = FitnessSuperscape(
            landscapes=landscapes,
            **sampler_kwargs,
        )
    else:
        logger.info(f'Launching {len(construction_jobs)} parallel construction jobs')
        superscape = FitnessSuperscape.from_parallel_construction(
            constructor_type=('directed' if directed_landscape else 'undirected'),
            construction_jobs=construction_jobs,
            _meta_cpu_chains=meta_cpu_chains,
            _show_progress=log_progress,
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
def phylo_landscape(sequences,
                    output,
                    replacement_matrix,
                    model_fitting,
                    compute_phylo_embeddings,
                    embedding_domain,
                    plm_model_name,
                    plm_batch_size,
                    plm_device):
    """
    Construct a single phylogenetic FitnessLandscape and save it to disk.

    This avoids the parallel Superscape flow to help isolate issues with
    Ray workers and focuses on a single phylogenetic reconstruction + ASR.
    """
    # Build the phylogenetic landscape (undirected) using the alignment
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
    )

    # Persist to disk
    with open(output, 'wb') as f:
        pickle.dump(landscape, f)


if __name__ == '__main__':
    cli()
