from networkx.generators import directed
import click
import os
import json
from pathlib import Path
from fitness_landscape.core.superscape import FitnessSuperscape
from fitness_landscape.core.landscape import DirectedFitnessLandscape
from fitness_landscape.utils import moving_window_alignment
from fitness_landscape.graph_matching.latent_alignment import BernoulliBeta
from cogent3 import load_aligned_seqs

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
@click.option('--embedding-domain', required=False, type=str, default="plm", help='The embedding domain to use for sequence embeddings. Options are "plm" for protein language model embeddings or "onehot" for one-hot encoded embeddings.')

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
@click.option('--embedding-domain', required=False, type=str, default='ohe', help='The embedding domain to use for sequence embeddings. Options are "plm" for protein language model embeddings or "ohe" for one-hot encoded embeddings.')
@click.option('--seed', required=False, type=int, default=None, help='Seed for the random number generator to make results reproducible.')

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
                     local_cpu_chains):
    """
    Constructs and aligns phylogenetic fitness landscapes in parallel.
    """
    
    sub_alignments = []
    
    if os.path.isdir(sequences):
        fasta_files = [f for f in os.listdir(sequences) if f.endswith(('.fasta', '.fa', '.fas'))]
        if not fasta_files:
            raise click.UsageError(f"The directory '{sequences}' contains no FASTA files.")
            
        for fasta_file in fasta_files:
            alignment_path = os.path.join(sequences, fasta_file)
            alignment = load_aligned_seqs(alignment_path)
            
            if fan_alignment:
                if not all([fan_alignment_window, fan_alignment_overlap]):
                    raise click.UsageError("If --fan-alignment is set, both --fan-alignment-window and --fan-alignment-overlap must be provided.")
                sub_alignments.extend(moving_window_alignment(alignment, fan_alignment_window, fan_alignment_overlap))
            else:
                sub_alignments.append(alignment)

    else: 
        alignment = load_aligned_seqs(sequences)
        if fan_alignment:
            if not all([fan_alignment_window, fan_alignment_overlap]):
                raise click.UsageError("If --fan-alignment is set, both --fan-alignment-window and --fan-alignment-overlap must be provided.")
            sub_alignments = moving_window_alignment(alignment, fan_alignment_window, fan_alignment_overlap)
        else:
            sub_alignments.append(alignment)

    construction_jobs = [{
        "sequences": sub_alignment,
        "graph_type": "phylogenetic",
        "_compute_phylo_embeddings": compute_phylo_embeddings,
        "embedding_domain": embedding_domain,
    } for sub_alignment in sub_alignments]

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

    superscape = FitnessSuperscape.from_parallel_construction(
        constructor_type='undirected' if not directed_landscape else 'phylogenetic',
        construction_jobs=construction_jobs,
        _meta_cpu_chains=meta_cpu_chains,
        **sampler_kwargs
    )

    superscape.save(Path(output))



if __name__ == '__main__':
    cli()