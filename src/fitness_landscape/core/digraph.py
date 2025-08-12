from pathlib import Path
from typing import Union, Dict, List
import numpy as np
import networkx as nx
from cogent3 import load_aligned_seqs, ArrayAlignment, PhyloNode, load_tree, get_app
import piqtree
from piqtree import Model
from piqtree.model import AaModel
import math
from cogent3.util.table import Table
from .sequence import SoftSequence, BaseNumpySequence
from .._const import ALPHABET_21, PROT_20
from .._sub_matrices import nq_pfam
from sklearn.neighbors import NearestNeighbors
from ..utils import calculate_gapped_soft_score
from ..embedding.particle_sampler import (
    EvolutionParticleSampler,
    SequenceGenerator,
    ParentSelector,
    TopPSampler,
    ESMEmbedder
)
from softalign.soft_alignment import align_soft_sequences


def create_phylo_digraph(sequences: Union[Path, ArrayAlignment],
                         phylogenetic_tree: Union[Path, PhyloNode] = None,
                         ancestral_states: Table = None) -> nx.DiGraph:
    """
    Factory function to create a Directed acyclic graph using
    phylogenetic inference and ancestral sequence reconstruction. 

    Parameters
    ----------
    alignment : Path or Alignment
        The alignment of extant sequences to use for ASR and
        phylogenetic infernece.
    
    phylogenetic_tree : Path or PhyloNode, default=`None`
        A precomputed phylogenetic tree. If `None`, tree topology is
        constructed on the fly.
    
    ancestral_states: Table, default=`None`
        Precomputed ancestral states.

    Returns
    -------
    G : nx.DiDraph
        The Directed graph output.
    """
    
    class ASRConstructor:
        """
        Helper class to manage phylogenetic tree inference and loading.
        """
        def __init__(self,
                    alignment,
                    phylogenetic_tree,
                    ancestral_states,
                    _reconstruct_ancestral_states=True) -> None:

            # Load alignment
            if isinstance(alignment, Path):
                self.alignment = load_aligned_seqs(alignment,
                                                format="fasta",
                                                moltype="protein")
            elif isinstance(alignment, ArrayAlignment):
                self.alignment = alignment
            
            else:
                raise ValueError("Alignment must be either Path or Alignment.")
            
            # Construct alignment header list.
            self.tip_names = self.alignment.names

            # Construct boolean gap alignment.
            self._boolean_gap_alignment = self.alignment.get_gap_array()
            self.boolean_gap_alignment = {}
            for node, arr in zip(self.tip_names, self._boolean_gap_alignment):
                self.boolean_gap_alignment[node] = arr

            # If no phylogenetic tree, infer one in piqtree.
            if phylogenetic_tree is None:
                self.build_tree()
            
            elif phylogenetic_tree is not None:
                
                if isinstance(phylogenetic_tree, PhyloNode):
                    self.phylogenetic_tree = phylogenetic_tree
                
                elif isinstance(phylogenetic_tree, Path):
                    self.phylogenetic_tree = load_tree(phylogenetic_tree)
                
                else:
                    raise ValueError("Phylogenetic tree must be either Path or PhyloNode.")
                
            # If ancestral states load or infer.
            if ancestral_states is None and _reconstruct_ancestral_states:
                self.reconstruct_ancestral_states()
                self.ancestral_reconstruction_bool()
            
            elif ancestral_states is not None:
                if isinstance(ancestral_states, Table):
                    self.asr_posterior_arr = ancestral_states
                
                else:
                    raise ValueError("Ancestral states must be Table.")
                
                self.ancestral_reconstruction_bool()
            
        def build_tree(self) -> None: 
            """
            Method to construct a phylogenetic tree using the piqtree
            Python binding.
            
            Parameters:
            -----------
            model : str
                The substitution model to use.
            """
            assert hasattr(self, 'alignment')

            # Use non-stationary model.
            model = Model(AaModel.NQ_pfam)

            phylogenetic_tree = piqtree.build_tree(
                self.alignment, 
                model
            )
            self.phylogenetic_tree = phylogenetic_tree

        def reconstruct_ancestral_states(self,
                                        model_name: str = "WG01") -> None:
            """
            Method to reconstruct ancestral states by empirical Bayesian
            (using the marginal algorithm) in cogent3.
            
            Parameters:
            -----------
            model_name : str
                The name of the substitution model.
            """
            model_app = get_app("model", model_name, tree=self.phylogenetic_tree)
            model_result = model_app(self.alignment)
            asr_app = get_app("ancestral_states")
            self.asr_posterior_arr = asr_app(model_result)

        def _two_state_transition_probs(self,
                                        branch_length: float,
                                        rate=1.0) -> np.ndarray:
            """
            Helper function to return a 2 x 2 JC-like 2-state model. Index
            0 denotes a gap, index 1 denotes no gap.

            Parameters
            ----------
            branch_length : float
                The branch-lenth.
            
            rate : float, default=`1.0`
                The replacement rate.
            
            Returns
            -------
            np.ndarray
                The transition proability matrix. 
            """
            e = math.exp(-2.0 * rate * branch_length)
            p_same = 0.5 + 0.5 * e
            p_diff = 0.5 - 0.5 * e
            return np.array([
                [p_same, p_diff],
                [p_diff, p_same]
            ], dtype=float)

        def _postorder_likelihoods_boolean(self,
                                        node: PhyloNode,
                                        alignment_bool: np.ndarray,
                                        tip_name_to_index: set,
                                        node_likelihoods: bool = None) -> Dict:
            """
            Recursively compute (L x 2) conditional likelihoods for each
            node in a post-order traversal.

            Parameters
            ----------
            node : cogent3.PhyloNode
                The current node (root or internal).
            alignment_bool : np.ndarray
                Boolean array, shape (N_tips, L), where `True` indicates a
                gap and `False` indicates no-gap.
            tip_name_to_index : Dict
                Maps node.name to row index in alignment_bool, for tips.
            node_likelihoods : dict
                Node likelihood dict.

            Returns
            -------
            node_likelihoods : dict
                Updated mapping { node_name : (L x 2) array } for each tip
                and internal node encountered.
            """
            if node_likelihoods is None:
                node_likelihoods = {}

            # If this node is a tip/leaf:
            if not node.children:  
                tip_idx = tip_name_to_index[node.name]
                # shape (L,)
                leaf_bools = alignment_bool[tip_idx, :]  
                L_node = np.zeros((leaf_bools.shape[0], 2), dtype=float)
                L_node[leaf_bools, 0] = 1.0
                L_node[~leaf_bools, 1] = 1.0

                node_likelihoods[node.name] = L_node
                return node_likelihoods

            for child in node.children:
                self._postorder_likelihoods_boolean(
                    child,
                    alignment_bool,
                    tip_name_to_index,
                    node_likelihoods
                )

            L_node = np.ones((alignment_bool.shape[1], 2), dtype=float)

            for child in node.children:
                L_child = node_likelihoods[child.name]  # shape (L,2)
                branch_len = child.length
                P = self._two_state_transition_probs(branch_len)  # shape (2,2)

                new_L_child = np.zeros_like(L_child)  # shape (L,2)
                for parent_state in (0, 1):
                    # Weighted sum over child states j=0..1
                    weighted_sum = (
                        L_child[:, 0] * P[parent_state, 0]
                        + L_child[:, 1] * P[parent_state, 1]
                    )
                    new_L_child[:, parent_state] = L_node[:, parent_state] * weighted_sum
                L_node = new_L_child

            row_sums = L_node.sum(axis=1)         # shape (L,)
            nonzero = (row_sums != 0)
            L_node[nonzero, :] /= row_sums[nonzero][:, None]

            node_likelihoods[node.name] = L_node

            return node_likelihoods

        def ancestral_reconstruction_bool(self) -> None:
            """
            Method to run a binary ancestral state reconstruction over a
            two-state discrete extant trait representin insertions.
            """
            tip_name_to_index = {name: i for i, name in enumerate(self.tip_names)}

            self.node_likelihoods = self._postorder_likelihoods_boolean(self.phylogenetic_tree,
                                                                self._boolean_gap_alignment,
                                                                tip_name_to_index)
            
        def construct_dag(self) -> nx.DiGraph:
            """
            Method to construct a directed acyclic graph (DAG) from the
            phylogenetic tree and the alignment.

            Returns
            -------
            nx.DiGraph
                A directed acyclic graph where nodes are the tips and
                internal nodes of the phylogenetic tree, and edges are
                directed from parent to child nodes. Each node contains
                the following attributes:
                - `sequence`: The sequence at that node, either as a
                `BaseNumpySequence` or `SoftSequence`.
                - `fitness`: Fitness value, initialized to NaN.
                - `gapped_arr`: A (L, 21) array representing the gapped
                sequence in one-hot encoding.
                - `ungapped_arr`: A (L, 20) array representing the ungapped
                sequence in one-hot encoding.
            """
            G = nx.DiGraph()
            for child, parent in self.phylogenetic_tree.child_parent_map().items():
                G.add_edge(parent, child)

            for tip in self.tip_names:
                
                hard_seq = BaseNumpySequence(self.alignment.get_gapped_seq(tip),
                                              alphabet=ALPHABET_21)

                gapped_mat   = hard_seq.to_one_hot()            # (L, 21), 0/1
                ungapped_mat = hard_seq.remove_gap_arr()

                G.nodes[tip].update(
                    sequence=hard_seq,
                    fitness=np.nan,
                    gapped_arr=gapped_mat,
                    ungapped_arr=ungapped_mat,
                )

            for anc in set(G.nodes) - set(self.tip_names):
                
                # (L, 20) AA posterior
                post = np.array(self.asr_posterior_arr[anc])
                
                # (L, 2)  gap posterior
                gap  = self.node_likelihoods[anc]
                
                # (L, 21)
                #Use conditional probability logic to combine.
                gapped_post = SoftSequence.compute_conditional_gap_dist(aa_post_dist=post,
                                                                        gap_post_dist=gap)        
                
                soft_seq = SoftSequence(
                    aa_posterior=post,
                    gap_posterior=self.node_likelihoods[anc],
                    alphabet=ALPHABET_21,
                    hard_rule="argmax",
                )

                G.nodes[anc].update(
                    sequence=soft_seq,
                    fitness=np.nan,
                    gapped_arr=gapped_post,
                    ungapped_arr=post,
                )
            return G
        
    constructor = ASRConstructor(sequences, phylogenetic_tree, ancestral_states)
    digraph = constructor.construct_dag()
    return digraph

#TODO: Add emergence time masking.
def create_evol_diffusion_digraph(sequences: List[BaseNumpySequence],
                                             embeddings: np.ndarray,
                                             replacement_matrix: np.ndarray = nq_pfam,
                                             k: int = 50,
                                             t: int = 5,
                                             tau: float = 1.0,
                                             connectivity_threshold: float = 1e-4,
                                             **kwargs) -> nx.DiGraph:
    """
    Constructs a diffusion graph by scoring standard alignments with an
    asymmetric non-equilibrium replacement matrix.

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of sequence in the landscape. 
    
    embeddings : np.ndarray
        Sequence embeddings indexed by the entry in `sequences`.
    
    k : int, default=50
        The number of neighbours to use for kNN pre-filtering.
    
    t : int, default=5
        The number of diffusion steps taken.
    
    tau : float, default=1.0
        The temperature parameter used to smooth the distance kernel.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The constructed graph.
    """
    # Type check alphabet first
    for seq in sequences:
        if seq.alphabet != PROT_20:
            raise ValueError("Sequence alpbahet must be PROT_20 for all entries.")
    
    n_sequences = len(sequences)
    if n_sequences == 0:
        return nx.DiGraph()

    # Find kNN in embedding space to identify candidate pairs
    # Should scale in O(N*k)
    
    # Update value of k if too large.
    if k > n_sequences - 1:
        k = n_sequences - 1

    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn.fit(embeddings)
    _, neighbor_indices = nn.kneighbors(embeddings)
    
    pairs_to_align = []
    for i in range(n_sequences):
        for j_idx in neighbor_indices[i]:
            if i != j_idx:
                
                pair = (i, j_idx)
                pairs_to_align.append(pair)
    
    # Remove duplicate pairs before aligning
    pairs_to_align = sorted(list(set(pairs_to_align)))
    
    # Make kernel matrix
    kernel_matrix = np.zeros((n_sequences, n_sequences))
    
    # Iterate through pairs of sequences to align
    for i, j in pairs_to_align:
        # Get sequence arrays, handling both SoftSequence and BaseNumpySequence
        seq_i = sequences[i]
        arr_i = seq_i.posterior if isinstance(seq_i, SoftSequence) else seq_i.to_one_hot()

        seq_j = sequences[j]
        arr_j = seq_j.posterior if isinstance(seq_j, SoftSequence) else seq_j.to_one_hot()

                
        alignment, _ = align_soft_sequences(sequences=[arr_i, arr_j],
                                            alphabet=PROT_20)
        
        score = calculate_gapped_soft_score(aligned_seq1 = alignment[0],
                                            aligned_seq2 = alignment[1],
                                            q = replacement_matrix)
        
        # Tau controls "sharpness" of kernel distances.
        kernel_matrix[i, j] = np.exp(score / tau)

    # Proceed with diffusion and graph construction (same as before)
    np.fill_diagonal(kernel_matrix, 0)
    
    row_sums = kernel_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = kernel_matrix / row_sums
    
    # Compute diffusion steps
    diffused_matrix = np.linalg.matrix_power(transition_matrix, t)
    
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(n_sequences))
    
    rows, cols = np.where(diffused_matrix > connectivity_threshold)
    
    for i, j in zip(rows, cols):
        if i != j:
            digraph.add_edge(i, j, weight=diffused_matrix[i, j])
        
    for i, seq in enumerate(sequences):
        digraph.nodes[i]['sequence'] = seq
        
    return digraph


def create_particle_filter_digraph(sequences: List[BaseNumpySequence],
                                   n_samples: int,
                                   traj_length: int,
                                   batch_size: int,
                                   max_state_size: int,
                                   _emb_array_key: str = 'emb_array',
                                   temperature: float = 1.0,
                                   top_p: float = 0.9,
                                   **kwargs) -> nx.DiGraph:
    """
    Factory function to create a directed graph using a Gibbs sampling
    approach based on a protein language model.

    Parameters
    ----------
    seed_sequences : List[BaseNumpySequence]
        The initial sequences to start the simulation.
    n_samples : int
        The number of child sequences to generate from each parent.
    traj_length : int
        The number of steps in the evolutionary trajectory.
    batch_size : int
        The batch size for the sequence generator.
    max_state_size : int
        The maximum number of parent nodes to select at each step.
    hmm_file : str, optional
        Path to an HMM file to use for the sequence space attractor.
    embedding_attribute : str, default='representation'
        The node attribute key for embeddings.
    temperature : float, default=1.0
        The temperature for the Top-p sampler.
    top_p : float, default=0.9
        The top-p value for the Top-p sampler.

    Returns
    -------
    DirectedFitnessLandscape
        The constructed directed fitness landscape.
    """
    selector = ParentSelector(max_state_size=max_state_size)
    embedder = ESMEmbedder(model_name=kwargs.get('model_name', "facebook/esm2_t6_8M_UR50D"))
    sampler = TopPSampler(temperature=temperature, top_p=top_p)
    generator = SequenceGenerator(embedder=embedder,sampler=sampler, batch_size=batch_size)

    evolution_exp = EvolutionParticleSampler(generator=generator,
                                             selector=selector,
                                             n_samples=n_samples,
                                             traj_length=traj_length)

    # Convert BaseNumpySequence to strings for sampling.
    evolution_exp.initialize(seed_sequences=[seq.to_str() for seq in sequences])
    evolution_exp.run()
    digraph = evolution_exp.G
    
    return digraph

    # TODO: Evolutionary velocity connectivity