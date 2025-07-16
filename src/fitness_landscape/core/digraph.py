from pathlib import Path
from typing import Union, Dict
import numpy as np
import networkx as nx
from cogent3 import load_aligned_seqs, Alignment, PhyloNode, load_tree, get_app, get_moltype
import piqtree
from piqtree import Model
from piqtree.model import AaModel
import math
from cogent3.util.table import Table
from .sequence import SoftSequence, BaseNumpySequence

PROT = get_moltype("protein")
PROT_20 = [aa for aa in PROT.alphabet if aa != 'U']
ALPHABET_21 = PROT_20 + ["gap"]

class ASRLandscapeConstructor:
    """
    Class to manage phylogenetic tree inference and loading.

    Attributes
    ----------
    alignment : Path or Alignment
        The path to a fasta formatted sequence alignment or an
        initialised cogent3.Alignment object.
    
    phylogenetic_tree : Path or PhyloNode, default=`None`
        The path to a newick treefile or an initialised PhyloNode
        object.

    ancestral_states : Path, Table, default=`None`
        Pre-computed and loaded ancestral states as a cogent3 tabular
        result.
    """
    def __init__(self,
                 alignment: Union[Path, Alignment],
                 phylogenetic_tree: Union[Path, PhyloNode] = None,
                 ancestral_states: Table = None,
                 _reconstruct_phylogeny: bool = True,
                 _reconstruct_ancestral_states: bool = True) -> None:


        # Load alignment
        if isinstance(alignment, Path):
            self.alignment = load_aligned_seqs(alignment,
                                               format="fasta",
                                               moltype="protein")
        elif isinstance(alignment, Alignment):
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
        if phylogenetic_tree is None and _reconstruct_phylogeny:
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
        
        """
        G = nx.DiGraph()
        for child, parent in self.phylogenetic_tree.child_parent_map().items():
            G.add_edge(parent, child)

        for tip in self.tip_names:
            hard_seq = BaseNumpySequence(self.alignment.get_gapped_seq(tip), alphabet=ALPHABET_21)
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
    
# TODO: Evolutionary velocity connectivity