from pathlib import Path
from typing import (
    Union,
    Dict,
    List,
    Literal
)
import numpy as np
import networkx as nx
from cogent3 import (
    load_aligned_seqs,
    load_tree,
    get_app
)
from cogent3.core.alignment import Alignment, make_aligned_seqs
try:
    from cogent3.core.tree import PhyloNode
except Exception:
    from cogent3 import PhyloNode  # fallback if available
import piqtree
from piqtree import model_finder
import math
import sys
import os
import tempfile
import subprocess as _subprocess
from ..core.sequence import (
    BaseNumpySequence,
    SoftSequence
)
from .._const import (
    PROT_20,
    ALPHABET_21
)
from ..utils import sanitize_alignment

class ASRConstructor:
    """
    Class to manage phylogenetic tree inference and reconstruction of
    internal node probability distributions.

    Attributes
    ----------
        alignment : Alignment or Path
        The alignment to use for phylogenetic reconstruction.
    
    phylogenetic_tree : PhyloNode or Path, default=`None`
        A precomputed phylogenetic tree. If None, the tree is inferred.
    
    replacement_matrix : List, defualt=`NQ_pfam`
        The replacement matrix used for tree-search. Multiple can be
        provided to fit the ML model. If `NQ_pfam`, output will be
        directed. If LG, output will be undirected.

    model_fitting : bool, default=`False`
        Boolean for whether or not to include ML model fitting and
        parameterisation.
    
    _reconstruct_ancestral_states : bool, default=`True`
        Boolean for whether ancestral states are inferred. 

    """
    def __init__(self,
                alignment: Union[Alignment, Path],
                phylogenetic_tree: Union[PhyloNode, Path] = None,
                model_fitting: bool = False,
                replacement_matrix: List = ['NQ.pfam'],
                _reconstruct_ancestral_states: bool = True,
                _log_progress: bool = False) -> None:

        # Load alignment
        if isinstance(alignment, Path):
            self.alignment = load_aligned_seqs(alignment,
                                            format="fasta",
                                            moltype="protein")
        elif isinstance(alignment, Alignment):
            self.alignment = alignment
        
        else:
            raise ValueError("Alignment must be either Path or Alignment.")
        
        # Keep the alignment as provided by cogent3 for maximum compatibility
        # Construct alignment header list.
        self.tip_names = self.alignment.names
        self._log_progress = _log_progress

        # Construct boolean gap alignment.
        self._boolean_gap_alignment = self.alignment.get_gap_array()
        self.boolean_gap_alignment = {}
        for node, arr in zip(self.tip_names, self._boolean_gap_alignment):
            self.boolean_gap_alignment[node] = arr

        # If no phylogenetic tree, infer one using IQ-TREE via piqtree.
        if phylogenetic_tree is None:
            self.build_tree(replacement_matrix=replacement_matrix,
                            model_fitting=model_fitting)
        
        elif phylogenetic_tree is not None:
            
            if isinstance(phylogenetic_tree, PhyloNode):
                self.phylogenetic_tree = phylogenetic_tree
            
            elif isinstance(phylogenetic_tree, Path):
                self.phylogenetic_tree = load_tree(phylogenetic_tree)
            
            else:
                raise ValueError("Phylogenetic tree must be either Path or PhyloNode.")
            
        # Infer ancestral states if requested
        if _reconstruct_ancestral_states:
            self.reconstruct_ancestral_states()
            self.ancestral_reconstruction_bool()
        
    def build_tree(self,
                   replacement_matrix: List[str] = ['NQ.pfam'],
                   model_fitting: bool = True,
                   _model_override: str = None) -> None: 
        """
        Method to construct a phylogenetic tree using the piqtree
        Python binding.
        
        Parameters:
        -----------
        model : List
            The substitution models to use.

        model_fitting : bool, default=`True`
            Whether to fit the ML model by AICC.
        
        _model_override : str, default=`None`
            A IQTREE convention model string to override the model. 
        """
        if not hasattr(self, 'alignment'):
            raise ValueError('expected alignment attribute.')
        
        # Keep alignment as-is for maximum compatibility.

        import logging as _logging
        _logger = _logging.getLogger('fitness_landscape')
        if self._log_progress:
            _logger.info('ASR.build_tree: start (models=%s, fit=%s)', replacement_matrix, model_fitting)
        if _model_override is not None:
            # Expect a string model spec for newer piqtree; coerce if needed
            model = str(_model_override)
        
        elif model_fitting:
            # Try model finding; if it fails (e.g., tiny datasets), fall back to first provided model
            try:
                result = model_finder(self.alignment, model_set=set(replacement_matrix))
                # Choose model by AICc; newer piqtree returns a string
                model = str(getattr(result, 'best_aicc', result))
            except Exception:
                model = str(replacement_matrix[0] if replacement_matrix else 'LG')
        
        # Use just the provided replacement matrix.
        else:
            if len(replacement_matrix) > 1:
                raise ValueError('Expected only single replacement matrix.')
            model = str(replacement_matrix[0])

        # Minimal wrapper around piqtree.build_tree
        def _try_build(_model: str, *_):
            # Minimal call mirroring piqtree docs; add a deterministic seed
            try:
                return piqtree.build_tree(self.alignment, _model, rand_seed=1)
            except TypeError:
                # Fallback if rand_seed not supported in older versions
                return piqtree.build_tree(self.alignment, _model)

        # Concurrency/stability guards: cap threaded libs and isolate working dir
        env_cap = {
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
        }
        old_env = {k: os.environ.get(k) for k in env_cap}
        os.environ.update(env_cap)

        # Choose working directory for IQ-TREE logs
        cwd0 = os.getcwd()
        _cleanup_tmp = True
        log_root = os.environ.get('FITNESS_LANDSCAPE_IQTREE_LOG_DIR')
        if log_root:
            try:
                os.makedirs(log_root, exist_ok=True)
                tag = f"iqtree_run_{int(os.getpid())}_{int(np.random.randint(1e9))}"
                tmpdir = os.path.join(log_root, tag)
                os.makedirs(tmpdir, exist_ok=True)
                _cleanup_tmp = False
                if self._log_progress:
                    _logger.info('ASR.build_tree: logging IQ-TREE run to %s', tmpdir)
            except Exception:
                # Fall back to temporary directory if creation fails
                tmp_ctx = tempfile.TemporaryDirectory(prefix='iqtree_run_')
                tmpdir = tmp_ctx.name
                _cleanup_tmp = True
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix='iqtree_run_')
            tmpdir = tmp_ctx.name
        primary_err = None
        try:
            os.chdir(tmpdir)
            # Persist the current alignment and run metadata for offline debugging
            try:
                # Write gapped alignment to FASTA
                aln_fp = os.path.join(tmpdir, 'alignment_gapped.fasta')
                with open(aln_fp, 'w') as _f:
                    for nm in self.tip_names:
                        seq = str(self.alignment.get_gapped_seq(nm))
                        _f.write(f">{nm}\n{seq}\n")
                # Write simple metadata
                meta_fp = os.path.join(tmpdir, 'run_meta.txt')
                with open(meta_fp, 'w') as _mf:
                    _mf.write(f"models={replacement_matrix}\n")
                    _mf.write(f"model_fitting={model_fitting}\n")
                    _mf.write(f"backend=iqtree\n")
                    try:
                        import piqtree as _pt
                        _mf.write(f"piqtree_version={getattr(_pt, '__version__', '?')}\n")
                    except Exception:
                        pass
                    _mf.write(f"n_tips={len(self.tip_names)}\n")
                    try:
                        first = self.tip_names[0]
                        _mf.write(f"L={len(str(self.alignment.get_gapped_seq(first)))}\n")
                    except Exception:
                        pass
            except Exception:
                pass
            # Optional process isolation to avoid native state leaks across runs
            if os.environ.get('FITNESS_LANDSCAPE_IQTREE_SUBPROC', '').lower() in {'1','true','yes'}:
                aln_fp = os.path.join(tmpdir, 'alignment_gapped.fasta')
                out_pkl = os.path.join(tmpdir, 'tree.pkl')
                code = (
                    "import sys, pickle; "
                    "import piqtree; "
                    "from cogent3 import load_aligned_seqs; "
                    "aln=load_aligned_seqs(sys.argv[1], moltype='protein'); "
                    "model=sys.argv[2]; "
                    "try:\n"
                    "    t=piqtree.build_tree(aln, model, rand_seed=1)\n"
                    "except TypeError:\n"
                    "    t=piqtree.build_tree(aln, model)\n"
                    "with open(sys.argv[3],'wb') as f: pickle.dump(t,f)"
                )
                child_env = os.environ.copy()
                child_env.update(env_cap)
                try:
                    _subprocess.check_call([os.environ.get('PYTHON', sys.executable), '-c', code, aln_fp, str(model), out_pkl], env=child_env)
                    with open(out_pkl, 'rb') as f:
                        phylogenetic_tree = pickle.load(f)
                except Exception as e:
                    primary_err = e
                    # Fallback: try with the first provided model or LG
                    try:
                        first = str(replacement_matrix[0]) if replacement_matrix else 'LG'
                        _subprocess.check_call([os.environ.get('PYTHON', sys.executable), '-c', code, aln_fp, first, out_pkl], env=child_env)
                        with open(out_pkl, 'rb') as f:
                            phylogenetic_tree = pickle.load(f)
                    except Exception as e2:
                        details = []
                        details.append(f"primary model={model!r} error={type(primary_err).__name__}: {primary_err}")
                        details.append(f"fallback model={first!r} error={type(e2).__name__}: {e2}")
                        pv = getattr(piqtree, '__version__', '?')
                        details.append(f"piqtree_version={pv}")
                        details.append("Set FITNESS_LANDSCAPE_IQTREE_LOG_DIR to preserve IQ-TREE logs for debugging")
                        msg = ("IQ-TREE (piqtree) tree building failed.\n" + "\n".join(details))
                        try:
                            with open(os.path.join(tmpdir, 'error.txt'), 'w') as _ef:
                                _ef.write(msg)
                        except Exception:
                            pass
                        raise RuntimeError(msg) from e2
            else:
                try:
                    phylogenetic_tree = _try_build(model)
                except Exception as e:
                    primary_err = e
                    # Fallback 1: retry with first provided model or LG
                    try:
                        first = str(replacement_matrix[0]) if replacement_matrix else 'LG'
                        phylogenetic_tree = _try_build(first)
                    except Exception as e2:
                        # Provide detailed error context including original exceptions
                        details = []
                        details.append(f"primary model={model!r} error={type(primary_err).__name__}: {primary_err}")
                        details.append(f"fallback model={first!r} error={type(e2).__name__}: {e2}")
                        pv = getattr(piqtree, '__version__', '?')
                        details.append(f"piqtree_version={pv}")
                        details.append("Set FITNESS_LANDSCAPE_IQTREE_LOG_DIR to preserve IQ-TREE logs for debugging")
                        msg = (
                            "IQ-TREE (piqtree) tree building failed.\n" +
                            "\n".join(details)
                        )
                        # Persist error text to the log directory for inspection
                        try:
                            with open(os.path.join(tmpdir, 'error.txt'), 'w') as _ef:
                                _ef.write(msg)
                        except Exception:
                            pass
                        raise RuntimeError(msg) from e2
        finally:
            # Restore environment and working directory; cleanup temp dir
            try:
                os.chdir(cwd0)
            except Exception:
                pass
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            # Only cleanup if we created a temporary directory
            if _cleanup_tmp:
                try:
                    tmp_ctx.cleanup()
                except Exception:
                    pass

        self.phylogenetic_tree = phylogenetic_tree
        if self._log_progress:
            _logger.info('ASR.build_tree: complete')

    def reconstruct_ancestral_states(self, model_name: str = "WG01") -> None:
        """
        Minimal placeholder ASR: assigns a uniform amino-acid posterior (L,20)
        for each internal node. Avoids plugin manager to prevent duplicate
        registration issues in interactive contexts.
        """
        import logging as _logging
        _logger = _logging.getLogger('fitness_landscape')
        if self._log_progress:
            _logger.info('ASR.reconstruct_ancestral_states: start (placeholder uniform)')
        # Alignment length (columns): use gapped sequence length of first tip
        first = self.tip_names[0]
        L = len(str(self.alignment.get_gapped_seq(first)))
        # Uniform posterior over 20 AAs
        uniform_post = np.full((L, len(PROT_20)), 1.0 / len(PROT_20), dtype=float)
        # Collect internal nodes from the tree
        child_parent = self.phylogenetic_tree.child_parent_map()
        nodes = set(child_parent.keys()) | set(child_parent.values())
        # Build mapping from PhyloNode -> posterior array
        post_map: Dict = {}
        for node in nodes:
            # tips are strings in self.tip_names; internal nodes are PhyloNode
            if getattr(node, 'name', None) in set(self.tip_names):
                continue
            post_map[node] = uniform_post.copy()
        self.asr_posterior_arr = post_map
        if self._log_progress:
            _logger.info('ASR.reconstruct_ancestral_states: complete')

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
        
    def construct_dag(self,
                      graph_type: Literal['undirected', 'directed'] = 'undirected') -> Union[nx.DiGraph, nx.Graph]:
        """
        Method to construct a directed acyclic graph (DAG) from the
        phylogenetic tree and the alignment.

        Parameters
        ----------
        type : str, default=`undirected`
            Whether the graph will be directed or undirected.

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
        if graph_type != 'undirected' and graph_type != 'directed':
            raise ValueError(f"Expected `graph_type` parameter to be `directed` or `undirected`, found {graph_type}")
        
        import logging as _logging
        _logger = _logging.getLogger('fitness_landscape')
        if self._log_progress:
            _logger.info('ASR.construct_dag: start (graph_type=%s)', graph_type)
        G = (nx.Graph() if graph_type == 'undirected' else nx.DiGraph())
        
        for child, parent in self.phylogenetic_tree.child_parent_map().items():
            G.add_edge(parent, child)

        for tip in self.tip_names:
            
            hard_seq = BaseNumpySequence(self.alignment.get_gapped_seq(tip),
                                            alphabet=ALPHABET_21)

            gapped_mat   = hard_seq.to_one_hot()            # (L, 21), 0/1
            ungapped_mat = hard_seq.remove_gap_arr()

            # Collect ungapped sequence for base.
            hard_seq = BaseNumpySequence.from_one_hot(ungapped_mat,
                                                      alphabet=PROT_20)

            G.nodes[tip].update(
                sequence=hard_seq,
                gapped_arr=gapped_mat,
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
            
            # (L,20)
            # Remove gaps where posterior_probability is less than 0.5.
            post = soft_seq.remove_gap_arr()
            soft_seq = SoftSequence(post,
                                    alphabet=PROT_20,
                                    hard_rule="argmax")
            
            G.nodes[anc].update(
                sequence=soft_seq,
                gapped_arr=gapped_post,)
        if self._log_progress:
            _logger.info('ASR.construct_dag: complete (nodes=%d, edges=%d)', G.number_of_nodes(), G.number_of_edges())
        return G
