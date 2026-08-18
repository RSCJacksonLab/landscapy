"""Infer phylogenies and reconstruct ancestral sequences."""

from pathlib import Path
from typing import (
    Union,
    Dict,
    List,
    Literal,
    Optional,
    Set
)
import numpy as np
import networkx as nx
from .._optional import require_optional

require_optional("cogent3", extra="phylogeny", purpose="phylogenetic reconstruction")
require_optional("piqtree", extra="phylogeny", purpose="phylogenetic reconstruction")
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
import pickle
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
    """Infer an undirected phylogeny and ancestral sequence posteriors.

    Parameters
    ----------
    alignment : cogent3.Alignment or pathlib.Path
        Protein alignment or FASTA path.
    phylogenetic_tree : cogent3.core.tree.PhyloNode or pathlib.Path, optional
        Precomputed tree or Newick path. If omitted, infer a tree.
    model_fitting : bool, default=False
        Select a substitution model by AICc when supported by the backend.
    replacement_matrix : list of str, default=['NQ.pfam']
        Candidate protein substitution models.
    phylo_backend : {'iqtree', 'cogent_nj'}, default='cogent_nj'
        Tree-inference backend used when no tree is supplied.
    _dist_calc : {'paralinear', 'pdist', 'hamming'}, default='pdist'
        Pairwise-distance calculator for neighbour joining.
    reconstruct_ancestral_states : bool, default=True
        Fit ancestral amino-acid posterior distributions for internal nodes.
    _log_progress : bool, default=False
        Emit progress logs for long-running inference steps.

    Attributes
    ----------
        alignment : Alignment or Path
        The alignment to use for phylogenetic reconstruction.
    
    phylogenetic_tree : PhyloNode or Path, default=`None`
        A precomputed phylogenetic tree. If None, the tree is inferred.
    
    replacement_matrix : List, defualt=`NQ_pfam`
        The replacement matrix used for tree-search. Multiple models can be
        provided for model fitting. The constructed topology is always
        undirected in the 0.9 publication API.

    model_fitting : bool, default=`False`
        Boolean for whether or not to include ML model fitting and
        parameterisation.
    
    reconstruct_ancestral_states : bool, default=`True`
        Boolean for whether ancestral states are inferred. 

    """
    def __init__(self,
                alignment: Union[Alignment, Path],
                phylogenetic_tree: Union[PhyloNode, Path] = None,
                model_fitting: bool = False,
                replacement_matrix: List = ['NQ.pfam'],
                phylo_backend: str = 'cogent_nj',
                _dist_calc: Literal['paralinear', 'pdist', 'hamming'] = 'pdist',
                reconstruct_ancestral_states: bool = True,
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
        self._replacement_models = tuple(replacement_matrix) if replacement_matrix is not None else tuple()
        self._tree_model_name: Optional[str] = None
        self._reconstruct_ancestral_states = bool(reconstruct_ancestral_states)
        self.asr_posterior_arr: dict = {}
        self.node_likelihoods: dict = {}
        self._sequence_length: int = 0
        if self.tip_names:
            first_tip = self.tip_names[0]
            try:
                self._sequence_length = len(self.alignment.get_gapped_seq(first_tip))
            except Exception:
                try:
                    self._sequence_length = len(str(self.alignment.get_gapped_seq(first_tip)))
                except Exception:
                    self._sequence_length = 0

        # Construct boolean gap alignment.
        self._boolean_gap_alignment = self.alignment.get_gap_array()
        self.boolean_gap_alignment = {}
        for node, arr in zip(self.tip_names, self._boolean_gap_alignment):
            self.boolean_gap_alignment[node] = arr

        # If no phylogenetic tree, infer one using IQ-TREE via piqtree.
        if phylogenetic_tree is None:
            self.build_tree(replacement_matrix=replacement_matrix,
                            model_fitting=model_fitting,
                            phylo_backend=phylo_backend,
                            _dist_calc=_dist_calc)
        
        elif phylogenetic_tree is not None:
            
            if isinstance(phylogenetic_tree, PhyloNode):
                self.phylogenetic_tree = phylogenetic_tree
            
            elif isinstance(phylogenetic_tree, Path):
                self.phylogenetic_tree = load_tree(phylogenetic_tree)
            
            else:
                raise ValueError("Phylogenetic tree must be either Path or PhyloNode.")
            
        # Infer ancestral states if requested
        if self._reconstruct_ancestral_states:
            self.reconstruct_ancestral_states()
            self.ancestral_reconstruction_bool()
        
    def build_tree(self,
                   replacement_matrix: List[str] = ['NQ.pfam'],
                   model_fitting: bool = True,
                   _model_override: str = None,
                   _dist_calc: Literal['paralinear', 'pdist', 'hamming'] = 'pdist',
                   phylo_backend: Literal['iqtree', 'cogent_nj'] = 'cogent_nj') -> None: 
        """
        Method to construct a phylogenetiphyloc tree using the piqtree
        Python binding.
        
        Parameters
        ----------
        replacement_matrix : list of str, default=['NQ.pfam']
            Substitution models considered by the selected backend.

        model_fitting : bool, default=`True`
            Whether to fit the ML model by AICC.
        
        _model_override : str, default=`None`
            A IQTREE convention model string to override the model. 

        _dist_calc : {'paralinear', 'pdist', 'hamming'}, default='pdist'
            The distance calculation to use in computing neighbors and 
            distance matrices for neighbor-joining algorithms.

        phylo_backend : {'iqtree', 'cogent_nj'}, default='cogent_nj'
            The phylogenetic reconstruction backend to use. 

        """
        if not hasattr(self, 'alignment'):
            raise ValueError('expected alignment attribute.')
        
        # Keep alignment as-is for maximum compatibility.

        import logging as _logging
        _logger = _logging.getLogger('fitness_landscape')
        if self._log_progress:
            _logger.info('ASR.build_tree: start (models=%s, fit=%s)', replacement_matrix, model_fitting)
        # Normalize backend selection
        backend = (phylo_backend or 'cogent_nj').lower()
        if backend not in {'iqtree', 'cogent_nj'}:
            raise ValueError(f"Unsupported phylo_backend={phylo_backend!r}; use 'iqtree' or 'cogent_nj'.")

        # Determine model string depending on backend
        if backend == 'cogent_nj':
            if _model_override is not None:
                model = str(_model_override)
            else:
                # For cogent quick_tree backend, default to WG01 (WAG equivalent)
                model = str(replacement_matrix[0]) if replacement_matrix else 'WG01'
        else:
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
            else:
                # Use just the provided replacement matrix.
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
                    _mf.write(f"backend={backend}\n")
                    if backend == 'cogent_nj':
                        try:
                            _mf.write(f"distance_calc={_dist_calc}\n")
                        except Exception:
                            pass
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
            
            # Backend: cogent3 neighbor-joining per cookbook
            if backend == 'cogent_nj':
                try:
                    from cogent3.phylo import nj as _c3_nj
                except Exception as _imp_err:
                    try:
                        # Fallback: Application API
                        nj_app = get_app('nj')
                        phylogenetic_tree = nj_app(self.alignment)
                    except Exception as e2:
                        primary_err = e2
                        details = [f"cogent_nj error={type(e2).__name__}: {e2}"]
                        details.append("Attempted imports: 'from cogent3.phylo import nj' and get_app('nj')")
                        msg = "Cogent3 NJ tree building failed.\n" + "\n".join(details)
                        try:
                            with open(os.path.join(tmpdir, 'error.txt'), 'w') as _ef:
                                _ef.write(msg)
                        except Exception:
                            pass
                        raise RuntimeError(msg) from e2
                else:
                    dists = None
                    last_err = None
                    tried: List[str] = []
                    # Try requested calculator first, then fall back to cheaper ones that
                    # avoid the numba parallel runtime (e.g. get_num_threads TypingError).
                    candidates = []
                    if _dist_calc:
                        candidates.append(_dist_calc)
                    for alt in ('paralinear', 'hamming'):
                        if alt not in candidates:
                            candidates.append(alt)
                    used_model = None
                    for calc_name in candidates:
                        try:
                            # Force serial computation to avoid numba parallel backend issues
                            dists = self.alignment.distance_matrix(calc=calc_name, parallel=False)
                        except Exception as e:
                            last_err = e
                            tried.append(f"{calc_name}: {type(e).__name__}: {e}")
                            continue
                        else:
                            used_model = calc_name
                            break

                    if dists is None:
                        primary_err = last_err or Exception('unknown distance error')
                        details = ["Failed to compute distance_matrix with candidates:"] + tried
                        msg = "Cogent3 NJ distance computation failed.\n" + "\n".join(details)
                        try:
                            with open(os.path.join(tmpdir, 'error.txt'), 'w') as _ef:
                                _ef.write(msg)
                        except Exception:
                            pass
                        raise RuntimeError(msg) from primary_err
                    try:
                        phylogenetic_tree = _c3_nj.nj(dists, show_progress=False)
                    except Exception as e:
                        primary_err = e
                        details = [f"cogent_nj error={type(e).__name__}: {e}"]
                        details.append(f"distance calc used={used_model}")
                        msg = "Cogent3 NJ tree building failed.\n" + "\n".join(details)
                        try:
                            with open(os.path.join(tmpdir, 'error.txt'), 'w') as _ef:
                                _ef.write(msg)
                        except Exception:
                            pass
                        raise RuntimeError(msg) from e

            # Backend: IQ-TREE via piqtree (in-process or subprocess)
            elif os.environ.get('FITNESS_LANDSCAPE_IQTREE_SUBPROC', '').lower() in {'1','true','yes'}:
                try:
                    _logger.warning('ASR.build_tree: using IQ-TREE via piqtree; this backend can be unstable and may segfault on some inputs/environments. Consider --phylo-backend cogent_nj for robustness.')
                except Exception:
                    pass
                aln_fp = os.path.join(tmpdir, 'alignment_gapped.fasta')
                out_pkl = os.path.join(tmpdir, 'tree.pkl')
                # IMPORTANT: keep compound statements (e.g., try:) on their own line.
                # Python does not allow a 'try' to follow a semicolon on the same line.
                code = "\n".join([
                    "import sys, pickle",
                    "import piqtree",
                    "from cogent3 import load_aligned_seqs",
                    # Read the FASTA we just wrote; be explicit about format for robustness
                    "aln = load_aligned_seqs(sys.argv[1], moltype='protein', format='fasta')",
                    "model = sys.argv[2]",
                    "try:",
                    "    t = piqtree.build_tree(aln, model, rand_seed=1)",
                    "except TypeError:",
                    "    t = piqtree.build_tree(aln, model)",
                    "with open(sys.argv[3], 'wb') as f:",
                    "    pickle.dump(t, f)",
                ])
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
                    _logger.warning('ASR.build_tree: using IQ-TREE via piqtree; this backend can be unstable and may segfault on some inputs/environments. Consider --phylo-backend cogent_nj for robustness.')
                except Exception:
                    pass
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
        try:
            self._tree_model_name = str(model)
        except Exception:
            self._tree_model_name = None
        if self._log_progress:
            _logger.info('ASR.build_tree: complete')

    def reconstruct_ancestral_states(self, model_name: str = "WG01") -> None:
        """Run maximum-likelihood ancestral-state reconstruction.

        Run maximum-likelihood ancestral state reconstruction using cogent3's
        composable ``model`` / ``ancestral_states`` applications. Raises an
        informative exception if model fitting or posterior extraction fails.

        Parameters
        ----------
        model_name : str, default='WG01'
            Preferred Cogent3 protein substitution model. Configured tree
            models and common protein-model fallbacks are tried if it fails.

        Raises
        ------
        ValueError
            If no tree or tip sequences are available.
        RuntimeError
            If every substitution model fails or posterior output is invalid.
        """
        import logging as _logging

        _logger = _logging.getLogger('fitness_landscape')
        if not hasattr(self, 'phylogenetic_tree'):
            raise ValueError("Phylogenetic tree not set; cannot perform ASR.")
        if not self.tip_names:
            raise ValueError("Alignment contains no tip sequences; cannot perform ASR.")

        # Alignment length (with gaps) used for shape checking.
        first_tip = self.tip_names[0]
        L = len(str(self.alignment.get_gapped_seq(first_tip)))

        # Gather internal nodes (cogent3 auto-names unnamed nodes as edge.*).
        internal_nodes: List[PhyloNode] = []

        def _collect_internal(node: PhyloNode) -> None:
            children = getattr(node, 'children', []) or []
            if children:
                internal_nodes.append(node)
                for child in children:
                    _collect_internal(child)

        _collect_internal(self.phylogenetic_tree)

        post_map: Dict[Union[PhyloNode, str], np.ndarray] = {}

        def _store(node: PhyloNode, arr: np.ndarray) -> None:
            name = getattr(node, 'name', None)
            if name:
                post_map[str(name)] = arr
            post_map[node] = arr

        # Build a list of model candidates. Some tree-building backends expose
        # models that cogent3 cannot fit (e.g. NQ.pfam); skip failures.
        candidates: List[str] = []
        seen: Set[str] = set()

        def _add_candidate(cand: Optional[str]) -> None:
            if not cand:
                return
            key = str(cand)
            if key not in seen:
                candidates.append(key)
                seen.add(key)

        _add_candidate(model_name)
        _add_candidate(getattr(self, '_tree_model_name', None))
        for cand in getattr(self, '_replacement_models', ()):
            _add_candidate(cand)
        # Sensible fallbacks recognised by cogent3 for proteins.
        for fallback in ("WG01", "LG", "WAG"):
            _add_candidate(fallback)

        chosen_model = None
        model_result = None
        last_exc = None
        for cand in candidates:
            try:
                model_app = get_app('model', cand, tree=self.phylogenetic_tree)
            except Exception as err:
                last_exc = err
                continue
            try:
                model_result = model_app(self.alignment)
            except Exception as err:
                last_exc = err
                continue
            chosen_model = cand
            break

        if model_result is None:
            detail = f" (last error: {last_exc!r})" if last_exc else ""
            raise RuntimeError(
                "ASR.reconstruct_ancestral_states: failed to fit a substitution model "
                f"from candidates {candidates}.{detail}"
            ) from last_exc

        try:
            asr_app = get_app('ancestral_states')
            asr_result = asr_app(model_result)
        except Exception as err:
            raise RuntimeError(
                "ASR.reconstruct_ancestral_states: cogent3 ancestral state inference failed."
            ) from err

        missing_named: List[str] = []

        for node in internal_nodes:
            node_name = getattr(node, 'name', None)
            site_dict = None
            if node_name:
                site_dict = asr_result.get(str(node_name))
            if site_dict is None:
                missing_named.append(str(node_name) if node_name else repr(node))
                continue

            positions = list(site_dict.keys())
            positions.sort()
            try:
                arr = np.stack(
                    [np.asarray(site_dict[pos].to_array(), dtype=float) for pos in positions],
                    axis=0
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ASR.reconstruct_ancestral_states: failed to coerce posterior for node "
                    f"{node_name or repr(node)}."
                ) from exc

            if arr.shape[0] != L:
                raise ValueError(
                    f"ASR.reconstruct_ancestral_states: posterior length mismatch for node "
                    f"{node_name or repr(node)} (expected {L}, observed {arr.shape[0]})."
                )
            if arr.shape[1] != len(PROT_20):
                raise ValueError(
                    f"ASR.reconstruct_ancestral_states: posterior width mismatch for node "
                    f"{node_name or repr(node)} (expected {len(PROT_20)}, observed {arr.shape[1]})."
                )

            _store(node, arr)

        if missing_named:
            raise KeyError(
                "ASR.reconstruct_ancestral_states: cogent3 ancestral_states did not emit "
                f"posteriors for nodes: {', '.join(missing_named)}."
            )

        self.asr_posterior_arr = post_map
        self._asr_model_name = chosen_model

        if self._log_progress:
            _logger.info(
                "ASR.reconstruct_ancestral_states: complete (model=%s, internal_nodes=%d)",
                chosen_model,
                len(internal_nodes),
            )

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
        
    def construct_topology(self) -> nx.Graph:
        """
        Construct an undirected topology from the phylogenetic tree.

        Returns
        -------
        nx.Graph
            An undirected graph whose nodes are the tips and internal
            nodes of the phylogenetic tree. Each node contains the
            following attributes:
            - `sequence`: The sequence at that node, either as a
            `BaseNumpySequence` or `SoftSequence`.
            - `fitness`: Fitness value, initialized to NaN.
            - `gapped_arr`: A (L, 21) array representing the gapped
            sequence in one-hot encoding.
            - `ungapped_arr`: A (L, 20) array representing the ungapped
            sequence in one-hot encoding.
        """
        import logging as _logging
        _logger = _logging.getLogger('fitness_landscape')
        if self._log_progress:
            _logger.info('ASR.construct_topology: start')
        G = nx.Graph()
        
        for child, parent in self.phylogenetic_tree.child_parent_map().items():
            G.add_edge(parent, child)

        seq_length: int = self._sequence_length
        for tip in self.tip_names:
            
            hard_seq = BaseNumpySequence(self.alignment.get_gapped_seq(tip),
                                            alphabet=ALPHABET_21)

            gapped_mat   = hard_seq.to_one_hot()            # (L, 21), 0/1
            ungapped_mat = hard_seq.remove_gap_arr()
            seq_length = gapped_mat.shape[0]

            # Collect ungapped sequence for base.
            hard_seq = BaseNumpySequence.from_one_hot(ungapped_mat,
                                                      alphabet=PROT_20)

            G.nodes[tip].update(
                sequence=hard_seq,
                gapped_arr=gapped_mat,
                asr_placeholder=False,
            )

        if seq_length <= 0:
            raise ValueError("Unable to determine sequence length from supplied alignment.")

        aa_dim = len(PROT_20)
        placeholder_aa = np.full((seq_length, aa_dim), 1.0 / aa_dim, dtype=float)
        placeholder_gap = np.zeros((seq_length, 2), dtype=float)
        placeholder_gap[:, 1] = 1.0  # favour 'no gap'

        for anc in set(G.nodes) - set(self.tip_names):
            if self._reconstruct_ancestral_states and anc in self.asr_posterior_arr:
                post = np.asarray(self.asr_posterior_arr[anc], dtype=float)
                gap = self.node_likelihoods.get(anc)
                gap = np.asarray(gap, dtype=float) if gap is not None else placeholder_gap
                gapped_post = SoftSequence.compute_conditional_gap_dist(
                    aa_post_dist=post,
                    gap_post_dist=gap
                )
                soft_seq = SoftSequence(
                    aa_posterior=post,
                    gap_posterior=gap,
                    alphabet=PROT_20,
                    hard_rule="argmax",
                )
                post = soft_seq.remove_gap_arr()
                soft_seq = SoftSequence(
                    post,
                    alphabet=PROT_20,
                    hard_rule="argmax",
                )
                G.nodes[anc].update(
                    sequence=soft_seq,
                    gapped_arr=gapped_post,
                    asr_placeholder=False,
                )
            else:
                placeholder_soft = SoftSequence.from_posteriors(
                    placeholder_aa,
                    alphabet=PROT_20,
                    gap_posterior=placeholder_gap,
                    hard_rule="argmax",
                )
                gapless_post = placeholder_soft.remove_gap_arr()
                soft_seq = SoftSequence(
                    gapless_post,
                    alphabet=PROT_20,
                    hard_rule="argmax",
                )
                G.nodes[anc].update(
                    sequence=soft_seq,
                    gapped_arr=np.asarray(placeholder_soft.posterior, dtype=float),
                    asr_placeholder=True,
                )
        if self._log_progress:
            _logger.info('ASR.construct_topology: complete (nodes=%d, edges=%d)', G.number_of_nodes(), G.number_of_edges())
        return G
