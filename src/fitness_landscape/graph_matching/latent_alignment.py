from __future__ import annotations

from scipy.sparse import csr_matrix
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import networkx as nx
from scipy.special import gammaln
from .isorank import cosine_similarity_matrix
import scipy.optimize  # heavy but only occasional

# Hyper parameter containers
@dataclass
class BernoulliBeta:
    """
    Conjugate prior for binary edges.
    """
    alpha1: float = 5  # successes (edge present in blueprint)
    alpha0: float = 1  # failures (edge absent)

    def log_marginal(self,
                     o11: int,
                     o10: int,
                     o01: int,
                     o00: int) -> float:
        """
        Method to compute the marginal likelihood for overlaps between
        permutation and latent slots, given beta prior.
        
        Parameters
        ----------
        o11 : int
            The number of matches between latent and permuted graphs.
        o10 : int
            The number of false positives.
        o01 : int
            The number of false negatives.
        o00 : int
            The number of true negatives. 
            
        Returns
        -------
        float : the log marginal likelihood.
        """

        n_matches = o11 + o00
        n_mismatches = o10 + o01

        alpha_post = self.alpha1 + n_matches
        beta_post = self.alpha0 + n_mismatches

        # log B(alpha_post, beta_post) – log B(alpha_prior, beta_prior)
        return (
            gammaln(alpha_post) + gammaln(beta_post) - gammaln(alpha_post + beta_post)
            - (gammaln(self.alpha1) + gammaln(self.alpha0)
                - gammaln(self.alpha1 + self.alpha0))
        )
    
    
    # Specifically for phylogenetic trees where true negatives will dominate.
    def log_marginal_edges(self,
                           o_success: int,
                           o_fail: int) -> float:
        """
        Log marginal method for binary edges, removing the true
        negatives due to inherent sparsity of phylogenetic trees.


        Parameters
        ----------
        o_success : int
            The number of edges present in both the latent and the
            observed graph.
        
        o_fail : int
            The number of edges present in the observed graph but
            absent in the latent graph.
        
        Returns
        -------
        float
            The log marginal likelihood of the observed edges given
            the latent graph.
        """
        a_post = self.alpha1 + o_success
        b_post = self.alpha0 + o_fail
        return (
        gammaln(a_post) + gammaln(b_post) - gammaln(a_post+b_post)
        - (gammaln(self.alpha1) + gammaln(self.alpha0)
            - gammaln(self.alpha1 + self.alpha0))
        )

#TODO: Test NormalGamma.
@dataclass
class NormalGamma:
    """
    Normal-gamma hyperprior for weighted edges.
    """
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0

    def log_marginal(self,
                     x: np.ndarray) -> float:

        """
        Method to compute the log marginal of an edge weight
        conditioned on binary present or absent condition. Assumes
        iid Nornmal with unknown mean and precision, integrated out
        in the gamma analytical solution.

        Parameters
        ----------
        x : np.ndarray
            The distance array
        
        Returns
        -------
        float
            log p(x|present) up to a constant independent of x.
        """

        n = x.size
        if n == 0:
            return 0.0
        mean = x.mean()
        ss   = ((x - mean) ** 2).sum()
        return (
        0.5 * np.log(self.kappa0 / (self.kappa0 + n))
        + self.alpha0 * np.log(self.beta0)
        - (self.alpha0 + 0.5 * n) * np.log(
            self.beta0 + 0.5 * ss + (self.kappa0 * n * (mean - self.mu0) ** 2) / (2 * (self.kappa0 + n))
        )
        + gammaln(self.alpha0 + 0.5 * n)
        - gammaln(self.alpha0)
    )

class RJMCMCAligner:
    """
    Reversible jump MCMC alignment of directed graphs. Implemented
    from https://www.nature.com/articles/s41467-025-59077-7.

    Attributes
    ----------
    graphs : list
        The list of graphs to align.
    alpha : float, default=`0.5`
        The scaling to balance topology vs. embedding similaity.
    bernouli_beta : BernouliBeta
        The bernouli beta hyperprior on condition edges.
    normal_gamma : NormalGamma
        The normal gamma hyperprior on branch lengths.
    birth_prior_gamma : float, default = `0.05`
        The shape prior on a birth event.
    burn_in : int, default=`2000`
        The burn in for the MCMC sampler.
    samples : int, default=`5000`
        The number of samples to draw in the MCMC sampling.
    thin : int, default=`50`
        The MCMC thinning parameter.
    auto_anchor : bool, default=`True`
        Boolean to auto anchor nodes to latent slots according to
        embedding cosine similarity.
    birth_death_prob : float, defulat=0.2
        The probability of a node birth or death event on an MCMC
        sample.
    cosine_anchor_threshold : float, default=`0.95`
        The cosine similairty threshold for anchoring.
    weight_key : str, default=`weight`
        The edge attribute dictionary  weight key.
    emb_key : str, default=`emb_arr`
        The node attribute dictionary embedding array key.
    directed : bool, default=`True`
        Boolean to indicate whether the graphs are directed.
    seed : int
        The random state.
    """
    
    def __init__(self,
                 graphs: List[nx.DiGraph],
                 *,
                 alpha: float = 0.5,
                 bernoulli_beta: Optional[BernoulliBeta] = None,
                 normal_gamma: Optional[NormalGamma] = None,
                 birth_prior_gamma: float = 0.05,
                 burn_in: int = 2000,
                 samples: int = 5000,
                 thin: int = 50,
                 auto_anchor: bool = True,
                 birth_death_prob: float = 0.2,
                 cosine_anchor_threshold: float = 0.95,
                 weight_key: str = 'weight',
                 emb_key: str = 'emb_arr',
                 directed: bool = True,
                 seed: Union[int, None] = None) -> None:
        
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.graphs = graphs
        self.K = len(graphs)
        self.burn_in, self.samples, self.thin, self.birth_death_prob = burn_in, samples, thin, birth_death_prob
        self.birth_gamma = birth_prior_gamma
        self.directed = directed

        # Burn in housekeeping.
        self._in_growth_phase = False
        self.trace_E = []
        self.trace_NL = []
        self.trace_edges = []
        self.accept_counts = {"swap":0, "cycle3":0, "resample":0, "birth":0, "death":0}
        self.proposal_counts = self.accept_counts.copy()

        
        # hyper‑priors
        self.bb = bernoulli_beta or BernoulliBeta()
        self.ng = normal_gamma or NormalGamma()

        # Auto-anchor nodes according to consine similarity. 
        if auto_anchor:
            auto_anchors_by_cosine(graphs=graphs,
                                   cos_threshold=cosine_anchor_threshold)

        # preprocess
        self.V: List[List] = [list(G.nodes()) for G in graphs]
        self.n: List[int] = [len(vs) for vs in self.V]

        # adjacency / weight matrices — store binary adjacency in CSR for sparse ops        
        self.W: List[csr_matrix] = []
        for G, vs in zip(graphs, self.V):
            dense = nx.to_numpy_array(G, nodelist=vs, weight=weight_key, nonedge=0.0)
            if np.isin(dense, [0.0,1.0]).all():
                binarized = (dense > 0.5).astype(np.int8)
            else:
                binarized = (dense > 0).astype(np.int8)
            self.W.append(csr_matrix(binarized))
        
        # everything in W is now 0/1 CSR
        self.binary_mode = True   

        # embeddings
        self.X = [np.vstack([np.asarray(G.nodes[v][emb_key], float) for v in vs]) for G, vs in zip(graphs, self.V)]

        # anchor handling
        # map anchor label to latent slot id (shared for all graphs)
        anchor_label_to_slot: dict[str | int, int] = {}
        next_slot = 0
        self.perm: List[np.ndarray] = []
        for k, (G, vs, Wk) in enumerate(zip(graphs, self.V, self.W)):
            pk = -np.ones(len(vs), dtype=int)
            for i, v in enumerate(vs):
                if G.nodes[v].get("anchor", False):
                    label = G.nodes[v].get("anchor_id", v)  # fall back to name
                    if label not in anchor_label_to_slot:
                        anchor_label_to_slot[label] = next_slot
                        next_slot += 1
                    pk[i] = anchor_label_to_slot[label]
            self.perm.append(pk)
        self.NL = max(max(p.max(initial=-1)+1 for p in self.perm), 1)

        # # fill non‑anchored nodes with degree ranking heuristic
        for k, (pk, Wk) in enumerate(zip(self.perm, self.W)):
            free_slots = [s for s in range(self.NL) if s not in pk]
            deg_order = np.argsort(Wk.sum(1))
            for i in deg_order:
                
                # Safeguard against popping from an empty list.
                if pk[i] == -1:
                    if not free_slots:
                        break
                    pk[i] = free_slots.pop(0)

        def _compute_Ck(pk: np.ndarray,
                        Wk: csr_matrix) -> np.ndarray:
            """
            """
            NL = self.NL
            rows, cols = Wk.nonzero()
            ik = pk[rows]; jk = pk[cols]
            mask = (ik >= 0) & (jk >= 0)
            flat = ik[mask] * NL + jk[mask]
            return np.bincount(flat, minlength=NL*NL).reshape(NL, NL)

        self._compute_Ck = _compute_Ck
        
        # per‐graph counts
        self.C_k: List[np.ndarray] = [
            self._compute_Ck(pk, Wk) for pk, Wk in zip(self.perm, self.W)
        ]
        
        # global sum (shape‐safe rebuild)
        self.C_global = self.C_k[0].copy()
        for Ck in self.C_k[1:]:
            self.C_global += Ck

        # first blueprint 
        self.L = self._gibbs_sample_blueprint()

        # bookkeeping of MCMC trace
        self._stored_L: List[np.ndarray] = []
        self._stored_pi: List[List[np.ndarray]] = [[] for _ in range(self.K)]

    #  Likelihood / posterior helpers

    def _log_prior(self) -> float:
        """
        Calculates the log-prior probability of the blueprint graph.
        This includes the size penalty for the number of latent slots.

        Returns
        -------
        size_penalty : float
            The log prior penalty on latent graph size. 
        """
        
        # Log prior penalty on size only applied after burn in.
        size_penalty = 0.0 if self._in_growth_phase else self.birth_gamma * self.NL
        # TODO: add more complex priors to penalize edge density, etc. 
        
        return -size_penalty

    def _energy_attributes(self) -> float:
        """
        Method to compute the log likelihood of the attribute.

        Returns
        -------
        attr_log_likelihood : float
            The log likelihood of the attributes given the latent
            slots.
        """
        attr_log_likelihood = 0.0
        for k in range(self.K):
            cos_sim_matrix = self._attr_cosine(k)
            for node_idx, latent_slot in enumerate(self.perm[k]):
                if latent_slot >= 0:
                    attr_log_likelihood += cos_sim_matrix[node_idx, latent_slot]
        return attr_log_likelihood

    def _energy(self) -> float:
        """
        Method to compute statistical energy as the negative log
        posterior, now balancing topology and attribute similarity.

        Returns
        -------
        float
            The negative log posterior probability of the latent
            graph given the observed graphs.
        """
        if self.binary_mode:
            topo_log_likelihood = self._energy_binary()
        else:
            topo_log_likelihood = self._energy_weighted()

        attr_log_likelihood = self._energy_attributes()

        # Combine the likelihoods using alpha.
        # alpha = 1.0 means topology only.
        # alpha = 0.0 means attributes only.
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("alpha must be between 0 and 1")

        log_likelihood = (self.alpha * topo_log_likelihood) + ((1 - self.alpha) * attr_log_likelihood)
        log_prior = self._log_prior()

        return -(log_likelihood + log_prior)

    # binary edges

    def _energy_binary(self) -> float:
        """
        Method to compute the binary edge log-likelihood using the
        Bernoulli-beta prior. Uses precomputed global counts to scale
        in O(1).
        
        Returns
        -------
        float
            The log likelihood of the binary edges given the latent
            graph.
        """
        # use precomputed global counts
        M   = int(self.C_global.sum())
        o11 = int((self.C_global * self.L).sum()) # kept edges
        o10 = M - o11 # dropped edges
        return self.bb.log_marginal_edges(o11, o10)

    
    # weighted edges
    def _energy_weighted(self) -> float:
        """
        Method to compute the weighted edge log-likelihood using the
        normal-gamma prior.

        Returns
        -------
        float
            The log likelihood of the weighted edges given the latent
            graph.
        """
        logp = 0.0
        for k, (pk, Wk) in enumerate(zip(self.perm, self.W)):
            
            # Ensure L is up-to-date for the calculation
            Lk = self.L[np.ix_(pk, pk)].astype(bool)
            present = Wk[Lk]
            absent  = Wk[~Lk]
            logp += self.ng.log_marginal(present)
            logp += self.ng.log_marginal(absent)
        
        # Returns the log marginal likelihood, NOT the energy.
        return logp

    # TODO: remove method when confirmed that Gibbs sampling is better.
    #  Blueprint update and attribute similarity helpers

    def _majority_blueprint(self) -> np.ndarray:
        """
        Method to compute the consensus blueprint.

        Returns
        -------
        L : np.ndarray
            The consensus adjacency matrix.
        """

        L = np.zeros((self.NL, self.NL), dtype=int)
        counts = np.zeros_like(L)
        for pk, Wk in zip(self.perm, self.W):
            A = (Wk > 0.5) if self.binary_mode else (Wk > 0)  # treat >0 as present
            tmp = np.zeros_like(L)
            tmp[np.ix_(pk, pk)] = A
            counts += tmp
        L[counts > self.K / 2] = 1
        # Forbid self edges
        np.fill_diagonal(L, 0)
        return L
    
    # Theoretically stronger Gibb's sampling blueprint. Don't use _majority_blueprint.
    def _gibbs_sample_blueprint(self) -> np.ndarray:
        """
        Method to sample the blueprint graph using Gibbs sampling. 
        Theoretically stronger than the majority consensus. Different
        methods implemented for directed and undirected graphs.

        Returns
        -------
        L : np.ndarray
            The sampled adjacency matrix of the latent blueprint graph.
        """
        NL, K = self.NL, self.K
        C = np.zeros((NL, NL), dtype=int)

        for pk, Wk in zip(self.perm, self.W):
            # nonzero() only visits actual edges
            rows, cols = Wk.nonzero()
            ik = pk[rows]
            jk = pk[cols]
            # only count if both endpoints assigned
            mask = (ik >= 0) & (jk >= 0)
            flat = ik[mask] * NL + jk[mask]
            
            # bincount over flattened (i,j) pairs
            counts = np.bincount(flat, minlength=NL*NL).reshape(NL, NL)
            
            if not self.directed:
                # force symmetry
                counts = np.triu(counts, 1) + np.triu(counts, 1).T
            C += counts

        # posterior edge‐keep probabilities
        a_post = self.bb.alpha1 + C
        b_post = self.bb.alpha0 + (K - C)
        P_keep = a_post / (a_post + b_post)

        # vectorized Bernoulli draws
        U = self.rng.random((NL, NL))
        if self.directed:
            L = (U < P_keep).astype(int)
        else:
            m = np.triu(U < P_keep, 1)
            L = m.astype(int) + m.T.astype(int)

        np.fill_diagonal(L, 0)
        return L

    # cosine similarity attr‑vs‑latent slot
    def _attr_cosine(self,
                     k: int,
                     _eps: float = 1e-9) -> np.ndarray:
        """
        Method to compute cosine similarity between observed nodes and
        latent slots.

        Parameters
        -----------
        k : int
            the grapn index.
        
        Returns
        -------
        np.ndarray
            The cosine similarity matrix between averaged latent
            embeddings in the blueprint graph and the observed node
            embeddings.
        """
        pk, Xk = self.perm[k], self.X[k] # (n_k,), (n_k, d)
        NL, d = self.NL, Xk.shape[1]

        # Accumulate slot‐means in one pass
        blue_mu = np.zeros((NL, d), float)
        counts  = np.zeros(NL,    int)

        mask = pk >= 0
        np.add.at(blue_mu, pk[mask], Xk[mask])
        counts[pk[mask]] += 1

        # avoid division by zero
        counts = counts.astype(float)
        counts[counts == 0] = 1.0
        blue_mu /= counts[:, None] # (NL, d)

        # Normalize
        x_norm = np.linalg.norm(Xk, axis=1, keepdims=True) + _eps
        b_norm = np.linalg.norm(blue_mu, axis=1, keepdims=True) + _eps

        Xn = Xk / x_norm # (n_k, d)
        Bn = blue_mu / b_norm # (NL, d)

        # Dot‐product all at once
        return Xn @ Bn.T 

    #  MCMC moves
    # elementary swap
    def _proposal_swap(self,
                       k: int) -> Optional[Tuple[int, int, np.ndarray]]:
        """
        Method to propose a new pairwise graph permuation.

        Parameters
        ----------
        k : int
            The index of the graph. 
        
        Returns
        -------
        i : int
            The permuted node index.
        j : int
            The permuted node index.
        new_pk : np.ndarray
            The new permuation.
        """
        # Find indices that are not anchored.
        idx = [i for i, slot in enumerate(self.perm[k]) if not self._is_anchor(k, i)]
        if len(idx) < 2:
            return None
        
        # Permute
        i, j = self.rng.choice(idx, 2, replace=False)
        gi = self._node_group(k, i)
        gj = self._node_group(k, j)
        
        # Do not allow swap if from different groups.
        if gi is not None and gj is not None and gi != gj:
            return None
        
        # Update permutation.
        new_pk = self.perm[k].copy()
        new_pk[i], new_pk[j] = new_pk[j], new_pk[i]
        return (i, j, new_pk)

    # 3‑cycle
    def _proposal_cycle3(self, k: int) -> Optional[np.ndarray]:
        """
        Method to propose a new 3-way graph permuation for improved
        mixing.

        Parameters
        ----------
        k : int
            The index of the graph. 
        
        Returns
        -------
        new_pk : np.ndarray
            The new permuation.
        """
        idx = [i for i, slot in enumerate(self.perm[k]) if not self._is_anchor(k, i)]
        if len(idx) < 3:
            return None
        i, j, l = self.rng.choice(idx, 3, replace=False)
        new_pk = self.perm[k].copy()
        new_pk[i], new_pk[j], new_pk[l] = new_pk[l], new_pk[i], new_pk[j]
        return new_pk

    # full resample via similarity
    def _proposal_resample(self,
                           k: int) -> np.ndarray:
        """
        Method to optimize permutations entirely by the cosine
        similarity between latent node embeddings and observed node
        embeddings .

        Parameters
        -----------
        k : int
            The graph index.
        
        Returns
        -------
        new_pk : np.ndarray
            The new permuation.
        """
        cos = self._attr_cosine(k)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-cos)
        new_pk = self.perm[k].copy()
        new_pk[row_ind] = col_ind
        # keep anchors in place
        for i, slot in enumerate(self.perm[k]):
            if self._is_anchor(k, i):
                new_pk[i] = slot
        return new_pk

    #  birth / death for partial overlap
    def _birth(self) -> bool:
        """
        Method to add a new latent slot to the blueprint graph.

        Returns
        -------
        bool
            Boolean indicating whether a birth was successful.
        """
        unmatched = [(k, i)
                      for k in range(self.K)
                      for i in range(self.n[k])
                      if self.perm[k][i] == -1 and not self._is_anchor(k, i)]
        if not unmatched:
            return False
        k, i = unmatched[self.rng.integers(len(unmatched))]

        # Broadcast over only the observed adjacency matrix.
        s = self.NL
        # expand blueprint matrix
        self.L = np.pad(self.L, ((0, 1), (0, 1)), constant_values=0)
        self.NL += 1

        # initialise blueprint‐slot s by looking up only the edges
        # in graph k that map to existing slots < s
        Ak = (self.W[k] > 0.5) if self.binary_mode else (self.W[k] > 0)
        row = np.zeros(self.NL, int)
        pk = self.perm[k]
        for t in range(s):

            js = np.where(pk == t)[0]
            if js.size:
                row[t] = int(Ak[i, js[0]])
        # row[s] stays 0
        self.L[s, :] = row
        self.L[:, s] = row

        # assign permutation
        self.perm[k][i] = s

        # First, recompute the count matrix for the graph 'k' where the birth occurred.
        # This will be created with the new, larger dimension (self.NL).
        self.C_k[k] = self._compute_Ck(self.perm[k], self.W[k])
        
        # Next, pad all *other* per-graph count matrices to match the new dimension.
        for idx in range(len(self.C_k)):
            if self.C_k[idx].shape[0] < self.NL:
                self.C_k[idx] = np.pad(self.C_k[idx], ((0, 1), (0, 1)), constant_values=0)

        # Now that all matrices in self.C_k are consistent, rebuild the global sum.
        self.C_global = np.sum(self.C_k, axis=0)

        # sanity check
        for idx, Ck in enumerate(self.C_k):
            assert Ck.shape == (self.NL, self.NL), (
                f"_birth: C_k[{idx}] has shape {Ck.shape}, expected {(self.NL,self.NL)}"
            )
        
        return True


    def _death(self) -> bool:
        """
        Delete an unassigned latent slot from the blueprint graph.

        Returns
        -------
        bool
            Boolean indicating whether a death was successful.
        """

        assigned = {slot for pk in self.perm for slot in pk if slot >= 0}
        empty = [s for s in range(self.NL) if s not in assigned]
        if not empty:
            return False
        s = empty[self.rng.integers(len(empty))]
        # delete slot
        self.L = np.delete(np.delete(self.L, s, axis=0), s, axis=1)
        self.NL -= 1
        for pk in self.perm:
            mask = pk == s
            pk[mask] = -1
            pk[pk > s] -= 1

        # Shrink all per-graph count matrices to new NL
        for idx in range(len(self.C_k)):
            Ck = self.C_k[idx]
            # drop row s and col s
            self.C_k[idx] = np.delete(np.delete(Ck, s, axis=0), s, axis=1)

        # rebuild global sum in a shape-safe way
        self.C_global = self.C_k[0].copy()
        for Ck in self.C_k[1:]:
            self.C_global += Ck

        # sanity check
        for idx, Ck in enumerate(self.C_k):
            assert Ck.shape == (self.NL, self.NL), (
                f"_death: C_k[{idx}] has shape {Ck.shape}, expected {(self.NL,self.NL)}"
            )
        return True

    #  Convenience helpers
    def _node_group(self,
                    k: int,
                    i: int) -> np.ndarray:
        """
        Method to return the group label of a node.
        
        Parameters
        ----------
        k : int
            The graph index.
        i : int
            The node index.
        
        Returns
        -------
        Any
            The group label (string or int typically).
        """
        return self.graphs[k].nodes[self.V[k][i]].get("group", None)

    def _is_anchor(self,
                   k: int,
                   i: int) -> bool:
        """
        Method to probe whether a node is anchored from permutation.

        Parameters
        -----------
        k : int
            The graph index.
        i : int
            The node index.

        Returns
        -------
        bool
            Boolean for whether the node is an anchor.
        """
        return self.graphs[k].nodes[self.V[k][i]].get("anchor", False)

    #  Sampler
    def sample(self) -> None:
        """
        Main method to sample the MCMC.

        """
        # Initial blueprint and energy calculation
        # Init latent graph from majority consensus not Gibbs sampling.
        self.L = self._gibbs_sample_blueprint()
        cur_E = self._energy()
        
        total_steps = self.burn_in + self.samples * self.thin
        
        for step in range(total_steps):

            # Bookkeeping on number of latent edges.
            self.trace_edges.append(self.L.sum() // 2)  # divide by 2 for undirected edges


            # if step > self.burn_in: 
            if step < self.burn_in:
                self._in_growth_phase = True
            else:
                self._in_growth_phase = False
            
            # Update trace
            self.trace_E.append(cur_E)
            self.trace_NL.append(self.NL)

            move_type = self.rng.random()
            accepted = False
            
            if move_type < self.birth_death_prob:
                if self.rng.random() < 0.5:
                    
                    # Book keeping on proposal counts.
                    self.proposal_counts["birth"] += 1 
                    
                    # Birth proposal
                    prev_state = (self.NL, [p.copy() for p in self.perm], self.L.copy(), [c.copy() for c in self.C_k], self.C_global.copy())

                    if self._birth():
                        
                        # The blueprint is updated implicitly within _birth resampling.
                        self.L = self._gibbs_sample_blueprint()
                        new_E = self._energy()

                        log_acc_prob = cur_E - new_E
                        if np.log(self.rng.random()) < log_acc_prob:
                            cur_E = new_E
                            accepted = True
                            # Book keeping on acceptance counts.
                            self.accept_counts["birth"] += 1
                        else:
                            
                            # Revert state if not accepted
                            self.NL, self.perm, self.L, self.C_k, self.C_global = prev_state

                else:
                    
                    # Death proposal
                    self.proposal_counts["death"] += 1
                    prev_state = (self.NL, [p.copy() for p in self.perm], self.L.copy(), [c.copy() for c in self.C_k], self.C_global.copy())

                    if self._death():
                        
                        # The blueprint is updated implicitly within _death
                        self.L = self._gibbs_sample_blueprint()
                        new_E = self._energy()

                        log_acc_prob = cur_E - new_E
                        if np.log(self.rng.random()) < log_acc_prob:
                            cur_E = new_E
                            accepted = True
                            self.accept_counts["death"] += 1
                        else:
                            
                            # Revert state if not accepted
                            self.NL, self.perm, self.L, self.C_k, self.C_global = prev_state
            else:
                # Permutation move proposal
                k = self.rng.integers(self.K)
                if move_type < 0.6:
                    self.proposal_counts["swap"] += 1
                    prop = self._proposal_swap(k)
                elif move_type < 0.8:
                    self.proposal_counts["cycle3"] += 1
                    prop = self._proposal_cycle3(k)
                else:
                    self.proposal_counts["resample"] += 1
                    prop = self._proposal_resample(k)

                if prop is not None:

                    # Use global counts to scale in O(1)
                    prev_pk = self.perm[k].copy()
                    prev_Ck = self.C_k[k].copy()

                    new_pk = prop if isinstance(prop, np.ndarray) else prop[2]
                    # recompute only graph k's counts
                    new_Ck = self._compute_Ck(new_pk, self.W[k])
                    # update the global sum
                    self.C_global += (new_Ck - prev_Ck)

                    # tentatively install the new permutation amd counts
                    self.perm[k] = new_pk
                    self.C_k[k]  = new_Ck
                    
                    # (re‐)sample blueprint if needed
                    self.L = self._gibbs_sample_blueprint()
                    new_E = self._energy()
                    
                    # Standard Metropolis-Hastings acceptance criterion
                    log_acc_prob = cur_E - new_E
                    if np.log(self.rng.random()) < log_acc_prob:
                        
                        # Accept the move
                        cur_E = new_E
                        accepted = True

                        # Book keeping on acceptance counts.
                        if move_type < 0.6:
                            self.accept_counts["swap"] += 1
                        elif move_type < 0.8:
                            self.accept_counts["cycle3"] += 1
                        else:
                            self.accept_counts["resample"] += 1
                    
                    else:
                        # revert swap and counts on rejection
                        self.perm[k]   = prev_pk
                        self.C_k[k]    = prev_Ck
                        self.C_global -= (new_Ck - prev_Ck)
                        self.L = self._gibbs_sample_blueprint()


            # Store the state after burn-in and thinning
            if step >= self.burn_in and (step - self.burn_in) % self.thin == 0:
                self._stored_L.append(self.L.copy())
                for k in range(self.K):
                    self._stored_pi[k].append(self.perm[k].copy())

    def latent_blueprint_graph(self,
                               posterior_prob_cutoff: float = 0.2) -> nx.DiGraph:
        """
        Method to return the latent blueprint graph by correctly averaging
        the posterior samples, even when the number of latent nodes changes.

        Returns
        -------
        nx.DiGraph
            The mean of the latent posterior.
        """
        if not self._stored_L:
            raise RuntimeError("Run sample() first.")

        max_nl = 0
        for l_matrix in self._stored_L:
            if l_matrix.shape[0] > max_nl:
                max_nl = l_matrix.shape[0]

        if max_nl == 0:
            return nx.DiGraph()

        tally_matrix = np.zeros((max_nl, max_nl))
        num_samples = len(self._stored_L)
        for l_matrix in self._stored_L:
            current_nl = l_matrix.shape[0]
            tally_matrix[:current_nl, :current_nl] += l_matrix

        Lavg = tally_matrix / num_samples

        Lbin = (Lavg >= posterior_prob_cutoff).astype(int)
        return nx.from_numpy_array(Lbin, create_using=nx.DiGraph)

    def posterior_match_probabilities(self) -> Dict:
        """
        Method to return the posterior probability of two nodes
        belonging to the same latent slot.

        Returns
        -------
        probs : Dict
            The dictionary of graph index and node latent node mapping.
        """
        if not any(self._stored_pi):
            raise RuntimeError("Run sample() first.")
        probs = {}
        for k in range(self.K):
            for l in range(k + 1, self.K):
                n_k, n_l = self.n[k], self.n[l]
                Pkl = np.zeros((n_k, n_l))
                for s in range(len(self._stored_pi[k])):
                    pk, pl = self._stored_pi[k][s], self._stored_pi[l][s]
                    for i in range(n_k):
                        slot = pk[i]
                        if slot < 0:
                            continue
                        js = np.where(pl == slot)[0]
                        Pkl[i, js] += 1.0
                Pkl /= len(self._stored_pi[k])
                probs[(k, l)] = Pkl
        return probs
    
    # Mapping of latent slot to input nodes.
    def get_node_to_latent_mapping(self) -> Dict[int, np.ndarray]:
            """
            Method to compute the mapping of original graph nodes to
            latent slots based on the stored permutations.

            Returns
            -------
            Dict[int, np.ndarray]
                A dictionary where keys are graph indices and values are
                arrays of probabilities for each latent slot.
            """
            if not any(self._stored_pi):
                raise RuntimeError("Run sample() first.")

            # Determine the maximum number of latent nodes to define the matrix size
            max_nl = 0
            for l_matrix in self._stored_L:
                if l_matrix.shape[0] > max_nl:
                    max_nl = l_matrix.shape[0]

            if max_nl == 0:
                return {k: np.array([]) for k in range(self.K)}

            num_samples = len(self._stored_pi[0])
        
            prob_mapping = {}

            # For each original graph
            for k in range(self.K):
                num_nodes_in_k = self.n[k]
                # Create a tally matrix: rows are original nodes, columns are latent slots
                tally_matrix = np.zeros((num_nodes_in_k, max_nl))

                # Go through each stored sample
                for s in range(num_samples):
                    # Get the permutation for this graph at this sample step
                    pk = self._stored_pi[k][s]
                    
                    # For each node in the original graph
                    for node_idx, latent_slot in enumerate(pk):
                        if latent_slot >= 0:
                            # Increment the count for this node mapping to this latent slot
                            tally_matrix[node_idx, latent_slot] += 1
                
                # Normalize the tally by the number of samples to get the probability
                prob_mapping[k] = tally_matrix / num_samples
                
            return prob_mapping


#TODO: update to FAISS to scale > 1e4
def auto_anchors_by_cosine( graphs: List[nx.DiGraph],
                           *,
                           emb_key: str = "emb_arr",
                           cos_threshold: float = 0.90,
                           anchor_attr: str = "anchor_id") -> None:
    """
    Function to map all nodes within a cosine threshold value to the
    same anchor point in the latent graph. 

    Parameters
    ----------
    graphs : List
        The list of graphs.
    
    emb_key : str, default=`emb_arr`
        The node attribute key for embeddings.
    
    cos_threshold : float, default=`0.90`
        The cosine similarity threshold to pin nodes to the same latent
        slot.
    
    anchor_attr : str, default=`anchor_id`
        The node attribute key for the anchor id.
    """
    
    emb_list : List[np.ndarray] = []
    backref : List[Tuple[int,int]] = []  # (graph_idx, node_idx)
    for k, G in enumerate(graphs):
        for i, v in enumerate(G.nodes()):
            emb = np.asarray(G.nodes[v][emb_key], float)
            emb_list.append(emb)
            backref.append((k, i))

    E = np.vstack(emb_list) # (N, d)
    S = cosine_similarity_matrix(E, E)

    parent = np.arange(len(E))

    def find(a):
        while parent[a]!=a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra!=rb:
            parent[rb]=ra

    # only union if from different graphs
    for a,b in zip(*np.where(S>=cos_threshold)):
        if a<b and backref[a][0] != backref[b][0]:
            union(a,b)

    # now assign anchor IDs, but only for clusters that span >1 graph
    root_to_members = {}
    for idx,(k,i) in enumerate(backref):
        r = find(idx)
        root_to_members.setdefault(r, []).append((k,i))

    next_id = 0
    for r, members in root_to_members.items():
        graphs_seen = {k for k,i in members}
        if len(graphs_seen) > 1:
            for k,i in members:
                v = list(graphs[k].nodes())[i]
                graphs[k].nodes[v]["anchor"] = True
                graphs[k].nodes[v][anchor_attr] = next_id
            next_id += 1