from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import networkx as nx
from scipy.special import gammaln
from isorank import cosine_similarity_matrix
import scipy.optimize  # heavy but only occasional

# Hyper parameter containers
@dataclass
class BernoulliBeta:
    """
    Conjugate prior for binary edges.
    """
    alpha1: float = 1.0  # successes (edge present in blueprint)
    alpha0: float = 1.0  # failures (edge absent)

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

        a1 = self.alpha1 + o11 + o01   # posterior “present” count
        a0 = self.alpha0 + o10 + o00   # posterior “absent”  count

        # log B(a1,a0) – log B(α1,α0)
        return (
            gammaln(a1) + gammaln(a0) - gammaln(a1 + a0)
            - (gammaln(self.alpha1) + gammaln(self.alpha0)
               - gammaln(self.alpha1 + self.alpha0))
        )

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
            + np.log(np.math.gamma(self.alpha0 + 0.5 * n))
            - np.log(np.math.gamma(self.alpha0))
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
    seed : int
        The random state.
    """
    
    def __init__(self,
                 graphs: List[nx.DiGraph],
                 *,
                 alpha: float = 0,
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
                 seed: Union[int, None] = None) -> None:
        
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.graphs = graphs
        self.K = len(graphs)
        self.burn_in, self.samples, self.thin, self.birth_death_prob = burn_in, samples, thin, birth_death_prob
        self.birth_gamma = birth_prior_gamma

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

        # adjacency / weight matrices
        self.W: List[np.ndarray] = []  # may be binary or real‑valued
        for G, vs in zip(graphs, self.V):

            Wk = nx.to_numpy_array(G, nodelist=vs, weight=weight_key, nonedge=0.0)
            self.W.append(Wk)
        self.binary_mode = all(np.isin(Wk, [0.0, 1.0]).all() for Wk in self.W)

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
        self.NL: int = max(max(p.max(initial=-1) + 1 for p in self.perm), max(self.n))  # start at max(n_k)

        # fill non‑anchored nodes with degree ranking heuristic
        for k, (pk, Wk) in enumerate(zip(self.perm, self.W)):
            free_slots = [s for s in range(self.NL) if s not in pk]
            deg_order = np.argsort(Wk.sum(1))
            for i in deg_order:
                if pk[i] == -1:
                    pk[i] = free_slots.pop(0)

        # first blueprint 
        self.L = self._majority_blueprint() #TODO: Base off IsoRank blueprint. 

        # bookkeeping of MCMC trace
        self._stored_L: List[np.ndarray] = []
        self._stored_pi: List[List[np.ndarray]] = [[] for _ in range(self.K)]

    #  Likelihood / posterior helpers
    def _energy(self) -> float:

        """
        Method to compute statistical energy as negative log posterior.

        Returns
        -------
        float
            The weighted or binary statistical energy.
        """

        if self.binary_mode:
            return self._energy_binary()
        return self._energy_weighted()

    # binary edges
    def _energy_binary(self) -> float:
        """
        Method to compute the binary edge (i.e., unweighted) energy. If
        in burn-in, the size is not penalised and the prior if a death /
        node contraction event is functionally null.

        Returns
        -------
        float
            the energy for a binary edge summed with the size penality
            for bith and death events.
        """
        O11 = O10 = O01 = O00 = 0
        for k, (pk, Wk) in enumerate(zip(self.perm, self.W)):
            Lk = self.L[np.ix_(pk, pk)]
            A = (Wk > 0.5).astype(int)
            O11 += (A & Lk).sum()
            O10 += (A & ~Lk).sum()
            O01 += (~A & Lk).sum()
            O00 += (~A & ~Lk).sum()
        log_like = -self.bb.log_marginal(O11, O10, O01, O00)
        size_penalty = 0.0 if self._in_growth_phase else self.birth_gamma * self.NL
        
        return log_like + size_penalty

    # weighted edges
    def _energy_weighted(self) -> float:
        """
        Method to compute the weighted edge energy. If in burn-in, the
        size is not penalised and the prior if a death / node
        ontraction event is functionally null.

        Returns
        -------
        float
            the energy for a weighred edge summed with the size penalty
            for bith and death events.
        """
        logp = 0.0
        for k, (pk, Wk) in enumerate(zip(self.perm, self.W)):
            Lk = self.L[np.ix_(pk, pk)].astype(bool)
            present = Wk[Lk]
            absent  = Wk[~Lk]
            logp -= self.ng.log_marginal(present)
            logp -= self.ng.log_marginal(absent)
        size_penalty = 0.0 if self._in_growth_phase else self.birth_gamma * self.NL
        
        return logp + size_penalty

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

    # cosine similarity attr‑vs‑latent slot
    def _attr_cosine(self,
                     k: int) -> np.ndarray:
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
        pk = self.perm[k]
        Xk = self.X[k]
        d = Xk.shape[1]
        blue_mu = np.zeros((self.NL, d))
        cnt = np.zeros(self.NL)
        for v, slot in enumerate(pk):
            if slot >= 0:
                blue_mu[slot] += Xk[v]
                cnt[slot] += 1
        cnt[cnt == 0] = 1
        blue_mu /= cnt[:, None]
        x_norm = np.linalg.norm(Xk, axis=1, keepdims=True) + 1e-9
        b_norm = np.linalg.norm(blue_mu, axis=1, keepdims=True) + 1e-9
        return (Xk / x_norm) @ (blue_mu / b_norm).T  # (n_k × NL)

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
        """
        unmatched = [(k, i)
                      for k in range(self.K)
                      for i in range(self.n[k])
                      if self.perm[k][i] == -1 and not self._is_anchor(k, i)]
        if not unmatched:
            return False
        k, i = unmatched[self.rng.integers(len(unmatched))]
        s = self.NL

        # expand blueprint matrix
        self.L = np.pad(self.L, ((0, 1), (0, 1)), constant_values=0)
        self.NL += 1

        # initialise edges by copying row/col from observed graph
        Ak = (self.W[k] > 0.5) if self.binary_mode else (self.W[k] > 0)
        row = np.zeros(self.NL, int)
        row[:-1] = Ak[i]
        self.L[s, :] = row
        self.L[:, s] = row
        self.L[s, s] = 0

        # assign permutation
        self.perm[k][i] = s
        return True

    def _death(self) -> bool:
        """
        Delete an unassigned latent slot from the blueprint graph.
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
    def sample(self):
        """
        Main method to sample the MCMC.
        """
        cur_E = self._energy()
        total_steps = self.burn_in + self.samples * self.thin
        
        self._in_growth_phase = True
        for step in range(total_steps):
            if step == self.burn_in:
                self._in_growth_phase = False

            move_type = self.rng.random()
            accepted = False
            if move_type < self.birth_death_prob:
                # birth/death 20 % of the time #TODO: Update to hyperparameter
                if self.rng.random() < 0.5:
                    # Birth
                    prev_state = (self.NL, [p.copy() for p in self.perm], self.L.copy())
                    if self._birth():
                        new_E = self._energy()
                        log_acc = -(new_E - cur_E)  # symmetric proposals approx.
                        if np.log(self.rng.random()) < log_acc:
                            cur_E = new_E
                            accepted = True
                        else:
                            self.NL, self.perm, self.L = prev_state  # revert
                else:
                    # Death
                    prev_state = (self.NL, [p.copy() for p in self.perm], self.L.copy())
                    if self._death():
                        new_E = self._energy()
                        log_acc = -(new_E - cur_E)
                        if np.log(self.rng.random()) < log_acc:
                            cur_E = new_E
                            accepted = True
                        else:
                            self.NL, self.perm, self.L = prev_state
            else:
                # permutation moves
                k = self.rng.integers(self.K)
                if move_type < 0.6:
                    prop = self._proposal_swap(k)
                elif move_type < 0.8:
                    prop = self._proposal_cycle3(k)
                else:
                    prop = self._proposal_resample(k)
                if prop is not None:
                    prev_pk = self.perm[k].copy()
                    self.perm[k] = prop if isinstance(prop, np.ndarray) else prop[2]
                    new_E = self._energy()
                    # attribute/topology proposal weight (symmetric approx.)
                    accept = new_E < cur_E or self.rng.random() < np.exp(cur_E - new_E)
                    if accept:
                        cur_E = new_E; accepted = True
                    else:
                        self.perm[k] = prev_pk
            # blueprint update every sweep of max(n_k)
            if step % max(self.n) == 0:
                self.L = self._majority_blueprint()
            # store after burn‑in
            if step >= self.burn_in and (step - self.burn_in) % self.thin == 0:
                self._stored_L.append(self.L.copy())
                for k in range(self.K):
                    self._stored_pi[k].append(self.perm[k].copy())

    # Main public methods
    def latent_blueprint_graph(self) -> nx.DiGraph:
        """
        Method to return the latent blueprint graph as the mean of the
        sampled posterior.

        Returns
        -------
        nx.DiGraph
            The mean of the latent posterior. 
        """
        if not self._stored_L:
            raise RuntimeError("Run sample() first.")
        Lavg = np.mean(self._stored_L, axis=0)
        Lbin = (Lavg >= 0.5).astype(int)
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
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    idx_i, idx_j = np.where(S >= cos_threshold)
    for a, b in zip(idx_i, idx_j):
        if a < b:
            union(a, b)

    root_to_id : dict[int, int] = {}
    next_id = 0
    for idx in range(len(E)):
        r = find(idx)
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        aid = root_to_id[r]
        k, i = backref[idx]
        v = list(graphs[k].nodes())[i]
        graphs[k].nodes[v]["anchor"] = True
        graphs[k].nodes[v][anchor_attr] = aid