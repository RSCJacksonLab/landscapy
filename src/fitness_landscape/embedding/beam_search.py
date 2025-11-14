from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class PseudoLogLikelihoodScorer:
    """
    Lightweight wrapper around an ESM embedder that caches pseudo
    log-likelihood (PLL) evaluations for discrete sequences.

    Parameters
    ----------
    embedder :
        Any object exposing ``lm_output_probabilities`` and an
        ``alphabet`` attribute (mirrors both hard and soft ESM
        embedders used elsewhere in the codebase).
    alphabet : Sequence[str], optional
        Alphabet to use for PLL extraction. Defaults to the embedder's
        alphabet.
    batch_size : int, optional
        Batch size to use when querying the embedder. Defaults to the
        embedder's internal ``batch_size`` attribute when available.
    log_eps : float, default=1e-8
        Numerical stabiliser added before taking logarithms.
    """

    def __init__(
        self,
        embedder,
        *,
        alphabet: Sequence[str] | None = None,
        batch_size: int | None = None,
        log_eps: float = 1e-8,
    ) -> None:
        if alphabet is None:
            if not hasattr(embedder, "alphabet"):
                raise ValueError("alphabet must be provided when embedder lacks an alphabet attribute.")
            alphabet = embedder.alphabet

        if not alphabet:
            raise ValueError("alphabet must contain at least one token.")

        self.embedder = embedder
        self.alphabet = list(alphabet)
        self.token_index = {str(tok): idx for idx, tok in enumerate(self.alphabet)}
        self.batch_size = batch_size if batch_size is not None else getattr(embedder, "batch_size", 1)
        self.log_eps = float(log_eps)
        self._cache: Dict[str, float] = {}

    def clear_cache(self) -> None:
        """Remove all cached PLL scores."""
        self._cache.clear()

    def score(self, sequences: Sequence[str]) -> List[float]:
        """
        Batch-compute PLL scores for the requested sequences, using the
        internal cache to avoid redundant PLM queries.
        """
        missing: List[str] = [seq for seq in sequences if seq not in self._cache]
        if missing:
            probs = self.embedder.lm_output_probabilities(missing, batch_size=self.batch_size)
            for seq, site_probs in zip(missing, probs):
                self._cache[seq] = self._pll_from_probs(seq, site_probs)
        return [self._cache[seq] for seq in sequences]

    def score_one(self, sequence: str) -> float:
        """Convenience wrapper to score a single sequence."""
        return self.score([sequence])[0]

    def _pll_from_probs(self, sequence: str, probs: np.ndarray) -> float:
        if probs.ndim != 2:
            raise ValueError("Probabilities array must have shape (L, alphabet_size).")
        if probs.shape[0] < len(sequence):
            raise ValueError("Probability array shorter than the provided sequence.")

        total = 0.0
        for idx, token in enumerate(sequence):
            key = self._normalise_token(token)
            tok_idx = self.token_index.get(key)
            if tok_idx is None:
                raise ValueError(f"Token {token!r} not present in embedder alphabet.")
            p = float(probs[idx, tok_idx])
            total += float(np.log(max(p, self.log_eps)))
        return total

    def _normalise_token(self, token: str) -> str:
        t = str(token)
        if t == "gap":
            return "-"
        if t in self.token_index:
            return t
        upper = t.upper()
        if upper in self.token_index:
            return upper
        return t


@dataclass
class BeamState:
    """Container for a candidate sequence within the beam."""

    sequence: str
    pll: float
    matches: int


class InterpolationBeamSearch:
    """
    Beam-search interpolation strategy that incrementally mutates a
    source sequence towards a target sequence while maximising the PLL.
    """

    def __init__(
        self,
        scorer: PseudoLogLikelihoodScorer,
        *,
        beam_width: int = 8,
        distance_penalty: float = 0.4,
        max_rounds: int = 25,
        max_children_per_parent: int | None = None,
        max_candidates_per_round: int | None = None,
        min_pll_gain: float = 1e-4,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1.")
        self.scorer = scorer
        self.beam_width = beam_width
        self.distance_penalty = distance_penalty
        self.max_rounds = max_rounds
        self.max_children_per_parent = max_children_per_parent
        self.max_candidates_per_round = max_candidates_per_round
        self.min_pll_gain = min_pll_gain

    def interpolate(
        self,
        start_seq: str,
        target_seq: str,
        *,
        target_counts: Sequence[int],
        diff_positions: Sequence[int] | None = None,
        start_pll: float | None = None,
    ) -> List[BeamState]:
        """
        Generate intermediate sequences that monotonically increase the
        PLL while moving towards the target sequence. Returns a list of
        ``BeamState`` objects ordered along the path (excludes the
        original start sequence).
        """
        if diff_positions is None:
            diff_positions = [i for i, (a, b) in enumerate(zip(start_seq, target_seq)) if a != b]

        if not diff_positions:
            return []

        total_diffs = len(diff_positions)
        valid_counts = sorted({c for c in target_counts if 0 < c < total_diffs})
        if not valid_counts:
            return []

        path: List[BeamState] = []
        current_seq = start_seq
        current_pll = self.scorer.score_one(start_seq) if start_pll is None else start_pll
        current_matches = self._count_matches(current_seq, target_seq, diff_positions)

        for target_count in valid_counts:
            if target_count <= current_matches:
                continue
            start_state = BeamState(sequence=current_seq, pll=current_pll, matches=current_matches)
            result = self._run_single_target(start_state, target_seq, target_count, diff_positions)
            if result is None:
                break
            if result.pll <= current_pll + self.min_pll_gain:
                continue
            path.append(result)
            current_seq = result.sequence
            current_pll = result.pll
            current_matches = result.matches

        return path

    def _run_single_target(
        self,
        start_state: BeamState,
        target_seq: str,
        target_matches: int,
        diff_positions: Sequence[int],
    ) -> BeamState | None:
        beam: List[BeamState] = [start_state]
        visited = {start_state.sequence}
        best: BeamState | None = None

        for _ in range(self.max_rounds):
            proposals = self._propose_candidates(beam, target_seq, diff_positions, visited)
            if not proposals:
                break

            if self.max_candidates_per_round is not None and len(proposals) > self.max_candidates_per_round:
                proposals = proposals[: self.max_candidates_per_round]

            pll_scores = self.scorer.score(proposals)
            ranked: List[Tuple[float, BeamState]] = []
            for seq, pll in zip(proposals, pll_scores):
                matches = self._count_matches(seq, target_seq, diff_positions)
                penalty = self.distance_penalty * abs(matches - target_matches)
                ranking_score = pll - penalty
                candidate = BeamState(sequence=seq, pll=pll, matches=matches)
                ranked.append((ranking_score, candidate))
                if matches >= target_matches and (best is None or pll > best.pll):
                    best = candidate

            ranked.sort(key=lambda item: item[0], reverse=True)
            beam = [state for _, state in ranked[: self.beam_width]]

            if best is not None and best.matches >= target_matches:
                return best

        return best

    def _propose_candidates(
        self,
        beam: Sequence[BeamState],
        target_seq: str,
        diff_positions: Sequence[int],
        visited: set[str],
    ) -> List[str]:
        proposals: List[str] = []
        for state in beam:
            mutable_positions = [pos for pos in diff_positions if state.sequence[pos] != target_seq[pos]]
            if not mutable_positions:
                continue
            if self.max_children_per_parent is not None:
                mutable_positions = mutable_positions[: self.max_children_per_parent]
            for pos in mutable_positions:
                seq_list = list(state.sequence)
                seq_list[pos] = target_seq[pos]
                candidate = "".join(seq_list)
                if candidate not in visited:
                    visited.add(candidate)
                    proposals.append(candidate)
        return proposals

    @staticmethod
    def _count_matches(seq: str, target: str, positions: Sequence[int]) -> int:
        return sum(1 for pos in positions if seq[pos] == target[pos])
