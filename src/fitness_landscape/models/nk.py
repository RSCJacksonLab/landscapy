"""Factories for binary and generalized NK fitness landscapes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from ..core.fitness import NumericFitness
from ..core.landscape import FitnessLandscape
from ..core.sequence import BinarySequence, MultialleleSequence


_DEFAULT_GNK_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
_Alphabet = Union[Sequence[Hashable], Mapping[int, Sequence[Hashable]]]


@dataclass(frozen=True)
class _NKSpecification:
    """Validated, normalized inputs for an NK state-space construction."""

    N: int
    K: int | None
    site_alphabets: dict[int, tuple[Hashable, ...]]
    alphabet_type: str
    variable_sites: tuple[int, ...]
    base_sequence: tuple[Hashable, ...] | None
    adjacency: np.ndarray | None
    interaction_degrees: tuple[int, ...]


def _require_integer(value: object, *, name: str) -> int:
    """Return an integer while rejecting booleans and non-integral values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_alphabet(
    values: Iterable[Hashable], *, name: str
) -> tuple[Hashable, ...]:
    """Return a non-empty, unique tuple of hashable allele values."""
    try:
        alphabet = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of allele values") from error
    if not alphabet:
        raise ValueError(f"{name} must not be empty")

    seen: set[Hashable] = set()
    for allele in alphabet:
        try:
            if allele in seen:
                raise ValueError(f"{name} values must be unique")
            seen.add(allele)
        except TypeError as error:
            raise TypeError(f"{name} values must be hashable") from error
    return alphabet


def _normalize_variable_sites(
    N: int,
    variable_sites: Optional[Sequence[int]],
    *,
    sequence_length: int,
) -> tuple[int, ...]:
    """Validate and normalize the global coordinates varied by the model."""
    if variable_sites is None:
        sites = tuple(range(N))
    else:
        try:
            raw_sites = tuple(variable_sites)
        except TypeError as error:
            raise TypeError("variable_sites must be an iterable of integers") from error
        sites = tuple(
            _require_integer(site, name="variable site") for site in raw_sites
        )

    if len(sites) != N:
        raise ValueError("Length of variable_sites must equal N")
    if len(set(sites)) != len(sites):
        raise ValueError("variable_sites must contain unique indices")
    if any(site < 0 or site >= sequence_length for site in sites):
        raise IndexError("variable_sites indices must be in range")
    return sites


def _validate_adjacency(adj_mat: object, N: int) -> np.ndarray:
    """Return a binary, symmetric NK interaction adjacency matrix."""
    adjacency = np.asarray(adj_mat)
    if adjacency.shape != (N, N):
        raise ValueError(f"adj_mat must have shape ({N}, {N}); got {adjacency.shape}")
    try:
        is_binary = np.logical_or(adjacency == 0, adjacency == 1)
    except (TypeError, ValueError) as error:
        raise ValueError("adj_mat values must be binary (0 or 1)") from error
    if not np.all(is_binary):
        raise ValueError("adj_mat values must be binary (0 or 1)")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("adj_mat must be symmetric")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("adj_mat diagonal must be zero")
    return np.array(adjacency, dtype=np.int8, copy=True)


def _normalize_nk_specification(
    N: int,
    K: Optional[int],
    alphabet: _Alphabet,
    adj_mat: Optional[np.ndarray],
    base_sequence: Optional[Union[Sequence[Hashable], str]],
    variable_sites: Optional[Sequence[int]],
) -> _NKSpecification:
    """Validate user inputs once for every NK entry point."""
    N = _require_integer(N, name="N")
    if N <= 0:
        raise ValueError("N must be positive")

    normalized_k = None
    if K is not None:
        normalized_k = _require_integer(K, name="K")
        if normalized_k < 0:
            raise ValueError("K must be non-negative")
        if normalized_k >= N:
            raise ValueError("K must be less than N")

    if base_sequence is None:
        base = None
        sequence_length = N
    else:
        try:
            base = tuple(base_sequence)
        except TypeError as error:
            raise TypeError("base_sequence must be a sequence") from error
        if not base:
            raise ValueError("base_sequence must not be empty")
        sequence_length = len(base)

    sites = _normalize_variable_sites(
        N,
        variable_sites,
        sequence_length=sequence_length,
    )

    if isinstance(alphabet, Mapping):
        site_alphabets: dict[int, tuple[Hashable, ...]] = {}
        for raw_site, values in alphabet.items():
            site = _require_integer(raw_site, name="alphabet site")
            site_alphabets[site] = _validate_alphabet(
                values,
                name=f"alphabet[{site}]",
            )
        missing = [site for site in sites if site not in site_alphabets]
        if missing:
            raise ValueError(f"Per-site alphabet missing for variable_sites {missing}")
        site_alphabets = {site: site_alphabets[site] for site in sites}
        alphabet_type = "per-site"
    else:
        uniform_alphabet = _validate_alphabet(alphabet, name="alphabet")
        site_alphabets = {site: uniform_alphabet for site in sites}
        alphabet_type = "uniform"

    if base is not None:
        for site in sites:
            if base[site] not in site_alphabets[site]:
                raise ValueError(
                    f"base_sequence[{site}]={base[site]!r} is not in alphabet[{site}]"
                )

    adjacency = None
    if adj_mat is not None:
        adjacency = _validate_adjacency(adj_mat, N)
        degrees = tuple(int(value) for value in adjacency.sum(axis=1))
        if normalized_k is not None and any(
            degree != normalized_k for degree in degrees
        ):
            raise ValueError(
                "K must equal every adj_mat row degree when both are provided"
            )
        if normalized_k is None and len(set(degrees)) == 1:
            normalized_k = degrees[0]
    else:
        if normalized_k is None:
            raise ValueError("Either K or adj_mat must be provided")
        degrees = (normalized_k,) * N

    return _NKSpecification(
        N=N,
        K=normalized_k,
        site_alphabets=site_alphabets,
        alphabet_type=alphabet_type,
        variable_sites=sites,
        base_sequence=base,
        adjacency=adjacency,
        interaction_degrees=degrees,
    )


def _generate_nk_states(
    specification: _NKSpecification,
    *,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the state space and fitness signal for a validated model."""
    rng = np.random.default_rng(seed)
    sites = specification.variable_sites
    variable_alphabets = [specification.site_alphabets[site] for site in sites]

    if specification.base_sequence is None:
        sequences = np.array(list(product(*variable_alphabets)), dtype=object)
    else:
        sequence_rows = []
        for alleles in product(*variable_alphabets):
            row = list(specification.base_sequence)
            for allele, site in zip(alleles, sites):
                row[site] = allele
            sequence_rows.append(row)
        sequences = np.array(sequence_rows, dtype=object)

    allele_maps = {
        site: {
            allele: index
            for index, allele in enumerate(specification.site_alphabets[site])
        }
        for site in sites
    }
    alphabet_sizes = {site: len(specification.site_alphabets[site]) for site in sites}

    interaction_sites: list[list[int]] = []
    if specification.adjacency is not None:
        for local_site, site in enumerate(sites):
            neighbor_indices = np.flatnonzero(specification.adjacency[local_site])
            neighbors = [sites[int(index)] for index in neighbor_indices]
            interaction_sites.append([site, *sorted(neighbors)])
    else:
        for site in sites:
            choices = [candidate for candidate in sites if candidate != site]
            neighbors = rng.choice(
                choices,
                size=specification.K,
                replace=False,
            ).tolist()
            interaction_sites.append([site, *sorted(neighbors)])

    contribution_tables = []
    for participating_sites in interaction_sites:
        bases = [alphabet_sizes[site] for site in participating_sites]
        table = rng.random(int(np.prod(bases, dtype=np.int64)))
        table -= table.mean()
        contribution_tables.append((participating_sites, bases, table))

    global_to_local = {site: index for index, site in enumerate(sites)}
    fitness_values = np.zeros(len(sequences), dtype=float)
    for row_index, sequence in enumerate(sequences):
        total = 0.0
        for participating_sites, bases, table in contribution_tables:
            if specification.base_sequence is None:
                alleles = [
                    sequence[global_to_local[site]] for site in participating_sites
                ]
            else:
                alleles = [sequence[site] for site in participating_sites]
            digits = [
                allele_maps[site][allele]
                for site, allele in zip(participating_sites, alleles)
            ]
            table_index = 0
            for digit, base in zip(digits, bases):
                table_index = table_index * base + digit
            total += table[table_index]
        fitness_values[row_index] = total / specification.N

    return sequences, fitness_values


def generate_NK_states(
    N: int,
    K: Optional[int] = None,
    alphabet: _Alphabet = (0, 1),
    seed: Optional[int] = None,
    adj_mat: Optional[np.ndarray] = None,
    base_sequence: Optional[Union[Sequence[Hashable], str]] = None,
    variable_sites: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate generalized NK sequences and their fitness values.

    Parameters
    ----------
    N : int
        Number of variable sites.
    K : int, optional
        Number of interacting neighbours per variable site. Required when
        ``adj_mat`` is omitted and constrained to ``0 <= K < N``.
    alphabet : sequence or mapping of int to sequence, default=(0, 1)
        A uniform ordered alphabet or ordered alphabets keyed by global variable
        site. Every alphabet must be non-empty and contain unique values.
    seed : int, optional
        Random seed controlling neighbourhood and contribution-table sampling.
    adj_mat : ndarray, optional
        Binary symmetric ``(N, N)`` interaction matrix with a zero diagonal.
    base_sequence : sequence or str, optional
        Full template sequence. Only ``variable_sites`` are varied.
    variable_sites : sequence of int, optional
        Unique global indices of the ``N`` varied sites. Defaults to the first
        ``N`` positions.

    Returns
    -------
    tuple of ndarray
        State-space array and aligned floating-point fitness array.
    """
    specification = _normalize_nk_specification(
        N,
        K,
        alphabet,
        adj_mat,
        base_sequence,
        variable_sites,
    )
    return _generate_nk_states(specification, seed=seed)


def _ordered_sequence_alphabet(
    specification: _NKSpecification,
) -> list[Hashable]:
    """Return the ordered union needed by full sequence objects."""
    ordered: list[Hashable] = []
    seen: set[Hashable] = set()
    for site in specification.variable_sites:
        for allele in specification.site_alphabets[site]:
            if allele not in seen:
                seen.add(allele)
                ordered.append(allele)
    if specification.base_sequence is not None:
        for allele in specification.base_sequence:
            if allele not in seen:
                seen.add(allele)
                ordered.append(allele)
    return ordered


def _model_metadata(
    specification: _NKSpecification,
    *,
    model_type: str,
) -> dict:
    """Describe variable alphabets and interactions without ambiguity."""
    metadata = {
        "N": specification.N,
        "K": specification.K,
        "type": model_type,
        "alphabet_type": specification.alphabet_type,
        "variable_sites": list(specification.variable_sites),
        "interaction_type": (
            "adjacency" if specification.adjacency is not None else "random"
        ),
        "interaction_degrees": list(specification.interaction_degrees),
    }
    if specification.alphabet_type == "uniform":
        alphabet = list(specification.site_alphabets[specification.variable_sites[0]])
        metadata.update(
            {
                "alphabet": alphabet,
                "alphabet_size": len(alphabet),
            }
        )
    else:
        metadata.update(
            {
                "site_alphabets": {
                    site: list(specification.site_alphabets[site])
                    for site in specification.variable_sites
                },
                "alphabet_sizes": {
                    site: len(specification.site_alphabets[site])
                    for site in specification.variable_sites
                },
            }
        )
    if specification.base_sequence is not None:
        metadata["base_sequence"] = list(specification.base_sequence)
    return metadata


def _build_nk_landscape(
    specification: _NKSpecification,
    fitness_values: np.ndarray,
    sequences_np: np.ndarray,
    *,
    binary: bool,
    **kwargs,
) -> FitnessLandscape:
    """Build a landscape from one authoritative NK state generator."""
    if binary:
        sequences = [BinarySequence(sequence) for sequence in sequences_np]
        model_type = "binary"
    else:
        sequence_alphabet = _ordered_sequence_alphabet(specification)
        sequences = [
            MultialleleSequence(sequence, alphabet=sequence_alphabet)
            for sequence in sequences_np
        ]
        model_type = "generalized"

    layer_name = (
        f"nk_k={specification.K}" if specification.K is not None else "nk_adjacency"
    )
    fitness_layers = {
        layer_name: NumericFitness(
            name=layer_name,
            values=[[value] for value in fitness_values],
            metadata=_model_metadata(specification, model_type=model_type),
        )
    }
    return FitnessLandscape.build(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph="hamming",
        **kwargs,
    )


def create_gnk_landscape(
    N: int,
    K: Optional[int] = None,
    alphabet: _Alphabet = _DEFAULT_GNK_ALPHABET,
    seed: Optional[int] = None,
    adj_mat: Optional[np.ndarray] = None,
    base_sequence: Optional[Union[Sequence[Hashable], str]] = None,
    variable_sites: Optional[Sequence[int]] = None,
    **kwargs,
) -> FitnessLandscape:
    """Create a generalized, potentially multiallelic NK landscape.

    Parameters
    ----------
    N : int
        Number of variable sites.
    K : int, optional
        Number of interacting neighbours per site. Required without an
        adjacency matrix. If supplied with an adjacency matrix, it must equal
        every row degree.
    alphabet : sequence or mapping of int to sequence, optional
        Uniform alphabet or per-site alphabets keyed by global variable site.
    seed : int, optional
        Random-number-generator seed.
    adj_mat : ndarray, optional
        Binary symmetric interaction matrix with a zero diagonal.
    base_sequence : sequence or str, optional
        Full template sequence whose non-variable positions remain fixed.
    variable_sites : sequence of int, optional
        Unique global coordinates varied in ``base_sequence``.
    **kwargs : dict, optional
        Additional arguments for :meth:`FitnessLandscape.build`.

    Returns
    -------
    FitnessLandscape
        Generalized NK landscape containing ``MultialleleSequence`` objects.
    """
    specification = _normalize_nk_specification(
        N,
        K,
        alphabet,
        adj_mat,
        base_sequence,
        variable_sites,
    )
    sequences_np, fitness_values = _generate_nk_states(
        specification,
        seed=seed,
    )
    return _build_nk_landscape(
        specification,
        fitness_values,
        sequences_np,
        binary=False,
        **kwargs,
    )


def create_nk_binary_landscape(
    N: int,
    K: Optional[int] = None,
    seed: Optional[int] = None,
    adj_mat: Optional[np.ndarray] = None,
    **kwargs,
) -> FitnessLandscape:
    """Create a binary NK landscape.

    Parameters
    ----------
    N : int
        Number of binary sites.
    K : int, optional
        Number of interacting neighbours per site. Required without an
        adjacency matrix. If supplied with an adjacency matrix, it must equal
        every row degree.
    seed : int, optional
        Random-number-generator seed.
    adj_mat : ndarray, optional
        Binary symmetric interaction matrix with a zero diagonal.
    **kwargs : dict, optional
        Additional arguments for :meth:`FitnessLandscape.build`.

    Returns
    -------
    FitnessLandscape
        Binary NK landscape containing ``BinarySequence`` objects.
    """
    specification = _normalize_nk_specification(
        N,
        K,
        (0, 1),
        adj_mat,
        None,
        None,
    )
    sequences_np, fitness_values = _generate_nk_states(
        specification,
        seed=seed,
    )
    return _build_nk_landscape(
        specification,
        fitness_values,
        sequences_np,
        binary=True,
        **kwargs,
    )


def create_nk_multi_landscape(
    N: int,
    K: int,
    alphabet: Sequence[Hashable],
    seed: Optional[int] = None,
    **kwargs,
) -> FitnessLandscape:
    """Create a multiallelic GNK landscape through a compatibility alias.

    Parameters
    ----------
    N : int
        Number of variable sites.
    K : int
        Number of interacting neighbours per site.
    alphabet : sequence
        Ordered uniform alphabet used at every site.
    seed : int, optional
        Random-number-generator seed.
    **kwargs : dict, optional
        Additional arguments for :meth:`FitnessLandscape.build`.

    Returns
    -------
    FitnessLandscape
        Generalized NK landscape returned by :func:`create_gnk_landscape`.

    Warns
    -----
    DeprecationWarning
        This compatibility name is deprecated in favour of
        :func:`create_gnk_landscape`.
    """
    warnings.warn(
        "create_nk_multi_landscape is deprecated; use create_gnk_landscape",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_gnk_landscape(
        N=N,
        K=K,
        alphabet=alphabet,
        seed=seed,
        **kwargs,
    )
