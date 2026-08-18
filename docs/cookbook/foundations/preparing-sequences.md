# Prepare sequence objects

Sequence values, order, alphabet, and identifiers are separate concepts. Choose
the narrowest class that represents the data. `BaseNumpySequence.from_string`
uses Landscapy's canonical 20-amino-acid alphabet by default. Cogent3 is an
optional interoperability layer for alignment and phylogenetic workflows; it
does not define this default.

## Input

Inputs are strings or one-dimensional iterables. Binary values must be `0` or
`1`; multiallelic values must occur in the declared alphabet; protein symbols
can use the canonical default or an explicit custom alphabet. Gaps are
represented consistently as `-` or `gap` according to the chosen alphabet.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape import BaseNumpySequence, BinarySequence, MultialleleSequence

binary = BinarySequence.from_bits([0, 1, 0], sequence_id="binary-010")
multiallele = MultialleleSequence.from_string(
    "ACG", alphabet=["A", "C", "G", "T"], sequence_id="dna-like"
)
protein = BaseNumpySequence(
    list("ACD-"),
    sequence_id="protein-with-gap",
    alphabet=["A", "C", "D", "gap"],
)

def audit_text(sequence):
    return "".join(map(str, sequence.to_array())).replace("gap", "-")

assert audit_text(binary) == "010"
assert audit_text(multiallele) == "ACG"
assert audit_text(protein) == "ACD-"
assert binary.id == "binary-010"
assert multiallele.alphabet == ["A", "C", "G", "T"]
assert protein.sequence.flags.writeable is False

# A returned array is a copy; editing it cannot mutate the hashable sequence.
copy = binary.to_array()
copy[0] = 1
assert audit_text(binary) == "010"

# IDs do not make identical sequence values distinct for equality or hashing.
duplicate = BinarySequence([0, 1, 0], sequence_id="another-id")
assert duplicate == binary
assert hash(duplicate) == hash(binary)

print(binary.id, audit_text(binary))
print(multiallele.id, multiallele.alphabet)
print(protein.id, audit_text(protein))

# Omitting alphabet from from_string selects the canonical protein alphabet.
canonical_alphabet = list("ACDEFGHIKLMNPQRSTVWY")
default_protein = BaseNumpySequence.from_string(
    "ACDE", sequence_id="protein-default"
)
assert default_protein.alphabet == canonical_alphabet

# Equal-length, site-aligned strings retain the same default alphabet and site order.
aligned_proteins = [
    BaseNumpySequence.from_string(text, sequence_id=sequence_id)
    for sequence_id, text in [("aligned-1", "ACDE"), ("aligned-2", "ACDF")]
]
assert {len(sequence) for sequence in aligned_proteins} == {4}
assert all(sequence.alphabet == canonical_alphabet for sequence in aligned_proteins)
assert [audit_text(sequence) for sequence in aligned_proteins] == ["ACDE", "ACDF"]
```

The examples round-trip their inputs without changing site order. The
underlying sequence view is read-only; `to_array()` returns a safe copy.

`moltype` is optional interoperability metadata backed by Cogent3 and therefore
requires `landscapy[phylogeny]`. It is not required for core graph construction.
Use an explicit alphabet whenever the canonical protein default is not the
intended domain; alphabet order determines one-hot column order.

## Alignment and duplicates

Hamming and one-hot sequence geometry require a common aligned length and a
consistent alphabet. PLM embedding workflows can embed unequal raw lengths,
but that does not make site-wise Hamming comparisons valid. Gaps must represent
an alignment decision, not silently pad unrelated strings.

Duplicate sequence values are a data-model decision: IDs may identify separate
experimental rows, but equality is based on sequence content. Decide whether
duplicates are technical replicates, biological replicates, or erroneous rows
before mapping layers by sequence.

## Common failures

- Symbols outside an explicit alphabet raise an error.
- Empty sequences and non-binary values are invalid for their respective
  classes.
- Inconsistent alphabet ordering changes one-hot columns even when symbols are
  the same.
- Mutating an external source array after construction is safe, but assuming a
  sequence object itself is mutable is not.
- Treating unequal raw protein lengths as aligned site coordinates invalidates
  Hamming and single-substitution interpretations.
