# Prepare sequence objects

Sequence values, order, alphabet, and identifiers are separate concepts. Choose
the narrowest class that represents the data.

## Install and input

```bash
python -m pip install landscapy
```

Inputs are strings or one-dimensional iterables. Binary values must be `0` or
`1`; multiallelic values must occur in the declared alphabet; protein symbols
should use an explicit alphabet. Gaps are represented consistently as `-` or
`gap` according to the chosen alphabet.

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
```

Expected output round-trips all three inputs without changing site order. The
underlying sequence view is read-only; `to_array()` returns a safe copy.

`moltype` is optional interoperability metadata backed by Cogent3 and therefore
requires `landscapy[phylogeny]`. It is not required for core graph construction.
An explicit alphabet is still required to make symbol validation and one-hot
column order auditable.

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
