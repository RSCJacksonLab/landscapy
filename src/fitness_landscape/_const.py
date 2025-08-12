from cogent3 import get_moltype

# Standard alphabetical order alphabet.
PROT = get_moltype("protein")
PROT_20 = [aa for aa in PROT.alphabet if aa != 'U']
ALPHABET_21 = PROT_20 + ["gap"]

# PAML ordered matrices (piqtree)
PAML_20 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']