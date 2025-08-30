from fitness_landscape.utils import sanitize_alignment
from cogent3 import make_aligned_seqs


def test_sanitize_alignment_replaces_illegal_and_unique_names():
    # Use '?' (allowed) to represent missing, and 'X' (allowed wildcard) for unknown
    aln = make_aligned_seqs({
        'a': 'ACD?',
        'a': 'ACD?',  # duplicate name will be de-duped
        'b': 'ACDX',  # X should be replaced by gap by sanitizer
    }, moltype='protein')
    clean = sanitize_alignment(aln)
    # Names must be unique
    assert len(set(clean.names)) == len(clean.names)
    # No '?' remains; X becomes '-'
    for name in clean.names:
        s = str(clean.get_gapped_seq(name))
        assert '?' not in s
        # All characters in allowed AA + '-' set
        assert set(s).issubset(set('ACDEFGHIKLMNPQRSTVWY-'))
