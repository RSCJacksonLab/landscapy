import pytest
import numpy as np

from fitness_landscape.utils import (
    alignment_to_base_numpy_sequences,
    fasta_to_prot20_sequences,
    sanitize_alignment,
)
from cogent3 import make_aligned_seqs


def test_alignment_to_base_numpy_sequences_global_ungap():
    aln = make_aligned_seqs(
        {
            "a": "AC-D",
            "b": "ACGD",
            "c": "AC-D",
        },
        moltype="protein",
    )
    aln = sanitize_alignment(aln)
    seqs = alignment_to_base_numpy_sequences(aln)
    # columns with any '-' dropped globally: keep A,C,G,D -> but G only present where no '-' in any sequence
    # Here the third column has '-' for a and c, so drop; keep others => length 3
    assert len(seqs) == 3
    lengths = {len(s) for s in seqs}
    assert lengths == {3}


def test_fasta_to_prot20_sequences_unaligned(tmp_text):
    p = tmp_text(
        "unaligned.fasta",
        ">s1\nACD.EFG\n>s2\nACDEFG-\n",
    )
    seqs = fasta_to_prot20_sequences(p)
    # Dots/gaps removed; sequences equal length
    assert len(seqs) == 2
    assert all(len(s) == 6 for s in seqs)


def test_fasta_to_prot20_sequences_alignment(tmp_text):
    p = tmp_text(
        "aligned.fasta",
        ">a\nAC-D\n>b\nACGD\n>c\nAC-D\n",
    )
    seqs = fasta_to_prot20_sequences(p)
    assert len(seqs) == 3
    lengths = {len(s) for s in seqs}
    assert lengths == {3}

def test_fasta_to_prot20_sequences_alignment_with_ambiguous(tmp_text):
    p = tmp_text(
        "ambig.fasta",
        ">a\nACX\n>b\nACY\n",
    )
    seqs = fasta_to_prot20_sequences(p, strict=False)
    assert len(seqs) == 2
    assert {len(s) for s in seqs} == {2}
    recovered = {''.join(seq.to_array()) for seq in seqs}
    assert recovered == {"AC"}

def test_fasta_to_prot20_sequences_return_gapped_alignment(tmp_text):
    p = tmp_text(
        "aligned_gapped.fasta",
        ">a\nAC-D\n>b\nACGD\n",
    )
    seqs, aligned = fasta_to_prot20_sequences(p, strict=False, return_gapped=True)
    assert len(seqs) == 2
    assert aligned is not None
    assert len(aligned) == 2
    assert {len(s) for s in aligned} == {4}
    assert ''.join(aligned[0].to_array()) == "AC-D"

def test_fasta_to_prot20_sequences_alignment_all_ambiguous(tmp_text):
    p = tmp_text(
        "ambig_full.fasta",
        ">a\nXXX\n>b\nXXX\n",
    )
    seqs = fasta_to_prot20_sequences(p, strict=False)
    # All characters were ambiguous; fallback path should not raise.
    assert seqs == []

def test_fasta_to_prot20_sequences_alignment_mixed_ambiguous(tmp_text):
    p = tmp_text(
        "ambig_mixed.fasta",
        ">a\nACD\n>b\nXXX\n",
    )
    seqs = fasta_to_prot20_sequences(p, strict=False)
    assert len(seqs) == 1
    assert ''.join(seqs[0].to_array()) == "ACD"

def test_fasta_to_prot20_sequences_gapped_alignment_all_gap_columns(tmp_text):
    p = tmp_text(
        "all_gap_cols.fasta",
        ">a\nA-\n>b\n-A\n",
    )
    seqs, aligned = fasta_to_prot20_sequences(p, strict=False, return_gapped=True)
    assert len(seqs) == 2
    assert {''.join(seq.to_array()) for seq in seqs} == {"A"}
    assert aligned is not None
    assert {len(s) for s in aligned} == {2}

def test_fasta_to_prot20_sequences_return_gapped_unaligned(tmp_text):
    p = tmp_text(
        "unaligned.fasta",
        ">s1\nABC\n>s2\nABCD\n",
    )
    seqs, aligned = fasta_to_prot20_sequences(p, strict=False, return_gapped=True)
    assert len(seqs) == 2
    assert aligned is None

def test_fasta_to_prot20_sequences_raises_on_noncanonical(tmp_text):
    p = tmp_text(
        "bad.fasta",
        ">s\nACXDEF\n",
    )
    with pytest.raises(ValueError):
        fasta_to_prot20_sequences(p)
