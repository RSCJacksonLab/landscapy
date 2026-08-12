# ESM embedding contract

`fitness_landscape.embedding.ESMEmbedder` is the single ESM implementation.
It has two deliberately separate input paths:

- `embed_sequences()` and `extract_features()` accept one protein string or a
  sequence of strings and send integer token ids to the model;
- `embed_relaxed_seqs()` accepts one NumPy/Torch probability matrix or a
  sequence of matrices and sends expected token embeddings to the model.

Hard strings are non-empty and contain no whitespace. Each character is one
residue position. Characters are upper-cased, `-` is the gap token, and a
character absent from the tokenizer vocabulary maps to its `<unk>` token.
This manual one-character mapping prevents tokenizer-dependent changes in
sequence length.

A relaxed matrix has shape `(length, len(alphabet))`. Its values are finite and
non-negative, and every row sums to one. Columns follow the configured
`alphabet` exactly; there is no implicit left/right padding or inferred column
order. `gap` is normalized to `-`. Unsupported alphabet entries map to
`<unk>`, but only one such column is allowed so distinct scientific states
cannot silently collapse onto the same model token. Model special tokens are
not valid relaxed-alphabet columns.

Both paths prepend one `<cls>` token, append one `<eos>` token, and right-pad
with `<pad>`. The integer attention mask is one for special and residue tokens
and zero only for right padding. Mean pooling includes residue positions,
including gaps, and excludes `<cls>`, `<eos>`, and padding. Inputs are sorted by
length for batching, then restored to caller order. The result is always a
NumPy `float32` array with shape `(n_sequences, hidden_size)`; a single input
still retains the leading sequence dimension.

`lm_output_probabilities()` accepts either homogeneous input path and returns
one caller-ordered `float32` matrix per sequence. Its columns follow
`alphabet`; they are selected from the full model softmax and are not
renormalized over that subset.

Offline unit tests use a mocked tokenizer and model. The pinned integration
test uses `facebook/esm2_t6_8M_UR50D` and is skipped unless
`LANDSCAPY_RUN_ESM_INTEGRATION=1` is set.
