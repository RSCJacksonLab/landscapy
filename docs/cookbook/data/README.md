# Cookbook example data

`toy_landscape.csv` is a synthetic, deterministic table created for the
Landscapy documentation. It enumerates the complete three-site binary sequence
cube and assigns illustrative assay values, replicate measurements, activity
classes, experimental backgrounds, split labels, and taxonomy labels.

- Version: 1.0 (introduced with cookbook issue #218)
- Source: generated for this repository; no external data were used
- Licence: MIT, matching the repository licence
- Provenance: values were chosen by hand to exercise APIs and edge cases
- Intended use: executable documentation and tests only
- Biological status: none; values and labels must not be interpreted as a real
  genotype–phenotype relationship

The primary key is `sequence`. Rows are in lexical binary order. Every recipe
that combines arrays with graph nodes must preserve or explicitly verify this
order.

## Schema

| Column | Type | Meaning |
| --- | --- | --- |
| `sequence` | three-character string | aligned binary sequence; leading zeroes are significant |
| `fitness` | float | illustrative scalar assay value |
| `replicate_1`, `replicate_2` | float | illustrative replicate measurements |
| `activity_class` | string | derived low/mid/high assay category |
| `background` | string | illustrative experimental background |
| `split` | string | illustrative train/test assignment |
| `taxonomy` | string | illustrative grouping annotation |

## Protein graph-construction fixtures

`toy_proteins.fasta` contains six synthetic aligned seven-residue protein
sequences. They were designed only to create small, auditable graph examples.
They are not derived from a natural protein family.

`toy_protein_embeddings.csv` is a four-component PCA cache derived from
mean-pooled `facebook/esm2_t6_8M_UR50D` embeddings of those six sequences. The
source model revision was
`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`. Embeddings were generated with
Landscapy `ESMEmbedder`, PyTorch 2.13.0, and Transformers 5.15.0 on CPU, then
reduced with scikit-learn 1.9.0 full-SVD PCA. The four components explain
approximately 99.49% of variance in this six-sequence fixture. The cache is
rounded to nine significant digits and is intended only for offline executable
documentation. It is not a benchmark or reusable biological representation.

Both files are version 1.0, generated for this repository, and distributed
under the repository MIT licence. Sequence order in the FASTA and embedding CSV
is identical and must be asserted before graph construction.

SHA-256 checksums for the version 1.0 fixtures are:

| File | SHA-256 |
| --- | --- |
| `toy_landscape.csv` | `3a88dc7441f18fa630fd6e748755d455b67b9f2f03a00cebaa6874aad302bb08` |
| `toy_proteins.fasta` | `9f543f96edcae0e269ecf5bce5a0986391e151914ab77a87a8521d8c6640a41d` |
| `toy_protein_embeddings.csv` | `d0783f5e58049f72bfa0c317ae8b2f854e15c9642a62e7e505929878dd478243` |
