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
