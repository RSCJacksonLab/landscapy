# Graph construction and representation choice

An edge is a modelling decision. These recipes construct every graph type in
the supported 0.9 release and make node order, geometry, parameters, components,
and edge meaning auditable.

1. [Exact Hamming graphs](hamming.md)
2. [OHE k-nearest neighbours for non-binary sequences](knn-sequence-space.md)
3. [PLM-embedding kNN graphs](plm-knn.md)
4. [Embedding-diffusion graphs](embedding-diffusion.md)
5. [Evolutionary-diffusion graphs](evolutionary-diffusion.md)
6. [TDA alpha-complex graphs](tda.md)
7. [Phylogenetic topology from an alignment](phylogenetic-topology.md)
8. [External adjacency matrices and edge tables](external-adjacency.md)
9. [Compare graph representations](compare-representations.md)

Formal contracts: [constructor inputs](../../graph_constructors.md), [kNN
embedding domains](../../knn_embedding_domains.md), [diffusion
semantics](../../diffusion_semantics.md), and [edge
semantics](../../edge_semantics.md). All examples use the [supported 0.9 public
API](../../public_api.md).
