# Export landscapes for downstream machine learning

Landscapy provides validated sequence, layer, graph, and tensor exports for
external machine-learning code. It does not provide a complete training,
evaluation, or causal-inference framework.

## Recipes

- [Prepare aligned features, targets, and split annotations](feature-and-target-preparation.md)
- [Export sequence tensors](sequence-tensor-export.md)
- [Export a PyTorch Geometric graph](pytorch-geometric-export.md)
- [Add predictions back without replacing measurements](adding-predictions-back-to-a-landscape.md)
- [Audit evaluation support without target leakage](evaluation-without-leakage.md)

Protein language-model weights are separate versioned inputs and may require a
network download. The examples use the repository's synthetic [version 1.0
fixture](../data/README.md) and fixed, offline feature arrays.
