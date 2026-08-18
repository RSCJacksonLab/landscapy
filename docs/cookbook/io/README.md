# Saving, sharing, CLI use, and visualization

These recipes turn an in-memory landscape into an inspectable artifact and
show how to visualize results with external tools. Landscapy  deliberately
has no plotting API; NetworkX, Matplotlib, and Cytoscape remain separate tools.

1. [CSV import and export](csv-import-and-export.md)
2. [Portable directory bundles](portable-directory-bundles.md)
3. [Deterministic `.lsbundle` archives](lsbundle-archives.md)
4. [Provenance and integrity](provenance-and-integrity.md)
5. [Command-line workflows](command-line-workflows.md)
6. [Cytoscape and Python visualization](cytoscape-and-python-visualization.md)

The canonical publication format is the portable bundle described in the
[repository README](../../../README.md#portable-landscape-export). Bundle
classes and exceptions are part of the [0.9 public API](../foundations/public-api.md).
The [release scope](../../../README.md#release-scope) excludes built-in
visualization.
