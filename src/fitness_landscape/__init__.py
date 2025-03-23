"""
Fitness landscape package for analyzing functions over network graphs.

This package provides tools for analyzing fitness landscapes modeled as network graphs,
with efficient implementations of Walsh-Hadamard transformations, graph Fourier transforms,
and eigenmode decomposition.
"""

from . import core
from . import transforms
from . import analysis
from . import models
from . import utils

# Import key classes and functions for convenience
from .core.sequence import (
    Sequence,
    BinarySequence,
    MultialleleSequence,
    generate_sequences,
    sequence_distance
)

from .core.landscape import FitnessLandscape

from .core.graph import (
    create_hamming_graph,
    create_knn_graph,
    graph_properties
)

__version__ = '0.1.0'
__all__ = [
    'core',
    'transforms',
    'analysis',
    'models',
    'utils',
    'Sequence',
    'BinarySequence',
    'MultialleleSequence',
    'generate_sequences',
    'sequence_distance',
    'FitnessLandscape',
    'create_hamming_graph',
    'create_knn_graph',
    'graph_properties'
]
