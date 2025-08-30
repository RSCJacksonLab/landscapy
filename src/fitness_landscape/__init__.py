from . import core
from . import transforms
from . import analysis
from . import models
from . import graph_matching
from . import phylo

from .core import (
    FitnessLandscape,
    BaseNumpySequence,
    BinarySequence,
    MultialleleSequence,
    generate_sequences,
    sequence_distance,
    create_hamming_graph,
    create_knn_graph,
    NumericFitness,
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
    FitnessSuperscape
)

__version__ = '0.9.0'

__all__ = [
    'core',
    'transforms',
    'analysis',
    'models',
    'BaseNumpySequence',
    'BinarySequence',
    'MultialleleSequence',
    'generate_sequences',
    'sequence_distance',
    'FitnessLandscape',
    'create_hamming_graph',
    'create_knn_graph',
    'NumericFitness',
    'CategoricalFitness',
    'ProbabilisticCategoricalFitness'
    'FitnessSuperscape'
]

# Compatibility patch for cogent3 MolType.make_seq positional API changes
try:
    from cogent3.core.moltype import MolType
except Exception:
    MolType = None

if MolType is not None:
    _orig_make_seq = MolType.make_seq
    def _compat_make_seq(self, *args, **kwargs):
        # Newer cogent3 requires keyword-only; support old positional form
        if args and 'seq' not in kwargs and 'data' not in kwargs:
            # accommodate either 'seq' or 'data' keyword accepted by versions
            try:
                return _orig_make_seq(self, seq=args[0], **kwargs)
            except TypeError:
                return _orig_make_seq(self, data=args[0], **kwargs)
        return _orig_make_seq(self, **kwargs)
    MolType.make_seq = _compat_make_seq
