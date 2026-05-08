from . import core
from . import transforms
from . import analysis
from . import models
from . import phylo
from . import io

from .core import (
    FitnessLandscape,
    DirectedFitnessLandscape,
    BaseNumpySequence,
    BinarySequence,
    MultialleleSequence,
    generate_sequences,
    sequence_distance,
    create_hamming_graph,
    create_knn_graph,
    create_trajectory_digraph,
    NumericFitness,
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
)
from .io import export_lsbundle, load_bundle_dir, save_bundle_dir

__version__ = '0.9.0'

__all__ = [
    'core',
    'transforms',
    'analysis',
    'models',
    'io',
    'BaseNumpySequence',
    'BinarySequence',
    'MultialleleSequence',
    'generate_sequences',
    'sequence_distance',
    'FitnessLandscape',
    'DirectedFitnessLandscape',
    'create_hamming_graph',
    'create_knn_graph',
    'create_trajectory_digraph',
    'NumericFitness',
    'CategoricalFitness',
    'ProbabilisticCategoricalFitness',
    'save_bundle_dir',
    'load_bundle_dir',
    'export_lsbundle',
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
