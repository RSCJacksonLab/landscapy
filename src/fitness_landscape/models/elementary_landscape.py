from typing import Optional, List, Literal
import numpy as np
from ..core.landscape import FitnessLandscape
from ..core.sequence import BaseNumpySequence, generate_sequences
from ..core.graph import create_knn_graph, create_hamming_graph
from ..analysis.eigenmode import eigenmode_decomposition

class ElementaryFitnessLandscape(FitnessLandscape):
    """
    Elementary fitness landscape subclass of FitnessLandscape. The
    fitness function of an elementary fitness landscape is an
    eigenfunction of the graph Laplacian.

    Attributes
    ----------

    """
    def __init__(self,
                 j: int,
                 N: int = None,
                 sequences: List[BaseNumpySequence] = None,
                 seed: Optional[int] = None,
                 alphabet: List = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
                 graph_type: Literal['knn', 'hamming'] = 'hamming',
                 **kwargs):

        if sequences is None and N is None:
            raise ValueError("Either `sequences` or `N` must be provided.")
        
        # CORRECTED: Infer N from sequences if not provided
        if N is None and sequences is not None:
            N = len(sequences[0])

        if graph_type == 'knn':
            if sequences is None:
                raise ValueError("`sequences` must be provided for kNN graph.")
            graph = create_knn_graph(sequences=sequences,
                                     k=int(np.sqrt(len(sequences))))

        elif graph_type == 'hamming':
            if N is None:
                raise ValueError("`N` must be provided for Hamming graph.")
            if sequences is None:
                sequences = generate_sequences(N, alphabet)
            graph = create_hamming_graph(sequences)
        
        eigenvalues, eigenvectors = eigenmode_decomposition(graph)
        fitness_values = eigenvectors[:, j]
        
        super().__init__(sequences=sequences,
                         fitness_values=fitness_values,
                         graph_type=graph_type,
                         **kwargs)