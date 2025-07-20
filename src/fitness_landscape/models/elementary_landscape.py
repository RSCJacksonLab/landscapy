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
    

        if graph_type == 'knn':
            if sequences is None:
                raise ValueError("`sequences` must be provided for kNN graph.")
            graph = create_knn_graph(sequences=sequences,
                                    k=int(np.sqrt(len(sequences))))
        
        elif graph_type == 'hamming':
            if N is None:
                raise ValueError("`N` must be provided for Hamming graph.")
            sequences = generate_sequences(length=N, alphabet=alphabet)
            graph = create_hamming_graph(sequences=sequences)
        
        _, eigenvectors = eigenmode_decomposition(graph=graph,
                                                  matrix='laplacian',
                                                  backend='numpy')
        fitness_signal = eigenvectors[:, j]
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]['fitness'] = fitness_signal[i]
        
        self.eigenvector_index = j
        self.seed = seed

        super().__init__(graph=graph,
                         **kwargs)