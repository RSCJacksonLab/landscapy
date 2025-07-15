from typing import Optional, List
from ..core.landscape import FitnessLandscape
from ..core.sequence import BaseNumpySequence
from ..core.graph import create_knn_graph
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
                 sequences: List[BaseNumpySequence],
                 j: int,
                 k: int,
                 seed: Optional[int] = None,
                 graph_type: str = 'knn',
                 **kwargs):

        graph = create_knn_graph(sequences=sequences,
                                 k=k)
        _, eigenvectors = eigenmode_decomposition(graph=graph,
                                                  matrix='laplacian',
                                                  backend='numpy')
        
        fitness_signal = eigenvectors[:, j]
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]['fitness'] = fitness_signal[i]
        
        self.k = k
        self.eigenvector_index = j
        self.seed = seed

        super().__init__(graph=graph,
                         **kwargs)