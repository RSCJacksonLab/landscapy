from typing import Optional, List, Literal
import numpy as np
from ..core.landscape import FitnessLandscape
from ..core.sequence import BaseNumpySequence, generate_sequences
from ..core.graph import create_knn_graph, create_hamming_graph
from ..analysis.eigenmode import eigenmode_decomposition
from ..core.fitness import NumericFitness # Import NumericFitness

def create_elementary_landscape(j: int,
                                sequences: List[BaseNumpySequence]=None,
                                graph_type: Literal['knn', 'hamming'] = 'hamming',
                                **kwargs) -> FitnessLandscape:
    """
    Factory function to create an elementary fitness landscape, where
    the fitness function is an eigenfunction of the graph Laplacian.

    Parameters
    ----------
    j : int
        The eigenfunction index to use as fitness signal.
    
    sequences : List[BaseNumpySequence], default=`None`
        List of optional sequences to construct the graph from. If
        `None`, combinatorially complete sequence dataset is 
        constructed and used. 
    
    graph_type : str, default=`hamming`
        The graph type to use.

        
    Returns
    -------
    FitnessLandscape
        The constructed elementary landscape.
    """
    if sequences is None:
        # Consistent kwargs with NK landscape
        sequences = generate_sequences(length=kwargs.get('N', 5),
                                       alphabet=kwargs.get('alphabet',
                                                           [0,1]))
    if not isinstance(sequences, BaseNumpySequence):
        sequences = [BaseNumpySequence(seq) for seq in sequences]
        
    if graph_type == 'knn':
        k = kwargs.get('k', int(np.sqrt(len(sequences))))
        graph = create_knn_graph(sequences=sequences, k=k)
    elif graph_type == 'hamming':
        graph = create_hamming_graph(sequences)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    _, eigenvectors = eigenmode_decomposition(graph)
    fitness_values = eigenvectors[:, j]

    replicates = [[val] for val in fitness_values]
    
    fitness_layers = {
        f'elementary_eign_index={j}': NumericFitness(
            name=f'elementary_eign_index={j}',
            values=replicates,
            metadata={'eigenvector_index': j,
                      'N' : kwargs.get('N', 5),
                      'alphabet' : kwargs.get('alphabet', [0,1]),
                      'graph_type' : graph_type}
        )
    }
    
    return FitnessLandscape(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph=graph, # Pass the pre-computed graph to the constructor
        **kwargs
    )