from pydantic import BaseModel, Field, validator, conlist, ValidationError  
from typing import Union, List, Literal, Iterable
import numpy as np
from sequence import BaseNumpySequence
from ..graph_matching.latent_alignment import RJMCMCAligner
from landscape import NodeModel, DirectedFitnessLandscape, _Record, _GraphLike, FitnessLandscape
import networkx as nx

class EmbNodeModel(BaseModel):
    emb_arr: np.ndarray = Field(..., repr=False)

    @validator("emb_arr")
    def _check_emb(cls, v):
        v = np.asarray(v)
        if v.ndim != 1:
            raise ValueError("emb_arr must be a 1-D array")
        return v

class DirectedSuperscape:
    """
    Directed superscape class to align and store multiple, independent
    directed fitness landscapes.

    Attributes
    ----------
    landscapes : List of DirectedFitnessLandscape or _GraphLike
        The independent fitness landscapes or graph-like objects.
    graph_aligner : str, default=`RJ_MCMC`
        The graph alignment algorithm. Currently only support for the
        reversible jump MCMC sampling. 
    """

    def __init__(self,
                 # Method does not suppoer undirected graphs (yet).
                 landscapes: List[Union[DirectedFitnessLandscape, _GraphLike]],
                 *,
                 graph_aligner: Literal['RJ_MCMC'] = 'RJ_MCMC',
                 **sampler_kwargs) -> None:
        
        self.landscapes = landscapes
        self._landscape_graphs = self._extract_graphs(landscapes=self.landscapes)

        self._validate_embeddings(self._landscape_graphs)

        if graph_aligner == 'RJ_MCMC':
            self.graph_aligner = RJMCMCAligner(self._landscape_graphs,
                                               **sampler_kwargs)
            self.graph_aligner.sample()
            self.posterior_mapping = self.graph_aligner.posterior_match_probabilities()
            self.supergraph = self.graph_aligner.latent_blueprint_graph()
            # Mapping of (graph idx, node) to latent node idx
            self.latent_node_mapping = self._make_slot_map(aligner=self.graph_aligner)
        
        else:
            raise ValueError("Unsupported graph alignment method")
    @staticmethod
    def _validate_embeddings(graphs: list[nx.DiGraph]) -> None:
        """
        Helper method to validate nodes have valid emb_arr attribute.

        Parameters
        ----------
        graphs : List
            List of nx.DiGraph objects to be aligned.
        """
        for G in graphs:
            for node, data in G.nodes(data=True):
                try:
                    EmbNodeModel(**data)        # will raise if missing/invalid
                except ValidationError as e:
                    raise ValueError(f"{node!r}: {e}") from None
                
    @staticmethod
    def _extract_graphs(landscapes: Iterable[Union[DirectedFitnessLandscape,
                                                   _GraphLike,
                                                   nx.DiGraph]]) -> list[nx.DiGraph]:
        """
        Helper method to extract directed graphs from directed fitness
        landscapes.

        Parameters
        ----------
        landscapes : Iterable
            The list of DirectedFitnessLandscapes, _GraphLike or
            nx.DiGraph objects.

        Returns
        -------
        out : list
            The list of nx.DiGraph objects indexed matched to the
            landscapes.
        """
        out = []
        for obj in landscapes:
            if isinstance(obj, DirectedFitnessLandscape):
                G = obj.graph
            elif isinstance(obj, nx.Graph):
                G = nx.DiGraph(obj)  # copy/upgrade
            elif isinstance(obj, _GraphLike):
                G = nx.DiGraph(obj) # last‑resort
            else:
                raise TypeError(f"Unsupported landscape/graph type: {type(obj)}")
            if not isinstance(G, nx.DiGraph):
                G = nx.DiGraph(G)
            out.append(G)
        return out
    
    @staticmethod
    def _make_slot_map(aligner: RJMCMCAligner) -> dict[tuple[int, str], int]:
        """
        Helper method to convert permutations into a node-to-slot
        mapping dictionary. 

        Parameters
        ----------
        aligner : RJCMCMCALigner
            The Reverse jump MCMC graph aligner with the stored
            posterior and graph permutations.s
        
        Returns
        -------
        mapping : Dict
            (graph_index, node_id): slot_id mapping.
        """
        mapping = {}
        for k, (pk, nodes) in enumerate(zip(aligner.perm, aligner.V)):
            for idx, slot in enumerate(pk):
                mapping[(k, nodes[idx])] = int(slot)
        return mapping
    
    # TODO: shard with FAISS and retrieve subgraph with cosine match to
    # the query vector. Current method scales O(N^2) over exhaustive
    # graph alignment (even with anchoring): subgraphing will scale
    # linearly.