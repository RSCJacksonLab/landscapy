import pytest

import fitness_landscape.core.landscape as L


def test_from_sequences_evol_diffusion_calls_constructor(monkeypatch):
    called = {}

    def fake_ctor(seqs, **kwargs):
        called['kwargs'] = kwargs
        # Minimal fake graph object with required API
        import networkx as nx
        G = nx.Graph()
        for i, s in enumerate(seqs):
            G.add_node(i, sequence=s)
        return G

    monkeypatch.setattr(L, 'create_evol_diffusion_graph', fake_ctor)

    from fitness_landscape.core.sequence import generate_sequences
    from fitness_landscape.core.fitness import NumericFitness

    seqs = generate_sequences(length=2, alphabet=[0, 1])
    layers = {"default": NumericFitness(name="default", values=[[1.0] for _ in seqs])}

    FL = L.FitnessLandscape.from_sequences(
        sequences=seqs,
        fitness_layers=layers,
        graph_type='evol_diffusion',
        embeddings=None,
        embedding_domain='ohe',
        k=10,
        t=2,
        tau=0.9,
        connectivity_threshold=1e-3,
        backend='auto',
    )

    # Ensure our fake constructor was used and kwargs passed through
    assert 'kwargs' in called
    assert called['kwargs']['k'] == 10
    assert called['kwargs']['t'] == 2
    assert called['kwargs']['tau'] == 0.9
    assert called['kwargs']['connectivity_threshold'] == 1e-3
    assert FL.graph.number_of_nodes() == len(seqs)

