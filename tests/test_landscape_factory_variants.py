import networkx as nx

import fitness_landscape.core.landscape as L


def test_build_variants_call_correct_ctors(monkeypatch):
    called = []

    def fake_knn(seqs, **kwargs):
        called.append(('knn', kwargs))
        G = nx.Graph()
        for i, s in enumerate(seqs):
            G.add_node(i, sequence=s)
        return G

    def fake_tda(seqs, **kwargs):
        called.append(('tda', kwargs))
        G = nx.Graph()
        for i, s in enumerate(seqs):
            G.add_node(i, sequence=s)
        return G

    def fake_diff(seqs, **kwargs):
        called.append(('diffusion', kwargs))
        G = nx.Graph()
        for i, s in enumerate(seqs):
            G.add_node(i, sequence=s)
        return G

    monkeypatch.setattr(L, 'create_knn_graph', fake_knn)
    monkeypatch.setattr(L, 'create_tda_graph', fake_tda)
    monkeypatch.setattr(L, 'create_diffusion_emb_graph', fake_diff)

    from fitness_landscape.core.sequence import generate_sequences
    from fitness_landscape.core.fitness import NumericFitness

    seqs = generate_sequences(length=2, alphabet=[0, 1])
    layers = {"default": NumericFitness(name="default", values=[[1.0] for _ in seqs])}

    # call each type and ensure attach_embeddings toggle works
    for gtype in ('knn', 'tda', 'diffusion'):
        FL = L.FitnessLandscape.build(
            sequences=seqs,
            fitness_layers=layers,
            graph=gtype,
            attach_embeddings=False,
        )
        assert isinstance(FL.graph, nx.Graph)

    kinds = [k for k, _ in called]
    assert set(kinds) == {'knn', 'tda', 'diffusion'}
