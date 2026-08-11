import networkx as nx

import fitness_landscape.core.landscape as L


def test_directed_build_diffusion_nq(monkeypatch):
    called = {}

    def fake_digraph(seqs, **kwargs):
        called['ok'] = True
        G = nx.DiGraph()
        for i, s in enumerate(seqs):
            G.add_node(i, sequence=s)
        return G

    monkeypatch.setattr(L, 'create_evol_diffusion_digraph', fake_digraph)

    from fitness_landscape.core.sequence import generate_sequences

    seqs = generate_sequences(length=2, alphabet=[0, 1])
    DG = L.DirectedFitnessLandscape.build(
        sequences=seqs,
        digraph='diffusion_nq',
        embedding_domain='ohe',
    )

    assert isinstance(DG.graph, nx.DiGraph)
    assert called.get('ok', False)
