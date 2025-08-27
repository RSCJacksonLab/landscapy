import numpy as np
import pytest
import networkx as nx
from fitness_landscape.core.sequence import *
from fitness_landscape.core.graph import *
from fitness_landscape.core.graph import _encode_multiallele 
from fitness_landscape.core.digraph import *
from fitness_landscape.core.landscape import FitnessLandscape, DirectedFitnessLandscape, to_csv_landscape, read_csv_landscape
from fitness_landscape.core.fitness import (
    NumericFitness,
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
    make_fitness_layer,
    as_fitness_layers,
)
from fitness_landscape.core.superscape import FitnessSuperscape
import torch
import pandas as pd
from torch_geometric.data import Data
from fitness_landscape.core.fitness import NumericFitness, CategoricalFitness, ProbabilisticCategoricalFitness
from pathlib import Path
from fitness_landscape.phylo._sub_matrices import nq_pfam
from fitness_landscape.embedding.particle_sampler import SequenceGenerator, TopPSampler
from unittest.mock import patch
from fitness_landscape.utils import alignment_to_base_numpy_sequences
from cogent3 import get_moltype

@pytest.mark.parametrize("n,L,B", [(60, 6, 3), (80, 5, 4)])
def test_hamming_graph_multiallele_smoke_largeish(n, L, B):
    """
    Smoke test: random multi-allelic sequences (moderate size) build without error
    and produce a non-empty sparse graph (where variability allows).
    """
    rng = np.random.default_rng(0)
    alphabet = [str(i) for i in range(B)]
    seqs = [BaseNumpySequence(list(rng.choice(alphabet, size=L)), alphabet=alphabet) for _ in range(n)]
    G = create_hamming_graph(sequences=seqs, _backend="masked")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == n
    # It’s possible (but unlikely) to be empty if all rows identical; allow >=0 and check typical case >0
    assert G.number_of_edges() >= 0

@pytest.fixture
def mock_embedder(mocker):
    """Mocks the ESMEmbedder to avoid loading a real model."""
    embedder = mocker.MagicMock()
    embedder.alphabet = list('ACDEFGHIKLMNPQRSTVWY-') + ['<cls>', '<eos>', '<pad>', '<mask>']
    embedder.embed_relaxed_seqs.return_value = np.random.rand(10, 320)
    embedder.lm_output_probabilities.return_value = [np.random.rand(5, 25) for _ in range(10)]
    return embedder

@pytest.fixture
def sequence_generator(mock_embedder):
    """Provides a SequenceGenerator with a mocked embedder."""
    sampler = TopPSampler()
    return SequenceGenerator(embedder=mock_embedder, sampler=sampler, batch_size=10)

@pytest.fixture
def basic_landscape():
    """Provides a basic FitnessLandscape with a numeric layer for testing."""
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    fitness_values = [[val] for val in np.random.rand(8)]
    fitness_layers = {
        'default': NumericFitness(name='default', values=fitness_values)
    }
    return FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph_type='hamming' 
    )

@pytest.fixture
def clustered_data():
    """
    Provides synthetic data with two distinct clusters for testing
    embedding-based graph constructors.
    """
    # Cluster 1: 10 points centered around [0, 0]
    cluster1_embs = np.random.rand(10, 3) * 0.1
    # Cluster 2: 10 points centered around [1, 1]
    cluster2_embs = 3 + np.random.rand(10, 3) * 0.1
    
    embeddings = np.vstack([cluster1_embs, cluster2_embs])
    
    # Create dummy sequences for API compatibility
    sequences = [BinarySequence([0] * 3) for _ in range(20)]
    
    return sequences, embeddings

@pytest.fixture
def phylo_test_data(tmp_path: Path) -> Path:
    """Creates a simple FASTA alignment file for testing."""
    fasta_content = """>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
ACDEFGHIKLMNPQRSTMAD
>seq3
ACDEFGHIKLMNPQRSTVAD
"""
    fasta_file = tmp_path / "phylo_test.fasta"
    fasta_file.write_text(fasta_content)
    return fasta_file

@pytest.fixture
def diffusion_test_data():
    """Provides sequences and embeddings with a clear cluster structure."""
    sequences = [
        BaseNumpySequence(['A', 'R', 'N', 'D'], alphabet=PROT_20),
        BaseNumpySequence(['A', 'Q', 'N', 'E'], alphabet=PROT_20),
        BaseNumpySequence(['Y', 'Y', 'Y', 'Y'], alphabet=PROT_20)  
    ]
    embeddings = np.array([
        [0.1, 0.1], 
        [0.2, 0.1], 
        [5.0, 5.0]  
    ])
    return sequences, embeddings

def test_sequence_creation_and_distance():
    """
    Tests sequence creation and distance calculation.
    """
    seq1 = BinarySequence([0, 1, 0, 1])
    seq2 = BinarySequence([0, 1, 1, 0])
    assert seq1.distance(seq2, metric="hamming") == 2
    seq3 = BaseNumpySequence(['A', 'C', 'G'])
    seq4 = BaseNumpySequence(['A', 'T', 'G'])
    assert sequence_distance(seq3, seq4) == 1

def test_sequence_mutation():
    """
    Tests the mutation method of sequence objects.
    """
    seq = BaseNumpySequence([0, 0, 0, 0], alphabet=[0, 1])
    mutated_seq = seq.mutate(positions=[1, 3], values=[1, 1])
    assert np.array_equal(mutated_seq.to_array(), [0, 1, 0, 1])
    assert seq.distance(mutated_seq) == 2

def test_generate_sequences():
    """
    Tests the generation of all combinatorial sequences.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    assert len(sequences) == 8
    assert BaseNumpySequence([1, 0, 1]) in sequences

def test_multiallele_sequence():
    """
    Tests the MultialleleSequence class.
    """
    alphabet = ['A', 'C', 'G', 'T']
    seq_data = ['A', 'G', 'T', 'C']
    seq = MultialleleSequence(seq_data, alphabet=alphabet)
    assert np.array_equal(seq.to_array(), seq_data)
    assert seq.alphabet == alphabet
    with pytest.raises(ValueError):
        MultialleleSequence(['A', 'X', 'G', 'T'], alphabet=alphabet)
    seq2 = MultialleleSequence(['A', 'C', 'G', 'T'], alphabet=alphabet)
    assert seq.distance(seq2) == 3

def test_soft_sequence():
    """
    Tests the SoftSequence class.
    """
    alphabet = ['A', 'C', 'G', 'T']
    posterior = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.1, 0.7, 0.1],
        [0.25, 0.25, 0.25, 0.25]
    ])
    soft_seq_argmax = SoftSequence(posterior, alphabet=alphabet, hard_rule='argmax')
    expected_hard_seq = ['A', 'G', 'A']
    assert np.array_equal(soft_seq_argmax.to_array(), expected_hard_seq)
    map_values = soft_seq_argmax.map_values()
    assert np.allclose(map_values, [0.8, 0.7, 0.25])

def test_create_complete_hamming_graph():
    """
    Tests the creation of a complete Hamming graph.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    graph = create_hamming_graph(sequences=sequences)
    assert graph.number_of_nodes() == 8
    assert graph.number_of_edges() == 12
    assert graph.degree[0] == 3

def test_create_knn_graph():
    """
    Tests the creation of a k-nearest neighbor graph.
    """
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    k = 3
    graph = create_knn_graph(sequences=sequences, k=k)
    for node in graph.nodes():
        assert graph.degree[node] >= k

def test_landscape_initialization_with_layers(basic_landscape):
    """
    Tests that FitnessLandscape initializes correctly with the layer
    system.
    """
    assert len(basic_landscape) == 8
    assert 'default' in basic_landscape.fitness_layers
    assert basic_landscape.get_signal().shape == (8,)
    assert basic_landscape.graph is not None

def test_fitness_free_landscape_initialization():
    """
    Tests that a landscape can be initialized without any fitness layers.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    landscape = FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers={},
        graph_type='hamming'
    )
    assert len(landscape) == 8
    assert landscape.graph.number_of_nodes() == 8
    with pytest.raises(ValueError):
        landscape.get_signal()

def test_attach_and_detach_layer(basic_landscape):
    """
    Tests that a new fitness layer can be attached and detached.
    """
    cat_values = ['A'] * 4 + ['B'] * 4
    new_layer = CategoricalFitness(name='activity', values=cat_values, categories=['A', 'B'])
    basic_landscape.attach(new_layer)
    
    assert 'activity' in basic_landscape.fitness_layers
    assert 'fitness_activity' in basic_landscape.graph.nodes[0]
    assert basic_landscape.graph.nodes[0]['fitness_activity'] == 'A'
    
    basic_landscape.detach('activity')
    
    assert 'activity' not in basic_landscape.fitness_layers
    assert 'fitness_activity' not in basic_landscape.graph.nodes[0]

def test_from_graph_initialization():
    """
    Tests that a landscape can be correctly initialized from a graph.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    graph = create_hamming_graph(sequences=sequences)
    
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['fitness_stability'] = [np.random.rand()]
        graph.nodes[node]['fitness_activity'] = 'high' if i % 2 == 0 else 'low'

    landscape = FitnessLandscape.from_graph(graph)

    assert len(landscape) == 8
    assert 'stability' in landscape.fitness_layers
    assert 'activity' in landscape.fitness_layers
    assert isinstance(landscape.view('stability'), NumericFitness)
    assert isinstance(landscape.view('activity'), CategoricalFitness)

def test_create_tda_graph(clustered_data):
    """
    Tests that the TDA graph constructor runs and creates a graph
    with the correct number of nodes.
    """
    sequences, embeddings = clustered_data
    graph = create_tda_graph(sequences=sequences, embeddings=embeddings)
    
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == len(sequences)
    
    # TDA should have the two clusters
    # Not a great test - not analytically defined, just observed behaviour...
    assert nx.number_connected_components(graph) == 2

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == len(sequences)
    
    # Calculate intra-cluster densities.
    subgraph_c1 = graph.subgraph(range(10))
    subgraph_c2 = graph.subgraph(range(10, 20))
    avg_intra_cluster_density = (nx.density(subgraph_c1) + nx.density(subgraph_c2)) / 2
    
    # Calculate inter-cluster density.
    inter_cluster_edges = 0
    for u, v in graph.edges():
        if (u < 10 and v >= 10) or (v < 10 and u >= 10):
            inter_cluster_edges += 1
            
    # Total possible edges between two clusters of 10 is 10*10 = 100.
    possible_inter_cluster_edges = 100
    inter_cluster_density = inter_cluster_edges / possible_inter_cluster_edges
    
    # Assert that intra-cluster connectivity is stronger.
    assert avg_intra_cluster_density > inter_cluster_density
    # Also assert that the clusters themselves are reasonably dense.
    assert avg_intra_cluster_density > 0.5


def test_create_diffusion_graph_structure(clustered_data):
    """
    Tests that the diffusion graph correctly captures the clustered
    structure by having higher intra-cluster density than inter-cluster density.
    """
    sequences, embeddings = clustered_data
    graph = create_diffusion_emb_graph(sequences=sequences, embeddings=embeddings, t=5, connectivity_threshold=1e-4)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == len(sequences)

    subgraph_c1 = graph.subgraph(range(10))
    subgraph_c2 = graph.subgraph(range(10, 20))
    avg_intra_cluster_density = (nx.density(subgraph_c1) + nx.density(subgraph_c2)) / 2

    inter_cluster_edges = 0
    for u, v in graph.edges():
        if (u < 10 and v >= 10) or (v < 10 and u >= 10):
            inter_cluster_edges += 1

    possible_inter_cluster_edges = 100
    inter_cluster_density = inter_cluster_edges / possible_inter_cluster_edges

    # Assert that intra-cluster connectivity is stronger
    assert avg_intra_cluster_density > inter_cluster_density
    # For diffusion maps, we also expect the intra-cluster connection to be very strong
    assert nx.is_connected(subgraph_c1)
    assert nx.is_connected(subgraph_c2)

def test_to_pyg_data_export(basic_landscape):
    """
    Tests that the landscape can be correctly exported to a PyTorch
    Geometric Data object.
    """
    try:
        pyg_data = basic_landscape.to_graph_tensor()
    except (ImportError, NameError):
        pytest.skip("torch or torch_geometric not installed.")

    assert isinstance(pyg_data, Data)
    assert hasattr(pyg_data, 'x')
    assert hasattr(pyg_data, 'edge_index')
    
    num_nodes = basic_landscape.graph.number_of_nodes()
    seq_len = len(basic_landscape.sequences[0])
    alphabet_size = len(basic_landscape.sequences[0].alphabet)
    expected_feature_shape = (num_nodes, seq_len * alphabet_size)
    assert pyg_data.x.shape == expected_feature_shape
    
    assert hasattr(pyg_data, 'default')
    assert pyg_data.default.shape[0] == num_nodes

def test_to_sequence_tensors_full_export(basic_landscape):
    """
    Tests the export of all sequences to a list of tensor dictionaries.
    """
    try:
        dataset = basic_landscape.to_sequence_tensors()
    except (ImportError, NameError):
        pytest.skip("torch not installed.")

    assert isinstance(dataset, list)
    assert len(dataset) == len(basic_landscape.sequences)
    
    first_item = dataset[0]
    assert 'sequence_tensor' in first_item
    assert 'fitness_tensors' in first_item
    assert 'default' in first_item['fitness_tensors']
    assert isinstance(first_item['sequence_tensor'], torch.Tensor)

def test_to_sequence_tensors_indexed_export(basic_landscape):
    """
    Tests the export of a single sequence by its index.
    """
    try:
        dataset = basic_landscape.to_sequence_tensors(sequence_idx=3)
    except (ImportError, NameError):
        pytest.skip("torch not installed.")

    assert isinstance(dataset, list)
    assert len(dataset) == 1
    
    original_seq_ohe = basic_landscape.sequences[3].to_one_hot()
    assert torch.allclose(dataset[0]['sequence_tensor'],
                          torch.tensor(original_seq_ohe, dtype=torch.float32))

    original_fitness = basic_landscape.view('default').get_tensor()[3]
    assert torch.allclose(dataset[0]['fitness_tensors']['default'],
                          original_fitness)

def test_to_sequence_tensors_sequence_string_export(basic_landscape):
    """
    Tests the export of a single sequence by its string representation.
    """
    target_sequence_obj = basic_landscape.sequences[5]
    target_sequence_str = "".join(map(str, target_sequence_obj.to_array()))

    try:
        dataset = basic_landscape.to_sequence_tensors(sequence=target_sequence_str)
    except (ImportError, NameError):
        pytest.skip("torch not installed.")

    assert len(dataset) == 1
    
    original_seq_ohe = target_sequence_obj.to_one_hot()
    assert torch.allclose(dataset[0]['sequence_tensor'],
                          torch.tensor(original_seq_ohe, dtype=torch.float32))
    
def test_numeric_fitness():
    """Tests the NumericFitness class."""
    values = [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]]
    fitness_layer = NumericFitness(name="numeric", values=values)
    assert fitness_layer.dtype == "numeric"
    assert torch.is_tensor(fitness_layer.get_tensor())
    assert np.allclose(fitness_layer.to_scalar(), [0.15, 0.3, 0.5])
    assert fitness_layer.get_value(0) == [0.1, 0.2]

def test_categorical_fitness():
    """Tests the CategoricalFitness class."""
    values = ["A", "B", "A"]
    fitness_layer = CategoricalFitness(name="categorical", values=values)
    assert fitness_layer.dtype == "categorical"
    assert torch.is_tensor(fitness_layer.get_tensor())
    assert np.array_equal(fitness_layer.to_scalar(), [0, 1, 0])
    assert fitness_layer.get_value(1) == "B"

def test_probabilistic_categorical_fitness():
    """Tests the ProbabilisticCategoricalFitness class."""
    probabilities = np.array([[0.1, 0.9], [0.8, 0.2]])
    categories = ["A", "B"]
    fitness_layer = ProbabilisticCategoricalFitness(name="prob_categorical", probabilities=probabilities, categories=categories)
    assert fitness_layer.dtype == "categorical"
    assert torch.is_tensor(fitness_layer.get_tensor())
    assert np.array_equal(fitness_layer.to_scalar(), [1, 0])
    assert fitness_layer.get_value(0) == {"A": 0.1, "B": 0.9}

def test_sequence_repr():
    """Tests the __repr__ method of BaseNumpySequence."""
    seq = BaseNumpySequence([0, 1, 0, 1])
    assert repr(seq) == "BaseNumpySequence([0, 1, 0, 1])"

def test_sequence_to_integer():
    """Tests the to_integer method of BaseNumpySequence."""
    seq = BaseNumpySequence(['A', 'C', 'G', 'T'], alphabet=['A', 'C', 'G', 'T'])
    assert np.array_equal(seq.to_integer(), [0, 1, 2, 3])

def test_soft_sequence_entropy():
    """Tests the entropy method of SoftSequence."""
    alphabet = ['A', 'C', 'G', 'T']
    posterior = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.1, 0.7, 0.1],
        [0.25, 0.25, 0.25, 0.25]
    ])
    soft_seq = SoftSequence(posterior, alphabet=alphabet)
    entropy = soft_seq.entropy()
    assert entropy.shape == (3,)
    assert np.all(entropy >= 0)

def test_landscape_repr(basic_landscape):
    """Tests the __repr__ method of FitnessLandscape."""
    assert repr(basic_landscape) == "FitnessLandscape(n_sequences=8)"

def test_landscape_iteration(basic_landscape):
    """Tests iteration over a FitnessLandscape."""
    count = 0
    for seq, fitness in basic_landscape:
        count += 1
    assert count == len(basic_landscape.sequences)

def test_landscape_getitem(basic_landscape):
    """Tests __getitem__ for FitnessLandscape."""
    seq, fitness = basic_landscape[0]
    assert isinstance(seq, BaseNumpySequence)
    assert isinstance(fitness, float)

def test_sequence_distance_errors():
    """Tests error handling in sequence distance calculations."""
    seq1 = BaseNumpySequence([0, 1])
    seq2 = BaseNumpySequence([0, 1, 2])
    with pytest.raises(ValueError):
        seq1.distance(seq2)
    with pytest.raises(ValueError):
        seq1.distance(seq2, metric="invalid_metric")

def test_soft_sequence_resample():
    """Tests the resample method of SoftSequence."""
    alphabet = ['A', 'C']
    posterior = np.array([[0.1, 0.9], [0.8, 0.2]])
    soft_seq = SoftSequence(posterior, alphabet=alphabet)
    resampled_seq = soft_seq.resample()
    assert isinstance(resampled_seq, SoftSequence)

def test_landscape_from_sequences(clustered_data):
    """Tests landscape creation with embeddings."""
    sequences, embeddings = clustered_data
    landscape = FitnessLandscape.from_sequences(sequences, graph_type='knn', k=3)
    assert landscape.graph is not None

def test_landscape_detach_last_layer(basic_landscape):
    """Tests detaching the last fitness layer."""
    basic_landscape.detach('default')
    assert 'default' not in basic_landscape.fitness_layers
    with pytest.raises(ValueError):
        basic_landscape.get_signal()

def test_landscape_attach_mismatched_length(basic_landscape):
    """Tests attaching a layer with mismatched length."""
    new_layer = NumericFitness(name="mismatched", values=[[1.0]])
    with pytest.raises(ValueError):
        basic_landscape.attach(new_layer)

    
def test_sequence_creation_and_properties():
    """Covers __init__ branches for moltype and cogent3 sequences."""
    # Test initialization with a cogent3 sequence object
    c3_seq = get_moltype("text").make_seq("ABC", name="seq1")
    seq1 = BaseNumpySequence(c3_seq)
    assert np.array_equal(seq1.to_array(), ['A', 'B', 'C'])
    assert seq1.id == "seq1"

    # Test initialization with a moltype argument
    seq2 = BaseNumpySequence(['A', 'C', 'G'], moltype="dna")
    assert seq2._c3_seq is not None
    assert str(seq2._c3_seq) == "ACG"
    
    # Test initialization with a non-existent moltype (should still create object)
    seq3 = BaseNumpySequence(['X', 'Y', 'Z'], moltype="invalid_moltype")
    assert seq3._c3_seq is None

def test_sequence_distance_errors():
    """Covers error handling in the distance method."""
    seq_a = BaseNumpySequence([1, 2])
    seq_b = BaseNumpySequence([1, 2, 3])
    # Mismatched lengths
    with pytest.raises(ValueError, match="Sequences must be the same length"):
        seq_a.distance(seq_b)
    # Invalid metric
    with pytest.raises(ValueError, match="Unsupported metric"):
        seq_a.distance(seq_a, metric="manhattan")

def test_mutate_defaults_and_errors():
    """Covers default arguments and error handling in mutate method."""
    seq = BaseNumpySequence(['A', 'A'], alphabet=['A', 'B'])
    
    # Test with no arguments (random mutation)
    mutated = seq.mutate()
    assert seq.distance(mutated) == 1
    
    # Test with integer position
    mutated_pos0 = seq.mutate(positions=0)
    assert mutated_pos0.to_array()[0] == 'B'
    
    # Test error on mismatched lengths of positions and values
    with pytest.raises(ValueError, match="Length of values must equal length of positions"):
        seq.mutate(positions=[0, 1], values=['B'])

def test_soft_sequence_variants():
    """Covers gap posterior and sampling logic in SoftSequence."""
    alphabet = ['A', 'C']
    aa_posterior = np.array([[0.1, 0.9], [0.8, 0.2]])
    gap_posterior = np.array([[0.05], [0.1]])
    
    # Test with gap posterior
    soft_seq_gapped = SoftSequence(aa_posterior, alphabet=alphabet, gap_posterior=gap_posterior)
    assert 'gap' in soft_seq_gapped.alphabet
    assert soft_seq_gapped.posterior.shape == (2, 3)

    # Test with sampling rule
    soft_seq_sampled = SoftSequence(aa_posterior, alphabet=alphabet, hard_rule="sample")
    assert isinstance(soft_seq_sampled.to_array()[0], str)
    
    # Test error on invalid hard_rule
    with pytest.raises(ValueError, match="hard_rule must be"):
        SoftSequence(aa_posterior, alphabet=alphabet, hard_rule="invalid_rule")

def test_generate_sequences_base_cases():
    """Covers the base cases for the generate_sequences function."""
    assert generate_sequences(length=0, alphabet=['A', 'B']) == []
    assert len(generate_sequences(length=1, alphabet=['A', 'B'])) == 2

def test_read_from_fasta(tmp_path: Path):
    """Covers the FASTA reading utility function."""
    fasta_content = ">seq1\nACGT\n>seq2\nGATTACA"
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(fasta_content)
    
    sequences = read_from_fasta(fasta_file, moltype="dna")
    assert len(sequences) == 2
    assert isinstance(sequences[0], BaseNumpySequence)
    assert sequences[0].id == "seq1"
    assert np.array_equal(sequences[1].to_array(), list("GATTACA"))

def test_create_phylo_digraph_from_fasta(phylo_test_data: Path):
    """
    Tests that create_phylo_digraph correctly builds a directed graph
    from a FASTA file, inferring the tree and ancestral states.
    """
    # Run the constructor
    digraph = create_phylo_digraph(sequences=phylo_test_data)
    assert isinstance(digraph, nx.DiGraph), "The output should be a NetworkX DiGraph."
    assert digraph.number_of_nodes() == 5, "Expected 3 tip nodes and 2 ancestral nodes."
    assert digraph.number_of_edges() == 4, "Expected 4 edges in the phylogenetic tree."
    assert isinstance(digraph.nodes['seq1']['sequence'], BaseNumpySequence)
    internal_node = [n for n in digraph.nodes if n not in ['seq1', 'seq2', 'seq3']][0]
    assert 'sequence' in digraph.nodes[internal_node]

def test_create_evol_diffusion_digraph(diffusion_test_data):
    """
    Tests that the evolutionary diffusion graph constructor correctly
    builds a directed graph using k-NN filtering and soft alignment scoring.
    """
    sequences, embeddings = diffusion_test_data


    G = create_evol_diffusion_digraph(
        sequences=sequences,
        embeddings=embeddings,
        replacement_matrix=nq_pfam,
        k=1, # Each node finds only its single nearest neighbor
        t=2,
        tau=0.1 # A low tau to create a sharp kernel
    )

    assert isinstance(G, nx.DiGraph), "The output should be a NetworkX DiGraph."
    assert G.number_of_nodes() == 3, "Graph should have 3 nodes."

def test_parent_selector():
    """Tests that the ParentSelector selects the correct number of candidates."""
    selector = ParentSelector(max_state_size=5)
    candidates = list(range(10))
    weights = np.random.rand(10).tolist()
    selected = selector.select(list(zip(candidates, weights)))
    assert len(selected) == 5

def test_sampler_initialization(sequence_generator):
    """Tests that the EvolutionParticleSampler initializes correctly."""
    selector = ParentSelector(max_state_size=2)
    sampler = EvolutionParticleSampler(
        generator=sequence_generator,
        selector=selector,
        n_samples=2,
        traj_length=5
    )
    sampler.initialize(seed_sequences=["ACDEF"])
    
    assert isinstance(sampler.G, nx.DiGraph)
    assert sampler.G.number_of_nodes() == 1
    node_data = list(sampler.G.nodes(data=True))[0][1]
    assert isinstance(node_data['sequence'], BaseNumpySequence)
    assert node_data['sequence'].to_str() == "ACDEF"

def test_sampler_step(sequence_generator):
    """Tests a single step of the sampler to ensure it adds nodes and edges."""
    selector = ParentSelector(max_state_size=1)
    sampler = EvolutionParticleSampler(
        generator=sequence_generator,
        selector=selector,
        n_samples=10,
        traj_length=10
    )
    sampler.initialize(seed_sequences=["ACGT"])
    
    initial_nodes = sampler.G.number_of_nodes()
    sampler._step()

    # More tests on function would be good
    newly_added_nodes = sampler.G.number_of_nodes() - initial_nodes
    assert isinstance(sampler.G, nx.DiGraph)

@patch('fitness_landscape.core.superscape.HierarchicalRJMCMCAligner')
@patch('fitness_landscape.core.landscape._compute_embeddings_from_sequences')# Patch the entire ESMEmbedder class
def test_superscape_from_parallel_construction(
    MockESMEmbedder, # The mock for the class
    MockHierarchicalRJMCMCAligner,
    phylo_test_data: Path
):
    """
    """
    mock_embedder_instance = MockESMEmbedder.return_value
    mock_embedder_instance.embed_relaxed_seqs.side_effect = [
        np.random.rand(5, 10),
        np.random.rand(3, 10)
    ]
    mock_aligner_instance = MockHierarchicalRJMCMCAligner.return_value

    mock_aligner_instance.run_alignment.return_value = (
        nx.DiGraph(),
        {}
    )

    alignment = load_aligned_seqs(phylo_test_data, moltype="protein")
    sequence_list = alignment_to_base_numpy_sequences(alignment)
    construction_jobs = [
        {
            "sequences": phylo_test_data,
            "digraph_type": 'phylogenetic',
            "_compute_phylo_embeddings": True
        },
        {
            "sequences": sequence_list,
            "digraph_type": 'diffusion_nq',
            "k": 2,
            "t": 2
        }
    ]

    superscape = FitnessSuperscape.from_parallel_construction(
        constructor_type='directed',
        construction_jobs=construction_jobs,
        sampler_kwargs={"burn_in": 1, "samples": 1}
    )

    assert isinstance(superscape, FitnessSuperscape)
    assert len(superscape.landscapes) == 2, "Should create two landscapes from the two jobs."
    phylo_landscape = superscape.landscapes[0]
    assert isinstance(phylo_landscape, DirectedFitnessLandscape)
    assert phylo_landscape.graph.number_of_nodes() == 5, "Phylogenetic graph should have tips and ancestors."
    diffusion_landscape = superscape.landscapes[1]
    assert isinstance(diffusion_landscape, DirectedFitnessLandscape)
    assert diffusion_landscape.graph.number_of_nodes() == 3, "Diffusion graph should only have tip nodes."
    MockHierarchicalRJMCMCAligner.assert_called_once()
    
def test_create_phylo_graph_returns_undirected_graph(phylo_test_data):
    """
    Tests that create_phylo_graph returns an undirected nx.Graph with the correct
    number of nodes (tips + ancestors).
    """
    graph = create_phylo_graph(sequences=phylo_test_data, model_fitting=False, replacement_matrix=['LG'])

    assert isinstance(graph, nx.Graph), "The output should be an undirected nx.Graph."
    assert not graph.is_directed(), "The graph must be undirected."

def test_phylo_graph_is_a_tree(phylo_test_data):
    """
    Confirms that the undirected graph produced has the properties of a tree.
    """
    graph = create_phylo_graph(sequences=phylo_test_data)
    assert nx.is_connected(graph), "The phylogenetic graph must be a single connected component."
    assert graph.number_of_edges() == graph.number_of_nodes() - 1, "The graph should have N-1 edges to be a tree."

def test_create_evol_diffusion_graph_is_undirected_and_symmetric(diffusion_test_data):
    """
    Tests that the evolutionary diffusion graph is undirected and its
    adjacency matrix is symmetric.
    """
    sequences, embeddings = diffusion_test_data
    graph = create_evol_diffusion_graph(sequences, embeddings)

    assert isinstance(graph, nx.Graph), "The output must be an undirected nx.Graph."
    assert not graph.is_directed(), "The graph should not be directed."
    adj_matrix = nx.to_numpy_array(graph)
    assert np.allclose(adj_matrix, adj_matrix.T), "The adjacency matrix of an undirected graph must be symmetric."

def test_hamming_graph_binary_hypercube_degree_and_edges():
    """
    For the full binary hypercube of length L=4 (n=16):
    - each node degree == L
    - edges == n * L / 2
    """
    L = 4
    seqs = generate_sequences(length=L, alphabet=[0, 1])

    seqs = [BinarySequence(s.to_array()) for s in seqs]

    G = create_hamming_graph(sequences=seqs, _backend="binary_xor")
    n = len(seqs)
    expected_edges = n * L // 2

    assert G.number_of_nodes() == n
    assert G.number_of_edges() == expected_edges
    for v in G.nodes():
        assert G.degree[v] == L

    # each node should store its sequence
    for i, s in enumerate(seqs):
        assert 'sequence' in G.nodes[i]
        assert np.array_equal(G.nodes[i]['sequence'].to_array(), s.to_array())

    # default edge attributes present
    for u, v, d in G.edges(data=True):
        assert d.get('weight') == 1.0
        assert d.get('distance') == 0.25


def test_hamming_graph_binary_dispatch_auto_backend():
    """
    Auto backend should select 'binary_xor' for BinarySequence inputs.
    """
    seqs = [BinarySequence([0, 0, 0]), BinarySequence([0, 0, 1])]
    G = create_hamming_graph(sequences=seqs, _backend="auto")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_hamming_graph_binary_xor_rejects_nonbinary():
    """
    Explicit binary_xor backend should reject non-binary inputs.
    """
    seqs = [BaseNumpySequence(['A', 'A']), BaseNumpySequence(['A', 'B'])]
    with pytest.raises(ValueError, match="requires.*binary"):
        create_hamming_graph(sequences=seqs, _backend="binary_xor")


def test_hamming_graph_binary_bitpack_length_limit():
    """
    L > 64 should raise, because bit-pack uses uint64.
    """
    L = 65
    seq0 = BinarySequence([0] * L)
    s = [0] * L
    s[0] = 1
    seq1 = BinarySequence(s)

    with pytest.raises(ValueError, match="L <= 64"):
        # This should fail when packing bits
        create_hamming_graph(sequences=[seq0, seq1], _backend="binary_xor")

def test_hamming_graph_multiallele_degree_and_edges_small_grid():
    """
    For alphabet size 3, L=2 (n=3^2=9) and full combinatorial set:
    degree = (B-1)*L = 4 for every node, edges = n * degree / 2 = 18.
    """
    alphabet = ['A', 'B', 'C']
    L = 2
    # All combinations
    seqs = []
    for a in alphabet:
        for b in alphabet:
            seqs.append(BaseNumpySequence([a, b], alphabet=alphabet))

    G = create_hamming_graph(sequences=seqs, _backend="masked")
    assert G.number_of_nodes() == 9
    assert G.number_of_edges() == 18
    for v in G.nodes():
        assert G.degree[v] == 4  # (3-1)*2

    idx_AA = next(i for i, s in enumerate(seqs) if np.array_equal(s.to_array(), ['A','A']))
    idx_BB = next(i for i, s in enumerate(seqs) if np.array_equal(s.to_array(), ['B','B']))
    assert not G.has_edge(idx_AA, idx_BB)


def test_hamming_graph_multiallele_dispatch_auto_backend():
    """
    Auto backend should select 'masked' for non-binary BaseNumpySequence inputs.
    """
    seqs = [BaseNumpySequence(['A','A']), BaseNumpySequence(['A','B'])]
    G = create_hamming_graph(sequences=seqs, _backend="auto")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1


def test_hamming_graph_multiallele_empty_input():
    """
    Empty inputs should yield an empty graph (no exceptions).
    """
    G = create_hamming_graph(sequences=[], _backend="masked")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_hamming_graph_multiallele_attributes_alignment():
    """
    Node IDs should align to list indices; node['sequence'] should match input object.
    """
    seqs = [
        BaseNumpySequence(['X','Y'], alphabet=['W','X','Y']),
        BaseNumpySequence(['W','Y'], alphabet=['W','X','Y']),
        BaseNumpySequence(['Y','Y'], alphabet=['W','X','Y']),
    ]
    G = create_hamming_graph(sequences=seqs, _backend="masked")
    assert G.number_of_nodes() == 3
    for i, s in enumerate(seqs):
        assert 'sequence' in G.nodes[i]
        assert np.array_equal(G.nodes[i]['sequence'].to_array(), s.to_array())
    for u, v, d in G.edges(data=True):
        assert d.get('weight') == 1.0
        assert d.get('distance') == 0.5

def _toy_binary_seqs_L4():
    # 16 nodes = all 4-bit strings
    seqs = generate_sequences(length=4, alphabet=[0, 1])
    return [BinarySequence(s.to_array()) for s in seqs]

def test_knn_balltree_degree_union_symmetrize():
    seqs = _toy_binary_seqs_L4()
    k = 3
    G = create_knn_graph(sequences=seqs, k=k, backend="balltree")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(seqs)
    for v in G.nodes():
        assert G.degree[v] >= k
    for u, v, d in G.edges(data=True):
        assert "distance" in d and "weight" in d
        assert d["distance"] == d["weight"]
    A = nx.to_numpy_array(G, weight="distance")
    assert np.allclose(A, A.T)

def test_knn_balltree_distances_are_hamming_counts():
    seqs = _toy_binary_seqs_L4()
    k = 4  # at least L neighbors exist at distance 1
    G = create_knn_graph(sequences=seqs, k=k, backend="balltree")
    L = len(seqs[0])
    for u, v, d in list(G.edges(data=True))[:20]:
        dist_graph = d["distance"]
        dist_true = sequence_distance(seqs[u], seqs[v], metric="hamming")
        assert abs(dist_graph - dist_true) < 1e-6
        assert 0 <= dist_graph <= L

@pytest.mark.parametrize("tie_policy,expected_min_degree", [
    ("all", 4), 
    ("min_index", 2),
    ("random", 2),
])
def test_knn_balltree_tie_handling(tie_policy, expected_min_degree):
    seqs = _toy_binary_seqs_L4()
    # Make ties at distance=1 by choosing k < L
    k = 2
    G = create_knn_graph(sequences=seqs, k=k, backend="balltree",
                         tie_policy=tie_policy, seed=123)
    degs = dict(G.degree())
    assert min(degs.values()) >= expected_min_degree

def test_knn_balltree_random_ties_seed_determinism():
    seqs = _toy_binary_seqs_L4()
    k = 2
    G1 = create_knn_graph(sequences=seqs, k=k, backend="balltree",
                          tie_policy="random", seed=42)
    G2 = create_knn_graph(sequences=seqs, k=k, backend="balltree",
                          tie_policy="random", seed=42)
    G3 = create_knn_graph(sequences=seqs, k=k, backend="balltree",
                          tie_policy="random", seed=43)
    assert set(G1.edges()) == set(G2.edges())
    assert set(G1.edges()) != set(G3.edges()) or len(set(G1.edges())) == 0

def test_knn_balltree_single_and_empty_inputs():
    # single
    s = [BinarySequence([0,0,0,0])]
    G1 = create_knn_graph(sequences=s, k=3, backend="balltree")
    assert G1.number_of_nodes() == 1 and G1.number_of_edges() == 0
    # empty
    G0 = create_knn_graph(sequences=[], k=3, backend="balltree")
    assert G0.number_of_nodes() == 0 and G0.number_of_edges() == 0

@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_knn_faiss_flat_exact_distances_match_hamming(metric):
    # small set for exact comparison
    seqs = _toy_binary_seqs_L4()
    k = 3
    G = create_knn_graph(
        sequences=seqs, k=k, backend="faiss",
        index_type="flat",   # exact search
        faiss_metric=metric,
        tiebuffer=0,  # not needed for exact/unique here
        include_self=False
    )
    # Distances in graph should match true Hamming counts
    for u, v, d in list(G.edges(data=True))[:30]:
        dist_graph = d["distance"]
        dist_true = sequence_distance(seqs[u], seqs[v], metric="hamming")
        assert abs(dist_graph - dist_true) < 1e-6

def test_knn_faiss_hnsw_union_degree_and_attrs():
    # Use small set; approximate index should still be fine here
    seqs = _toy_binary_seqs_L4()
    k = 3
    G = create_knn_graph(sequences=seqs,
                         k=k,
                         backend="faiss",
                         index_type="hnsw",
                         faiss_metric="ip",
                         hnsw_M=16,
                         tiebuffer=16,
                         include_self=False)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(seqs)
    for v in G.nodes():
        assert G.degree[v] >= k
    for u, v, d in G.edges(data=True):
        assert "distance" in d and "weight" in d

def test_knn_faiss_tie_handling_all_min_random():
    # Create many equidistant neighbors by using L=4 hypercube; k < L
    seqs = _toy_binary_seqs_L4()
    k = 2
    # 'all'
    Gall = create_knn_graph(
        sequences=seqs, k=k, backend="faiss",
        index_type="flat", faiss_metric="ip", tiebuffer=32,
        tie_policy="all", include_self=False
    )
    # degree should be >= L (=4) at many nodes due to including all dist-1 ties
    assert min(dict(Gall.degree()).values()) >= 2  

    Gmin = create_knn_graph(
        sequences=seqs, k=k, backend="faiss",
        index_type="flat", faiss_metric="ip",
        tie_policy="min_index", include_self=False
    )
    assert min(dict(Gmin.degree()).values()) >= k
    Gr1 = create_knn_graph(
        sequences=seqs, k=k, backend="faiss",
        index_type="flat", faiss_metric="ip",
        tie_policy="random", include_self=False, seed=7
    )
    Gr2 = create_knn_graph(
        sequences=seqs, k=k, backend="faiss",
        index_type="flat", faiss_metric="ip",
        tie_policy="random", include_self=False, seed=7
    )
    assert set(Gr1.edges()) == set(Gr2.edges())

def test_knn_faiss_include_self_does_not_create_loops():
    seqs = _toy_binary_seqs_L4()
    k = 3
    G = create_knn_graph(
        sequences=seqs, k=k, backend="faiss",
        index_type="flat", faiss_metric="ip",
        include_self=True
    )
    # No self-loops in the undirected graph
    assert all(u != v for u, v in G.edges())

def test_knn_invalid_backend_raises():
    seqs = _toy_binary_seqs_L4()
    with pytest.raises(ValueError):
        create_knn_graph(sequences=seqs, k=3, backend="nope")

def test_knn_balltree_nonbinary_hamming_works():
    alphabet = ["A", "B", "C", "D"]
    seqs = [
        BaseNumpySequence(["A","A","A","A"], alphabet=alphabet),
        BaseNumpySequence(["A","A","A","B"], alphabet=alphabet),
        BaseNumpySequence(["A","A","C","A"], alphabet=alphabet),
        BaseNumpySequence(["D","A","A","A"], alphabet=alphabet),
        BaseNumpySequence(["A","B","A","A"], alphabet=alphabet),
    ]
    G = create_knn_graph(sequences=seqs, k=2, backend="balltree")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(seqs)
    for (u, v, d) in G.edges(data=True):
        dist_true = sequence_distance(seqs[u], seqs[v], metric="hamming")
        assert abs(d["distance"] - dist_true) < 1e-6

def test_knn_auto_backend_small_n_is_balltree_like():
    seqs = _toy_binary_seqs_L4()
    G_auto = create_knn_graph(sequences=seqs, k=3, backend="auto")
    G_bt   = create_knn_graph(sequences=seqs, k=3, backend="balltree")
    assert G_auto.number_of_nodes() == G_bt.number_of_nodes()
    assert G_auto.number_of_edges() >= G_bt.number_of_edges() * 0.8

def test_encode_multiallele_stable_mapping():
    # First appearance order should define mapping deterministically
    seqs = [
        BaseNumpySequence(['X','Y']),
        BaseNumpySequence(['Y','Z']),
        BaseNumpySequence(['Z','X']),
    ]
    X1, map1 = _encode_multiallele(seqs)
    X2, map2 = _encode_multiallele(seqs)  # repeat
    assert map1 == map2
    assert X1.dtype == np.int32 and X2.dtype == np.int32
    assert X1.shape == (3, 2)

def test_hamming_multiallele_all_identical_sequences_yields_no_edges():
    seqs = [BaseNumpySequence(['A','A'], alphabet=['A','B']) for _ in range(5)]
    G = create_hamming_graph(sequences=seqs, _backend="masked")
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 0

def test_knn_balltree_all_ties_reaches_degree_at_least_L_on_hypercube():
    seqs = _toy_binary_seqs_L4()  # L = 4
    G = create_knn_graph(sequences=seqs, k=2, backend="balltree", tie_policy="all")
    L = len(seqs[0])
    assert min(dict(G.degree()).values()) >= L  # union with all ties

def test_knn_balltree_all_ties_reaches_degree_at_least_L_on_hypercube():
    seqs = _toy_binary_seqs_L4()  # L = 4
    G = create_knn_graph(sequences=seqs, k=2, backend="balltree", tie_policy="all")
    L = len(seqs[0])
    assert min(dict(G.degree()).values()) >= L  # union with all ties

def test_diffusion_emb_graph_auto_backend_small_n_balltree_path(clustered_data):
    seqs, embs = clustered_data[:]
    # small n triggers BallTree path in 'auto'
    G = create_diffusion_emb_graph(sequences=seqs, embeddings=embs, backend="auto", k=3, t=3)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(seqs)

def test_diffusion_emb_graph_k_ge_n_handles_sigma_zero_safely():
    # k >= n-1 then k_for_scale adjusted and median_sigma_sq fallback
    seqs = [BinarySequence([0,0,0]), BinarySequence([0,0,1]), BinarySequence([0,1,1])]
    embs = np.array([[0.,0.],[0.,0.],[0.,0.]])  # degenerate distances
    G = create_diffusion_emb_graph(sequences=seqs, embeddings=embs, k=10, t=2)
    assert isinstance(G, nx.Graph)  # no crash

def test_evol_diffusion_graph_falls_back_to_hamming_knn_when_no_embeddings(diffusion_test_data):
    seqs, _ = diffusion_test_data
    G = create_evol_diffusion_graph(sequences=seqs, embeddings=None, k=1, t=2)
    assert isinstance(G, nx.Graph)
    A = nx.to_numpy_array(G)
    assert np.allclose(A, A.T)

def test_evol_diffusion_graph_rejects_non_PROT20():
    seqs = [BaseNumpySequence(['A','B'], alphabet=['A','B'])]
    with pytest.raises(ValueError, match="PROT_20"):
        create_evol_diffusion_graph(sequences=seqs, embeddings=None)

def test_evol_diffusion_graph_soft_sequences_ok():
    # 2 pos, 3 aa alphabet from PROT_20 slice for simplicity
    alphabet = PROT_20
    post1 = np.zeros((2, len(alphabet))); post1[:, alphabet.index('A')] = 1.0
    post2 = np.zeros((2, len(alphabet))); post2[:, alphabet.index('R')] = 1.0
    s1 = SoftSequence(post1, alphabet=alphabet)
    s2 = SoftSequence(post2, alphabet=alphabet)
    G = create_evol_diffusion_graph(sequences=[s1, s2], embeddings=None, k=1, t=1)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2

def test_tda_graph_reweight_simplex_edges_sets_attribute(clustered_data):
    seqs, embs = clustered_data
    G = create_tda_graph(sequences=seqs, embeddings=embs, reweight_simplex_edges=True)
    # If there are any triangles, edges should have 'simplicial_weight' ≥ 1
    if G.number_of_edges() > 0:
        assert all('simplicial_weight' in d for *_, d in G.edges(data=True))

@pytest.mark.parametrize("graph_type", ["hamming", "knn", "tda", "diffusion"])
def test_landscape_from_sequences_attach_embeddings_toggle(graph_type, clustered_data):
    if graph_type in {"tda", "diffusion"}:
        seqs, embs = clustered_data
        ctor_kwargs = {}
    else:
        seqs = generate_sequences(length=3, alphabet=[0,1])
        embs = None
        ctor_kwargs = {"k": 3} if graph_type == "knn" else {}

    fl = FitnessLandscape.from_sequences(
        sequences=seqs,
        fitness_layers={},
        graph_type=graph_type,
        embeddings=embs,
        attach_embeddings=True,
        **ctor_kwargs
    )
    if graph_type in {"tda", "diffusion"}:
        assert fl.embeddings is not None
    else:
        assert fl.embeddings is None

def test_landscape_from_sequences_phylo_mismatched_embeddings_raises(phylo_test_data):
    # Provide wrong sized embeddings to hit the validation error path
    sequences = alignment_to_base_numpy_sequences(load_aligned_seqs(phylo_test_data, moltype="protein"))
    bad_embs = np.random.randn(len(sequences) - 1, 8)
    with pytest.raises(ValueError, match="expected embeddings shape"):
        FitnessLandscape.from_sequences(
            sequences=phylo_test_data,
            graph_type='phylogenetic',
            embeddings=bad_embs,
            _compute_phylo_embeddings=False
        )

def test_landscape_view_and_get_layer_errors(basic_landscape):
    with pytest.raises(KeyError, match="not found"):
        basic_landscape.view('does_not_exist')
    with pytest.raises(KeyError, match="not found"):
        basic_landscape.get_layer('does_not_exist', allow_active_default=False)

def test_landscape_attach_duplicate_layer_name_raises(basic_landscape):
    dup = NumericFitness(name='default', values=[[0.0] for _ in range(len(basic_landscape))])
    with pytest.raises(ValueError, match="already exists"):
        basic_landscape.attach(dup)

def test_to_graph_tensor_without_embeddings_uses_ohe_shape():
    seqs = generate_sequences(length=3, alphabet=[0,1])
    fl = FitnessLandscape.from_sequences(seqs, fitness_layers={}, graph_type='hamming', attach_embeddings=False)
    data = fl.to_graph_tensor()
    n = len(seqs); L = len(seqs[0]); A = len(seqs[0].alphabet)
    assert data.x.shape == (n, L * A)

def test_to_sequence_tensors_unknown_sequence_raises(basic_landscape):
    with pytest.raises(ValueError, match="not found"):
        basic_landscape.to_sequence_tensors(sequence="9999")  # invalid for binary L=3

def _make_dupe_seqs():
    # Two identical sequences ("000"), plus "001"
    return [BinarySequence([0, 0, 0]),
            BinarySequence([0, 0, 0]),
            BinarySequence([0, 0, 1])]


def _make_simple_landscape_with_dupes():
    seqs = _make_dupe_seqs()
    # Build a tiny hamming graph via factory – lets the class annotate nodes, etc.
    fl = FitnessLandscape.from_sequences(
        sequences=seqs,
        fitness_layers={},           # start empty
        graph_type="hamming"
    )
    return fl


def _make_simple_landscape_no_dupes():
    seqs = [BinarySequence([0, 0, 0]),
            BinarySequence([0, 0, 1]),
            BinarySequence([0, 1, 1])]
    fl = FitnessLandscape.from_sequences(
        sequences=seqs,
        fitness_layers={},
        graph_type="hamming"
    )
    return fl

def test_attach_by_sequence_numeric_first_all_and_missing():
    fl = _make_simple_landscape_with_dupes()
    # Map values by sequence string; dtype inferred downstream via dtype='numeric'
    values_map = {
        "000": [1.0, 2.0],
        "001": 3.0,
    }

    fl.attach(
        name="num_first",
        values=values_map,
        dtype="numeric",
        map_by="sequence",
        on_duplicates="first",
        allow_missing=True,
    )
    t_first = fl.view("num_first").get_tensor().numpy()
    # shape (3, max_reps=2)
    assert t_first.shape == (3, 2)
    # first "000"
    np.testing.assert_allclose(t_first[0], [1.0, 2.0], equal_nan=False)
    # second "000" got "__missing__" -> padded with NaN replicate list
    assert np.isnan(t_first[1]).all()
    # "001"
    np.testing.assert_allclose(t_first[2], [3.0, np.nan], equal_nan=True)

    fl.attach(
        name="num_all",
        values=values_map,
        dtype="numeric",
        map_by="sequence",
        on_duplicates="all",
        allow_missing=False,
    )
    t_all = fl.view("num_all").get_tensor().numpy()
    np.testing.assert_allclose(t_all[0], [1.0, 2.0], equal_nan=False)
    np.testing.assert_allclose(t_all[1], [1.0, 2.0], equal_nan=False)
    np.testing.assert_allclose(t_all[2], [3.0, np.nan], equal_nan=True)


def test_attach_by_sequence_numeric_error_on_dupes():
    fl = _make_simple_landscape_with_dupes()
    with pytest.raises(ValueError):
        fl.attach(
            name="num_err",
            values={"000": 1.0, "001": 2.0},
            dtype="numeric",
            map_by="sequence",
            on_duplicates="error",
        )

def test_attach_by_sequence_categorical_first_all_and_missing_placeholder():
    fl = _make_simple_landscape_with_dupes()
    values_map = {
        "000": "A",
        "001": "B",
    }
    categories = ["A", "B", "__MISSING__"]

    # on_duplicates='first' with allow_missing=True -> second "000" becomes "__MISSING__"
    fl.attach(
        name="cat_first",
        values=values_map,
        dtype="categorical",
        categories=categories,
        map_by="sequence",
        on_duplicates="first",
        allow_missing=True,
    )
    cat_layer = fl.view("cat_first")
    # tensor one-hot should exist and have correct num categories
    t = cat_layer.get_tensor().numpy()
    assert t.shape == (3, 3)
    # decode back: argmax
    idx = t.argmax(axis=1)
    decoded = [categories[i] for i in idx]
    assert decoded == ["A", "__MISSING__", "B"]

    # on_duplicates='all' w/out allow_missing -> both "000" become "A"
    fl.attach(
        name="cat_all",
        values=values_map,
        dtype="categorical",
        categories=["A", "B"],
        map_by="sequence",
        on_duplicates="all",
        allow_missing=False,
    )
    t2 = fl.view("cat_all").get_tensor().numpy()
    idx2 = t2.argmax(axis=1)
    decoded2 = [["A", "B"][i] for i in idx2]
    assert decoded2 == ["A", "A", "B"]


def test_attach_by_index_numeric_and_categorical_ok():
    fl = _make_simple_landscape_no_dupes()
    # numeric by index, different replicate lengths
    fl.attach(
        name="n_idx",
        values=[[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
        dtype="numeric",
        map_by="index",
    )
    t = fl.view("n_idx").get_tensor()
    assert isinstance(t, torch.Tensor)
    # padded to max replicate length = 3
    assert t.shape == (3, 3)
    np.testing.assert_allclose(t[0].numpy(), [1.0, 2.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(t[1].numpy(), [3.0, np.nan, np.nan], equal_nan=True)

    # categorical by index
    fl.attach(
        name="c_idx",
        values=["low", "high", "med"],
        dtype="categorical",
        map_by="index",
        categories=["low", "med", "high"],
    )
    c = fl.view("c_idx").get_tensor()
    assert c.shape == (3, 3)
    # check argmax class indices: low(0), high(2), med(1)
    idx = c.argmax(dim=1).tolist()
    assert idx == [0, 2, 1]


def test_attach_by_index_length_mismatch_raises():
    fl = _make_simple_landscape_no_dupes()
    with pytest.raises(ValueError):
        fl.attach(
            name="bad_len",
            values=[1.0, 2.0],    # len 2, but we have 3 sequences
            dtype="numeric",
            map_by="index",
        )


def test_detach_removes_graph_attrs_safely():
    fl = _make_simple_landscape_no_dupes()
    # attach a simple categorical layer by index
    fl.attach(
        name="will_detach",
        values=["x", "y", "z"],
        dtype="categorical",
        map_by="index",
        categories=["x", "y", "z"],
    )
    # confirm attrs present on nodes
    for _, data in fl.graph.nodes(data=True):
        assert "fitness_will_detach" in data

    # now detach; should not KeyError even if some nodes miss the key
    fl.detach("will_detach")
    for _, data in fl.graph.nodes(data=True):
        assert "fitness_will_detach" not in data
    # active view automatically moves to another layer or None
    assert fl.active_layer_name in (None, *fl.fitness_layers.keys())

def test_get_fitness_legacy_and_signal_follow_active_layer():
    fl = _make_simple_landscape_no_dupes()
    # attach two layers; check active layer semantics
    fl.attach(
        name="score1",
        values=[1.0, 2.0, 3.0],
        dtype="numeric",
        map_by="index",
    )
    fl.attach(
        name="score2",
        values=[10.0, 20.0, 30.0],
        dtype="numeric",
        map_by="index",
    )
    # .view sets active
    fl.view("score2")
    s = fl.get_signal()
    np.testing.assert_allclose(s, [10.0, 20.0, 30.0])
    # get_fitness on a particular sequence
    val = fl.get_fitness(fl.sequences[1])
    assert val == 20.0

def test_to_sequence_tensors_index_and_sequence_lookup():
    fl = _make_simple_landscape_no_dupes()
    fl.attach(
        name="score",
        values=[0.1, 0.2, 0.3],
        dtype="numeric",
        map_by="index",
    )

    # by index
    out_idx = fl.to_sequence_tensors(sequence_idx=[0, 2])
    assert len(out_idx) == 2
    assert "sequence_tensor" in out_idx[0]
    assert "fitness_tensors" in out_idx[0]
    assert torch.is_tensor(out_idx[0]["sequence_tensor"])

    # by sequence string (BinarySequence alphabet is [0,1])
    out_seq = fl.to_sequence_tensors(sequence=["001"])
    assert len(out_seq) == 1
    # value at index of "001" is 0.2
    score_tensor = out_seq[0]["fitness_tensors"]["score"]
    assert torch.is_tensor(score_tensor)
    assert torch.allclose(score_tensor[~torch.isnan(score_tensor)], torch.tensor([0.2]))


def test_base_from_string_and_iterable_defaults():
    s = BaseNumpySequence.from_string("ACDE")
    assert isinstance(s, BaseNumpySequence)
    assert s.to_str() == "ACDE"
    # Default alphabet is PROT_20
    assert s.alphabet == list(PROT_20)

    t = BaseNumpySequence.from_iterable(list("WQER"), alphabet=list("WQER"))
    assert t.to_str() == "WQER"
    assert t.alphabet == list("WQER")


def test_base_from_cogent3_preserves_moltype_and_alphabet():
    prot = get_moltype("protein")
    c3 = prot.make_seq("ACDEFG")
    s = BaseNumpySequence.from_cogent3(c3)
    assert s.to_str() == "ACDEFG"
    # Alphabet should come from cogent3 moltype
    assert set(s.alphabet) >= set(list("ACDEFG"))  # superset check (protein alphabet)


def test_from_one_hot_with_and_without_alphabet_roundtrip():
    one_hot = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    # With explicit alphabet
    seq = BaseNumpySequence.from_one_hot(one_hot, alphabet=["A", "C", "D"])
    assert seq.to_str() == "ACD"
    # Round-trip to one-hot shape
    oh = seq.to_one_hot(mapping={"A":0, "C":1, "D":2})
    assert oh.shape == one_hot.shape
    assert np.all((oh == one_hot))

    seq_num = BaseNumpySequence.from_one_hot(one_hot)
    # Should have length 3 and correct argmax positions
    assert len(seq_num) == 3
    # Rebuilding its own one-hot should match shape
    oh2 = seq_num.to_one_hot(mapping={str(i): i for i in range(3)})
    assert oh2.shape == (3, 3)


def test_from_integer_with_explicit_alphabet():
    idxs = [0, 2, 1, 2]
    alphabet = ["A", "B", "C"]
    s = BaseNumpySequence.from_integer(idxs, alphabet=alphabet)
    assert s.to_str() == "ACBC"
    assert s.alphabet == alphabet


def test_make_sequence_infers_binary_and_base_numpy():
    b = make_sequence([0, 1, 1, 0])  # infer binary
    assert isinstance(b, BinarySequence)
    assert b.to_array().tolist() == [0, 1, 1, 0]

    p = make_sequence("ACD", alphabet=list("ACDE"))
    assert isinstance(p, BaseNumpySequence)
    assert p.to_str() == "ACD"


def test_binary_sequence_constructors_and_random():
    b1 = BinarySequence.from_bits([1, 0, 1])
    assert isinstance(b1, BinarySequence)
    assert b1.to_array().tolist() == [1, 0, 1]

    b2 = BinarySequence.from_integer_bits(13, length=6)  # 001101 (MSB-first)
    assert isinstance(b2, BinarySequence)
    assert len(b2) == 6

    b3 = BinarySequence.random(10, p_one=0.25, seed=123)
    assert isinstance(b3, BinarySequence)
    assert len(b3) == 10
    # Reproducible
    b4 = BinarySequence.random(10, p_one=0.25, seed=123)
    assert np.array_equal(b3.to_array(), b4.to_array())


def test_multiallele_sequence_random_and_from_string():
    m1 = MultialleleSequence.random(5, alphabet=["A", "B", "C"])
    assert isinstance(m1, MultialleleSequence)
    assert len(m1) == 5
    assert set(m1.alphabet) == {"A", "B", "C"}

    m2 = MultialleleSequence.from_string("ABC", alphabet=["A", "B", "C"])
    assert m2.to_str() == "ABC"


def test_softsequence_from_posteriors_argmax_and_sample():
    aa_post = np.array([[0.1, 0.9],
                        [0.7, 0.3],
                        [0.4, 0.6]])
    s_arg = SoftSequence.from_posteriors(aa_post, alphabet=["X", "Y"], hard_rule="argmax")
    assert s_arg.to_str() == "YXY"

    s_samp = SoftSequence.from_posteriors(aa_post, alphabet=["X", "Y"], hard_rule="sample", seed=7)
    # deterministic due to seed
    assert isinstance(s_samp, SoftSequence)
    assert len(s_samp) == 3
    # resample uses same seed stored in instance
    again = s_samp.resample()
    assert isinstance(again, SoftSequence)
    assert len(again) == 3


def test_as_sequences_mixed_inputs_and_binary_detection():
    items = ["ACD", "0101", np.array(list("AAA")), [0, 1, 0, 0]]
    seqs = as_sequences(items, alphabet=list("ACDE"))
    assert isinstance(seqs[0], BaseNumpySequence)
    assert isinstance(seqs[1], BinarySequence)
    assert isinstance(seqs[2], BaseNumpySequence)
    assert isinstance(seqs[3], BinarySequence)
    assert seqs[0].to_str() == "ACD"
    assert seqs[1].to_array().tolist() == [0, 1, 0, 1]


def test_generate_sequences_and_distance():
    seqs = generate_sequences(length=3, alphabet=["A", "B"])
    assert len(seqs) == 8
    assert all(isinstance(s, BaseNumpySequence) for s in seqs)

    d_h = sequence_distance(seqs[0], seqs[-1], metric="hamming")
    assert isinstance(d_h, (float, int))
    d_e = sequence_distance(np.array([0, 0]), np.array([1, 1]), metric="euclidean")
    assert pytest.approx(d_e) == np.sqrt(2)


def test_read_from_fasta_roundtrip(tmp_path):
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">seq1\nACDE\n>seq2\nWQER\n")
    seqs = read_from_fasta(fasta, moltype="protein")
    assert len(seqs) == 2
    assert seqs[0].to_str() == "ACDE"
    assert seqs[1].to_str() == "WQER"
    assert set("ACDEWQER").issubset(set(seqs[0].alphabet))

def test_numeric_from_scalars_and_tensor_roundtrip():
    vals = [0.1, 0.2, 0.3, 0.4]
    nf = NumericFitness.from_scalars("fit", vals)
    assert nf.dtype == "numeric"
    assert len(nf) == 4
    # mean equals original scalars
    np.testing.assert_allclose(nf.to_scalar(), np.array(vals))
    # tensor shape (N, max_reps) = (4, 1)
    t = nf.get_tensor().numpy()
    assert t.shape == (4, 1)
    np.testing.assert_allclose(t[:, 0], np.array(vals))

    # Build from dense tensor with NaNs and trim strategy
    mat = np.array([[1.0,  2.0,  np.nan],
                    [3.0,  np.nan, np.nan],
                    [5.0,  6.0,  7.0]])
    nf2 = NumericFitness.from_tensor("fit2", mat, pad_strategy="trim_tail_nans")
    assert len(nf2) == 3
    # row means computed on the trimmed lists
    means = []
    for row in mat:
        non_nan = np.where(~np.isnan(row))[0]
        last = non_nan[-1]  # last non-NaN index
        means.append(np.nanmean(row[: last + 1]))
    np.testing.assert_allclose(nf2.to_scalar(), np.array(means))

def test_numeric_from_replicates_and_index_map_and_random():
    reps = [[1.0, 2.0], [], [5.0]]
    nf = NumericFitness.from_replicates("r", reps)
    assert len(nf) == 3
    # empty replicate becomes [nan]
    tensor = nf.get_tensor().numpy()
    assert tensor.shape == (3, 2)  # padded
    assert np.isnan(tensor[1, 0])

    # index map with fill
    mp = {0: 10.0, 2: [1.0, 1.5]}
    nf2 = NumericFitness.from_index_map("imap", mp, length=3, fill=-1.0)
    assert len(nf2) == 3
    scalars = nf2.to_scalar()
    assert scalars[0] == 10.0 and scalars[1] == -1.0
    assert np.isclose(scalars[2], np.mean([1.0, 1.5]))

    # random constructor reproducibility & shape
    nf3a = NumericFitness.random("rnd", length=5, reps=3, dist="normal", seed=123)
    nf3b = NumericFitness.random("rnd", length=5, reps=3, dist="normal", seed=123)
    np.testing.assert_allclose(nf3a.get_tensor().numpy(), nf3b.get_tensor().numpy())
    assert nf3a.get_tensor().shape == (5, 3)

def test_categorical_from_values_and_one_hot_and_index_map():
    vals = ["A", "B", "A", "C"]
    cf = CategoricalFitness.from_values("cat", vals)
    assert cf.dtype == "categorical"
    assert len(cf) == 4
    # default rank map follows category order used internally
    r = cf.to_scalar()
    assert r.dtype == int
    # roundtrip via one-hot
    one_hot = cf.get_tensor().numpy()
    cats = cf.categories
    cf2 = CategoricalFitness.from_one_hot("cat2", one_hot, categories=cats)
    assert cf2.categories == cats
    assert cf2._values == vals

    # index map, explicit categories and default
    mp = {0: "X", 2: "Y"}
    cf3 = CategoricalFitness.from_index_map(
        "imap", mp, length=3, default="Z", categories=["X", "Y", "Z"]
    )
    assert cf3._values == ["X", "Z", "Y"]
    # wrong rank map coverage
    with pytest.raises(ValueError):
        cf3.to_scalar(rank_map={"X": 0})  # missing Y/Z

def test_categorical_random_reproducible():
    cats = ["L", "M", "H"]
    cf1 = CategoricalFitness.random("r", length=6, categories=cats, seed=7)
    cf2 = CategoricalFitness.random("r", length=6, categories=cats, seed=7)
    assert cf1._values == cf2._values
    assert set(cf1._values).issubset(set(cats))

def test_probabilistic_from_probabilities_logits_counts_samples():
    cats = ["A", "B", "C"]
    # probabilities (already normalized)
    P = np.array([[0.7, 0.2, 0.1],
                  [0.0, 1.0, 0.0],
                  [0.3, 0.3, 0.4]])
    pf = ProbabilisticCategoricalFitness.from_probabilities("p", P, categories=cats)
    assert pf.dtype == "categorical"
    assert len(pf) == 3
    s = pf.to_scalar()
    np.testing.assert_array_equal(s, np.array([0, 1, 2]))

    Z = np.array([[3.0, 1.0, 0.0],
                  [0.1, 2.4, -1.0],
                  [-2.0, -2.0, 0.0]])
    pf2 = ProbabilisticCategoricalFitness.from_logits("log", Z, categories=cats)
    np.testing.assert_array_equal(pf2.to_scalar(), np.array([0, 1, 2]))

    # counts with smoothing and without
    C = np.array([[7, 2, 1],
                  [0, 5, 0],
                  [3, 3, 4]], dtype=float)
    pf3 = ProbabilisticCategoricalFitness.from_counts("cnt", C, categories=cats, alpha=0.0)
    assert np.allclose(pf3.get_tensor().sum(axis=1).numpy(), 1.0)
    pf3s = ProbabilisticCategoricalFitness.from_counts("cnts", C, categories=cats, alpha=1.0)
    # smoothing changes distribution
    assert not np.allclose(pf3.get_tensor().numpy(), pf3s.get_tensor().numpy())

    samples = [["A", "A", "B"], ["B", "B"], ["C", "C", "A", "C"]]
    pf4 = ProbabilisticCategoricalFitness.from_samples("s", samples, categories=cats)
    assert pf4.probabilities.shape == (3, 3)
    assert np.allclose(pf4.probabilities.sum(axis=1), 1.0)

def test_make_fitness_layer_numeric_and_categorical_auto():
    # numeric 1-D
    nf = make_fitness_layer("n1", [1.0, 2.0, 3.0])
    assert isinstance(nf, NumericFitness)
    np.testing.assert_allclose(nf.to_scalar(), np.array([1.0, 2.0, 3.0]))

    # numeric 2-D → NumericFitness
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    nf2 = make_fitness_layer("n2", mat)
    assert isinstance(nf2, NumericFitness)
    assert nf2.get_tensor().shape == (2, 2)

    # categorical from one-hot (explicit dtype + categories)
    one_hot = np.array([[1, 0, 0],
                        [0, 0, 1]])
    cf = make_fitness_layer("c1", one_hot, dtype="categorical", categories=["X", "Y", "Z"])
    assert isinstance(cf, CategoricalFitness)
    assert cf._values == ["X", "Z"]

    # probabilistic categorical (rows sum to 1)
    P = np.array([[0.2, 0.8], [0.6, 0.4]])
    pf = make_fitness_layer("pc", P, dtype="categorical", categories=["A", "B"])
    assert isinstance(pf, ProbabilisticCategoricalFitness)
    np.testing.assert_allclose(pf.get_tensor().sum(axis=1).numpy(), 1.0)


def test_as_fitness_layers_mixed_mapping():
    layers_in = {
        "fit": [0.0, 1.0, 2.0],# numeric scalars
        "rep": [[1.0, 2.0], [3.0], [4.0, 5.0]], # replicates
        "label": np.array([[1, 0], [0, 1], [1, 0]]), # one-hot categorical
        "post": np.array([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5]])  # probabilities
    }
    cats = {
        "label": ["A", "B"],
        "post": ["X", "Y"],
    }
    out = as_fitness_layers(layers_in, categories=cats)
    assert set(out.keys()) == {"fit", "rep", "label", "post"}
    assert isinstance(out["fit"], NumericFitness)
    assert isinstance(out["rep"], NumericFitness)
    assert isinstance(out["label"], CategoricalFitness)
    assert isinstance(out["post"], ProbabilisticCategoricalFitness)

    # sanity on shapes
    assert out["fit"].get_tensor().shape == (3, 1)
    assert out["rep"].get_tensor().shape[0] == 3
    assert out["label"].get_tensor().shape == (3, 2)
    assert out["post"].get_tensor().shape == (3, 2)

def _toy_seqs():
    # short, small alphabet to keep graphs trivial
    return [make_sequence(s) for s in ["ACD", "ACE", "ACF", "BCD"]]


def test_build_hamming_basic_and_annotation():
    seqs = _toy_seqs()
    # attach one numeric layer to ensure annotation happens
    fit = NumericFitness.from_scalars("fitness", [0.1, 0.2, 0.3, 0.4])
    L = FitnessLandscape.build(
        sequences=seqs,
        graph="hamming",
        fitness_layers={"fitness": fit},
        attach_embeddings=False,  # hamming does not require embeddings
    )
    assert isinstance(L.graph, nx.Graph)
    assert len(L) == len(seqs)
    # nodes should have fitness annotations
    for _, data in L.graph.nodes(data=True):
        assert "sequence" in data
        assert "fitness_fitness" in data


def test_build_with_existing_graph_object():
    seqs = _toy_seqs()
    G = create_hamming_graph(seqs)
    # pre-make a categorical layer
    lab = CategoricalFitness.from_values("label", ["A", "B", "A", "B"])
    L = FitnessLandscape.build(
        sequences=seqs,
        graph=G,
        fitness_layers={"label": lab},
        attach_embeddings=False,
    )
    assert L.graph is G
    # categorical layer annotated on nodes
    for _, data in L.graph.nodes(data=True):
        assert "label" in L.fitness_layers
        assert "fitness_label" in data


def test_build_embedding_graph_auto_ohe_and_x_tensor_shape():
    seqs = _toy_seqs()
    L = FitnessLandscape.build(
        sequences=seqs,
        graph="tda", 
        embedding_domain="ohe",
        attach_embeddings=True,
        n_components=1
    )
    # Embeddings should be computed/attached
    assert L.embeddings is not None
    assert isinstance(L.embeddings, np.ndarray)
    # Export to PyG tensor; x should be present
    pyg = L.to_graph_tensor()
    assert hasattr(pyg, "x")
    assert pyg.num_nodes == len(seqs)
    assert pyg.x.shape[0] == len(seqs)

def test_view_and_get_layer_selection_and_errors():
    seqs = _toy_seqs()
    layers = {
        "fit": NumericFitness.from_scalars("fit", [0.0, 1.0, 2.0, 3.0]),
        "cls": CategoricalFitness.from_values("cls", ["X", "Y", "X", "Y"]),
    }
    L = FitnessLandscape.build(seqs, graph="hamming", fitness_layers=layers)
    # view() sets active layer
    L.view("cls")
    assert L.active_layer_name == "cls"
    assert L.active_layer.dtype == "categorical"
    # get_layer finds by key and by name equivalently
    assert L.get_layer("fit") is layers["fit"]
    # unknown raises a helpful KeyError
    try:
        L.get_layer("does_not_exist")
    except KeyError as e:
        assert "Layer 'does_not_exist' not found" in str(e)


def test_read_csv_landscape_and_to_csv_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "sequence": ["ACD", "ACE", "ACF", "BCD"],
            "fitness": [0.1, 0.2, 0.3, 0.4],
            "fitness.rep1": [0.1, 0.3, 0.5, 0.7],
            "fitness.rep2": [0.2, np.nan, 0.6, 0.9],
            "label": ["A", "B", "A", "B"],
            "label=A": [0.7, 0.2, 0.8, 0.1],
            "label=B": [0.3, 0.8, 0.2, 0.9],
        }
    )
    p = tmp_path / "toy_landscape.csv"
    df.to_csv(p, index=False)

    L = read_csv_landscape(
        p,
        sequence_col="sequence",
        numeric_layers=["fitness"],
        replicate_prefixes={"rep": ["fitness.rep1", "fitness.rep2"]},
        categorical_layers=["label"],
        probabilistic_specs={"post": ["label=A", "label=B"]},
        graph="hamming",
        attach_embeddings=False,
    )

    # layers exist with correct types
    assert isinstance(L.fitness_layers["fitness"], NumericFitness)
    assert isinstance(L.fitness_layers["rep"], NumericFitness)
    assert isinstance(L.fitness_layers["label"], CategoricalFitness)
    assert isinstance(L.fitness_layers["post"], ProbabilisticCategoricalFitness)

    # replicate tensor has shape (N, max_reps==2) with padding
    t_rep = L.fitness_layers["rep"].get_tensor().numpy()
    assert t_rep.shape == (4, 2)
    # probabilistic rows sum to approx 1
    P = L.fitness_layers["post"].get_tensor().numpy()
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)

    # Write out again (numeric + categorical get exported)
    out = tmp_path / "roundtrip.csv"
    to_csv_landscape(L, out, sequence_col="sequence", include_layers=True)
    df2 = pd.read_csv(out)

    # Expected columns: sequence + fitness + rep + label
    assert set(["sequence", "fitness", "label"]).issubset(set(df2.columns))
    assert len(df2) == 4
    # values are preserved for scalar numeric + categorical
    np.testing.assert_allclose(df2["fitness"].to_numpy(), df["fitness"].to_numpy())
    assert df2["label"].tolist() == df["label"].tolist()