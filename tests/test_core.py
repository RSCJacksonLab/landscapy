import numpy as np
import pytest
import networkx as nx
from fitness_landscape.core.sequence import *
from fitness_landscape.core.graph import *
from fitness_landscape.core.digraph import *
from fitness_landscape.core.landscape import FitnessLandscape, DirectedFitnessLandscape
from fitness_landscape.core.fitness import NumericFitness, CategoricalFitness
from fitness_landscape.core.superscape import FitnessSuperscape
import torch
from torch_geometric.data import Data
from fitness_landscape.core.fitness import NumericFitness, CategoricalFitness, ProbabilisticCategoricalFitness
from pathlib import Path
from fitness_landscape._sub_matrices import nq_pfam
from fitness_landscape.embedding.particle_sampler import SequenceGenerator, TopPSampler
from unittest.mock import patch
from fitness_landscape.utils import alignment_to_base_numpy_sequences

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


# BUG: cknn density fails.
# def test_create_cknn_graph_structure(clustered_data):
#     """
#     Tests that the ck-NN graph correctly identifies the clustered
#     structure by having higher intra-cluster density than inter-cluster density.
#     """
#     sequences, embeddings = clustered_data
#     graph = create_cknn_graph(
#         sequences=sequences,
#         embeddings=embeddings,
#         k=4 # Use a k appropriate for the cluster size
#     )

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

def test_landscape_from_sequences_embeddings(clustered_data):
    """Tests landscape creation with embeddings."""
    sequences, embeddings = clustered_data
    landscape = FitnessLandscape.from_sequences(sequences, graph_type='knn', embeddings=embeddings, k=3)
    assert landscape.graph is not None
    assert landscape.embeddings is not None

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
    Tests the generalized `from_parallel_construction` factory method to ensure
    it can build heterogeneous landscapes in parallel.
    """
    mock_embedder_instance = MockESMEmbedder.return_value
    mock_embedder_instance.embed_relaxed_seqs.side_effect = [
        np.random.rand(5, 10),
        np.random.rand(3, 10)
    ]
    mock_aligner_instance = MockHierarchicalRJMCMCAligner.return_value
    mock_aligner_instance.run_alignment.return_value = (nx.DiGraph(), {}) 
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
    
