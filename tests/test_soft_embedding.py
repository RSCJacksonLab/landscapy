import numpy as np
import torch
import pytest
from pathlib import Path

# Assuming the ESMEmbedder class is in fitness_landscape/soft_embeddings.py
from fitness_landscape.embedding.soft_embedding import ESMEmbedder

@pytest.fixture(scope="module")
def embedder():
    """
    Provides a ESMEmbedder instance.
    Defaults to esm2_t6_8M_UR50D as the smallest ESM model.
    """
    model_name = "facebook/esm2_t6_8M_UR50D"
    return ESMEmbedder(model_name=model_name, device="cpu")

def test_embedder_initialization(embedder):
    """
    Tests that the ESMEmbedder initializes correctly.
    """
    assert embedder.model_name == "facebook/esm2_t6_8M_UR50D"
    assert embedder.device == "gpu" or embedder.device == "cpu"
    assert hasattr(embedder, "model")
    assert hasattr(embedder, "tokenizer")
    
    # Check that the embedding matrix for the alphabet is created
    alphabet_size = len(embedder.alphabet)
    embedding_dim = embedder.model.config.hidden_size
    assert embedder.embeddings_matrix.shape == (alphabet_size, embedding_dim)

def test_get_seq_ohe(embedder):
    """
    Tests the one-hot encoding helper function in a robust way
    that does not depend on the default alphabet order.
    """
    sequence = "ACW"
    alphabet_size = len(embedder.alphabet)
    ohe = embedder.get_seq_ohe(sequence)
    
    # --- Assertions ---
    assert ohe.shape == (len(sequence), alphabet_size)

    # Test 'A'
    idx_A = embedder.alphabet.index('A')
    expected_A = np.zeros(alphabet_size)
    expected_A[idx_A] = 1.0
    assert np.allclose(ohe[0], expected_A)

    # Test 'C'
    idx_C = embedder.alphabet.index('C')
    expected_C = np.zeros(alphabet_size)
    expected_C[idx_C] = 1.0
    assert np.allclose(ohe[1], expected_C)

    # Test 'W'
    idx_W = embedder.alphabet.index('W')
    expected_W = np.zeros(alphabet_size)
    expected_W[idx_W] = 1.0
    assert np.allclose(ohe[2], expected_W)

def test_forward_pass_shapes(embedder):
    """
    Tests the core `forward_pass` method for correct output shapes.
    """
    seq_len = 4
    batch_size = 2
    alphabet_size = len(embedder.alphabet)
    embedding_dim = embedder.model.config.hidden_size
    
    dummy_ohe = torch.zeros(batch_size, seq_len, alphabet_size, device=embedder.device)
    dummy_ohe[:, :, 0] = 1.0
    
    with torch.no_grad():
        out = embedder.forward_pass(dummy_ohe)
        hidden_state = out['hidden_states'][-1] 
    
    expected_shape = (batch_size, seq_len, embedding_dim)
    assert hidden_state.shape == expected_shape

def test_embedding_consistency(embedder):
    """
    Tests that embedding a string sequence and its corresponding one-hot
    tensor yield the same result.
    """
    sequence = "GATTACA"
    
    # 1. Embed the string sequence
    string_embedding = embedder.embed_sequences([sequence])
    
    # 2. Embed the equivalent one-hot encoded tensor
    ohe = embedder.get_seq_ohe(sequence)
    ohe_tensor = torch.from_numpy(ohe)
    ohe_embedding = embedder.embed_relaxed_seqs(ohe_tensor)

    # The resulting numpy arrays should be identical
    assert np.allclose(string_embedding, ohe_embedding, atol=1e-6)

def test_batch_embedding(embedder):
    """
    Tests that the high-level embedding functions work correctly for a batch.
    """
    sequences = ["ACD", "GWYQ", "RNSAA"]
    num_sequences = len(sequences)
    seq_len = len(sequences[0])
    embedding_dim = embedder.model.config.hidden_size
    
    embeddings = embedder.embed_sequences(sequences, batch_size=2)
    
    expected_shape = (num_sequences, embedding_dim)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == expected_shape

def test_save_and_load_embeddings(embedder, tmp_path: Path):
    """
    Tests that embeddings can be saved and loaded correctly.
    """
    # Create a dummy embedding
    original_embedding = np.random.rand(2, 5, 320).astype(np.float32)
    
    # Define a file path in the temporary directory
    file_path = tmp_path / "test_embedding.npy"
    
    # Save and then load the embedding
    embedder.save_embeddings(original_embedding, file_path)
    loaded_embedding = embedder.load_embeddings(file_path)
    
    # The loaded array should be identical to the original
    assert np.array_equal(original_embedding, loaded_embedding)


def test_extract_features_delegates_to_embedding_path(monkeypatch):
    """The convenience alias must not recurse into itself."""
    embedder = object.__new__(ESMEmbedder)
    expected = np.ones((2, 3), dtype=np.float32)
    calls = []

    def fake_embed(sequences, batch_size):
        calls.append((sequences, batch_size))
        return expected

    monkeypatch.setattr(embedder, "embed_relaxed_seqs", fake_embed)

    result = embedder.extract_features(["AC", "GT"], batch_size=7)

    assert result is expected
    assert calls == [(["AC", "GT"], 7)]
