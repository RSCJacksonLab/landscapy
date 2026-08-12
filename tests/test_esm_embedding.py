"""Offline tests for the consolidated ESM embedding implementation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fitness_landscape.embedding import esm
from fitness_landscape.embedding.esm import ESMEmbedder
from fitness_landscape.embedding.soft_embedding import ESMEmbedder as SoftAlias
from fitness_landscape.core.sequence import BaseNumpySequence, SoftSequence
from fitness_landscape import embedding as embedding_package
from fitness_landscape import utils


class FakeTokenizer:
    vocab = {
        "<cls>": 0,
        "<pad>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "A": 4,
        "C": 5,
        "D": 6,
        "-": 7,
        "X": 8,
        "<mask>": 9,
    }
    cls_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    all_special_tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(len(FakeTokenizer.vocab), 4)
        with torch.no_grad():
            for index in range(len(FakeTokenizer.vocab)):
                self.word_embeddings.weight[index] = torch.tensor(
                    [index, index + 0.25, -index, 1.0]
                )
        self.esm = SimpleNamespace(
            token_dropout=True,
            embeddings=SimpleNamespace(
                word_embeddings=self.word_embeddings,
                token_dropout=True,
            ),
        )
        self.config = SimpleNamespace(hidden_size=4, token_dropout=True)
        self.token_dropout = True
        self.calls = []

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        self.calls.append(
            {
                "input_ids": None
                if input_ids is None
                else input_ids.detach().cpu().clone(),
                "inputs_embeds": inputs_embeds.detach().cpu().clone(),
                "attention_mask": attention_mask.detach().cpu().clone(),
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )
        vocab_axis = torch.arange(
            len(FakeTokenizer.vocab),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        logits = inputs_embeds[..., :1] + vocab_axis
        return SimpleNamespace(hidden_states=(inputs_embeds,), logits=logits)


@pytest.fixture
def fake_backend(monkeypatch):
    loaded = {"tokenizers": [], "models": []}

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(model_name):
            loaded["tokenizers"].append(model_name)
            return FakeTokenizer()

    class ModelLoader:
        @staticmethod
        def from_pretrained(model_name):
            loaded["models"].append(model_name)
            return FakeModel()

    monkeypatch.setattr(esm, "AutoTokenizer", TokenizerLoader)
    monkeypatch.setattr(esm, "EsmForMaskedLM", ModelLoader)
    monkeypatch.setattr(esm, "tqdm", lambda iterator, **_kwargs: iterator)
    return loaded


@pytest.fixture
def embedder(fake_backend):
    return ESMEmbedder(
        model_name="pinned/fake-esm",
        device="cpu",
        alphabet=["A", "C", "D", "gap", "X"],
        batch_size=2,
    )


def _token_embedding(embedder, token):
    token_id = embedder.tokenizer.convert_tokens_to_ids(token)
    return embedder.model.word_embeddings.weight[token_id].detach().numpy()


def test_public_and_compatibility_imports_share_one_implementation():
    assert SoftAlias is ESMEmbedder


def test_initialization_builds_one_relaxed_mapping_and_disables_dropout(
    embedder,
    fake_backend,
):
    assert fake_backend == {
        "tokenizers": ["pinned/fake-esm"],
        "models": ["pinned/fake-esm"],
    }
    assert embedder.alphabet == ["A", "C", "D", "-", "X"]
    assert embedder.embeddings_matrix.shape == (5, 4)
    assert embedder.embeddings_matrix.dtype == torch.float32
    assert embedder.model.token_dropout is False
    assert embedder.model.config.token_dropout is False
    assert embedder.model.esm.token_dropout is False
    assert embedder.model.esm.embeddings.token_dropout is False


def test_hard_embedding_special_tokens_masks_mean_pooling_and_order(embedder):
    result = embedder.embed_sequences(["A", "ACD", "C-"], batch_size=2)

    expected = np.stack(
        [
            _token_embedding(embedder, "A"),
            np.mean(
                [_token_embedding(embedder, token) for token in ("A", "C", "D")],
                axis=0,
            ),
            np.mean(
                [_token_embedding(embedder, token) for token in ("C", "-")],
                axis=0,
            ),
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(result, expected)
    assert result.shape == (3, 4)
    assert result.dtype == np.float32

    first_call = embedder.model.calls[0]
    np.testing.assert_array_equal(
        first_call["input_ids"],
        np.array(
            [
                [0, 4, 5, 6, 2],
                [0, 5, 7, 2, 1],
            ]
        ),
    )
    np.testing.assert_array_equal(
        first_call["attention_mask"],
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
            ]
        ),
    )


def test_hard_embedding_uppercases_and_maps_unknown_to_unk(embedder):
    embedder.embed_sequences("a?", batch_size=1)
    np.testing.assert_array_equal(
        embedder.model.calls[-1]["input_ids"],
        np.array([[0, 4, 3, 2]]),
    )


def test_relaxed_embedding_mapping_masks_pooling_dtype_and_order(embedder):
    short = np.array([[0.25, 0.75, 0.0, 0.0, 0.0]], dtype=np.float64)
    long = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    result = embedder.embed_relaxed_seqs([short, long], batch_size=2)

    expected_short = 0.25 * _token_embedding(embedder, "A") + 0.75 * _token_embedding(
        embedder, "C"
    )
    expected_long = np.mean(
        [_token_embedding(embedder, "D"), _token_embedding(embedder, "-")],
        axis=0,
    )
    np.testing.assert_allclose(result, np.stack([expected_short, expected_long]))
    assert result.dtype == np.float32

    call = embedder.model.calls[-1]
    assert call["input_ids"] is None
    np.testing.assert_array_equal(
        call["attention_mask"],
        np.array([[1, 1, 1, 1], [1, 1, 1, 0]]),
    )
    np.testing.assert_allclose(
        call["inputs_embeds"][0, 0], _token_embedding(embedder, "<cls>")
    )
    np.testing.assert_allclose(
        call["inputs_embeds"][0, 3], _token_embedding(embedder, "<eos>")
    )
    np.testing.assert_allclose(
        call["inputs_embeds"][1, 3], _token_embedding(embedder, "<pad>")
    )


def test_hard_and_equivalent_one_hot_paths_match(embedder):
    hard = embedder.embed_sequences(["AC", "D-"])
    relaxed = embedder.embed_relaxed_seqs(
        [embedder.get_seq_ohe("AC"), embedder.get_seq_ohe("D-")]
    )

    np.testing.assert_allclose(hard, relaxed)


def test_batch_iterator_exposes_consistent_wrapping_and_indices(embedder):
    hard_batch = next(embedder.batch_iterator(["A", "AC"], batch_size=2))
    assert hard_batch[0].ndim == 2
    assert hard_batch[2] == [2, 1]
    assert hard_batch[3] == (1, 0)

    relaxed_batch = next(
        embedder.batch_iterator(
            [
                np.array([[1, 0, 0, 0, 0]], dtype=float),
                np.array([[0, 1, 0, 0, 0]], dtype=float),
            ],
            batch_size=2,
        )
    )
    assert relaxed_batch[0].shape == (2, 3, 4)
    assert relaxed_batch[1].dtype == torch.long
    assert relaxed_batch[2] == [1, 1]
    assert relaxed_batch[3] == (0, 1)


def test_forward_pass_is_low_level_and_validates_mask(embedder):
    relaxed = torch.zeros((2, 3, len(embedder.alphabet)))
    relaxed[:, :, 0] = 1.0
    output = embedder.forward_pass(relaxed)
    assert output.hidden_states[-1].shape == (2, 3, 4)
    np.testing.assert_array_equal(
        embedder.model.calls[-1]["attention_mask"],
        np.ones((2, 3)),
    )

    with pytest.raises(ValueError, match="attention_mask shape"):
        embedder.forward_pass(relaxed, torch.ones((2, 2)))
    with pytest.raises(ValueError, match="relaxed_seqs must have shape"):
        embedder.forward_pass(torch.zeros((2, 3, 4, 5)))


def test_get_seq_ohe_maps_gap_and_rejects_unrepresentable_unknown(embedder):
    encoding = embedder.get_seq_ohe("a-")
    assert encoding.dtype == np.float32
    np.testing.assert_array_equal(
        encoding,
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        ),
    )
    with pytest.raises(ValueError, match="absent from the relaxed alphabet"):
        embedder.get_seq_ohe("?")


def test_language_model_probabilities_support_hard_and_relaxed_inputs(embedder):
    hard = embedder.lm_output_probabilities(["AC", "A"], batch_size=2)
    relaxed = embedder.lm_output_probabilities(
        np.array([[1, 0, 0, 0, 0]], dtype=float),
        batch_size=1,
    )

    assert [array.shape for array in hard] == [(2, 5), (1, 5)]
    assert [array.dtype for array in hard] == [np.float32, np.float32]
    assert relaxed[0].shape == (1, 5)
    assert relaxed[0].dtype == np.float32
    assert np.all((hard[0] >= 0) & (hard[0] <= 1))


def test_progress_wraps_lazy_iterator_without_materializing_batches(
    embedder, monkeypatch
):
    observed = {}

    def fake_tqdm(iterator, **kwargs):
        observed["is_list"] = isinstance(iterator, list)
        observed["total"] = kwargs["total"]
        return iterator

    monkeypatch.setattr(esm, "tqdm", fake_tqdm)
    embedder.embed_sequences(["A", "AC", "ACD"], batch_size=2)

    assert observed == {"is_list": False, "total": 2}


def test_extract_features_delegates_to_hard_embedding_path(monkeypatch):
    embedder = object.__new__(ESMEmbedder)
    expected = np.ones((2, 3), dtype=np.float32)
    calls = []

    def fake_embed(sequences, batch_size):
        calls.append((sequences, batch_size))
        return expected

    monkeypatch.setattr(embedder, "embed_sequences", fake_embed)
    result = embedder.extract_features(["AC", "GT"], batch_size=7)

    assert result is expected
    assert calls == [(["AC", "GT"], 7)]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(fake_backend, batch_size):
    with pytest.raises(ValueError, match="positive"):
        ESMEmbedder(alphabet=["A"], batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [True, 1.5, "2"])
def test_batch_size_must_be_an_integer(fake_backend, batch_size):
    with pytest.raises(TypeError, match="integer"):
        ESMEmbedder(alphabet=["A"], batch_size=batch_size)


@pytest.mark.parametrize(
    "alphabet, message",
    [
        ("AC", "sequence of token strings"),
        ([], "must not be empty"),
        (["gap", "-"], "unique after gap normalization"),
        (["<mask>"], "cannot contain model special tokens"),
        (["?", "!"], "distinct model tokens"),
    ],
)
def test_relaxed_alphabet_validation(fake_backend, alphabet, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ESMEmbedder(alphabet=alphabet)


@pytest.mark.parametrize(
    "sequences, message",
    [
        (123, "string or a sequence of strings"),
        (["A", 2], "only strings"),
        ("", "must not be empty"),
        ("A C", "must not contain whitespace"),
    ],
)
def test_hard_input_validation(embedder, sequences, message):
    with pytest.raises((TypeError, ValueError), match=message):
        embedder.embed_sequences(sequences)


@pytest.mark.parametrize(
    "sequence, message",
    [
        ([[1, 0, 0, 0, 0]], "only NumPy or Torch matrices"),
        (np.ones(5), "shape"),
        (np.empty((0, 5)), "must not be empty"),
        (np.ones((1, 4)), "width"),
        (np.array([[np.nan, 0, 0, 0, 0]]), "finite"),
        (np.array([[-1, 2, 0, 0, 0]]), "non-negative"),
        (np.array([[0.2, 0.2, 0.2, 0.2, 0.0]]), "sum to one"),
    ],
)
def test_relaxed_input_validation(embedder, sequence, message):
    with pytest.raises((TypeError, ValueError), match=message):
        embedder.embed_relaxed_seqs(sequence)


def test_empty_batches_have_stable_return_types(embedder):
    hard = embedder.embed_sequences([])
    relaxed = embedder.embed_relaxed_seqs([])
    probabilities = embedder.lm_output_probabilities([])

    assert hard.shape == relaxed.shape == (0, 4)
    assert hard.dtype == relaxed.dtype == np.float32
    assert probabilities == []


def test_freeze_and_embedding_round_trip(embedder, tmp_path: Path):
    embedder._freeze_esm()
    assert all(not parameter.requires_grad for parameter in embedder.model.parameters())

    expected = np.arange(8, dtype=np.float32).reshape(2, 4)
    destination = tmp_path / "embeddings.npy"
    embedder.save_embeddings(expected, destination)
    np.testing.assert_array_equal(embedder.load_embeddings(destination), expected)


def test_sequence_helper_routes_hard_strings_to_authoritative_embedder(monkeypatch):
    observed = {}

    class RecordingEmbedder:
        def __init__(self, **kwargs):
            observed["constructor"] = kwargs

        def embed_sequences(self, sequences, batch_size):
            observed["sequences"] = sequences
            observed["batch_size"] = batch_size
            return np.zeros((len(sequences), 2), dtype=np.float32)

    monkeypatch.setattr(embedding_package, "ESMEmbedder", RecordingEmbedder)
    sequences = [
        BaseNumpySequence(["A", "gap"], alphabet=["A", "gap"]),
        BaseNumpySequence(["C", "A"], alphabet=["A", "C"]),
    ]

    result = utils._compute_embeddings_from_sequences(
        sequences,
        model_name="pinned/fake-esm",
        device="cpu",
        batch_size=2,
        embedding_mode="hard",
    )

    assert result.shape == (2, 2)
    assert observed["sequences"] == ["A-", "CA"]
    assert observed["batch_size"] == 2
    assert "alphabet" not in observed["constructor"]


def test_sequence_helper_remaps_relaxed_columns_to_one_shared_alphabet(monkeypatch):
    observed = {}

    class RecordingEmbedder:
        def __init__(self, **kwargs):
            observed["constructor"] = kwargs

        def embed_relaxed_seqs(self, sequences, batch_size):
            observed["sequences"] = sequences
            observed["batch_size"] = batch_size
            return np.zeros((len(sequences), 2), dtype=np.float32)

    monkeypatch.setattr(embedding_package, "ESMEmbedder", RecordingEmbedder)
    first = SoftSequence(
        np.array([[0.25, 0.75]], dtype=float),
        alphabet=["C", "A"],
    )
    second = SoftSequence(
        np.array([[0.6, 0.4]], dtype=float),
        alphabet=["A", "C"],
        gap_posterior=np.array([[0.5]], dtype=float),
    )

    result = utils._compute_embeddings_from_sequences(
        [first, second],
        model_name="pinned/fake-esm",
        device="cpu",
        batch_size=2,
        embedding_mode="soft",
    )

    assert result.shape == (2, 2)
    assert observed["constructor"]["alphabet"] == ["C", "A", "-"]
    np.testing.assert_allclose(observed["sequences"][0], [[0.25, 0.75, 0.0]])
    np.testing.assert_allclose(observed["sequences"][1], [[0.2, 0.3, 0.5]])
    assert observed["batch_size"] == 2
