"""Authoritative hard-token and relaxed-distribution ESM embedding support."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._const import PROT_20
from .._optional import require_optional

torch = require_optional(
    "torch",
    extra="embeddings",
    purpose="protein language-model embeddings",
)
transformers = require_optional(
    "transformers",
    extra="embeddings",
    purpose="protein language-model embeddings",
)
tqdm_module = require_optional(
    "tqdm",
    extra="embeddings",
    purpose="protein language-model embedding progress",
)

AutoTokenizer = transformers.AutoTokenizer
EsmForMaskedLM = transformers.EsmForMaskedLM
tqdm = tqdm_module.tqdm

DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_RELAXED_ALPHABET = tuple(PROT_20) + ("-",)

HardSequences = str | Sequence[str]
RelaxedSequence = np.ndarray | torch.Tensor
RelaxedSequences = RelaxedSequence | Sequence[RelaxedSequence]


class ESMEmbedder:
    """Embed hard strings or relaxed protein sequences with one ESM model.

    Hard sequences use one model token per input character. Characters are
    upper-cased, ``"-"`` is the gap token, and symbols absent from the model
    vocabulary use the tokenizer's unknown-token id. Relaxed inputs are
    non-negative, finite ``[length, alphabet]`` probability matrices whose rows
    sum to one. Their columns follow ``alphabet`` exactly; ``"gap"`` is
    normalized to ``"-"``, and at most one column may map to the model's
    unknown token.

    Both paths add exactly one classifier and end-of-sequence token, mask only
    right padding, mean-pool residue positions (including gaps but excluding
    special and padding tokens), restore caller order after length sorting, and
    return a float32 NumPy matrix with shape ``[n_sequences, hidden_size]``.

    Parameters
    ----------
    model_name : str, default='facebook/esm2_t6_8M_UR50D'
        Pinned Hugging Face ESM masked-language-model identifier.
    device : str, optional
        Torch device. If omitted, select CUDA when available, otherwise CPU.
    alphabet : sequence of str, optional
        Ordered model tokens corresponding to relaxed-input columns. The
        default is the 20 canonical amino acids followed by the gap token.
    batch_size : int, default=1
        Default inference batch size.

    Attributes
    ----------
    model_name : str
        Hugging Face model identifier.
    device : str
        Torch device used for inference.
    alphabet : list of str
        Normalized ordered tokens used for relaxed-input columns.
    batch_size : int
        Default inference batch size.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        device: str | None = None,
        alphabet: Sequence[str] | None = None,
        batch_size: int = 1,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = self._validate_batch_size(batch_size)
        self.alphabet = self._normalize_alphabet(
            DEFAULT_RELAXED_ALPHABET if alphabet is None else alphabet
        )
        self._load_model()

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return batch_size

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        token = str(symbol)
        if token.lower() == "gap":
            return "-"
        return token.upper() if len(token) == 1 else token

    @classmethod
    def _normalize_alphabet(cls, alphabet: Sequence[str]) -> list[str]:
        if isinstance(alphabet, str):
            raise TypeError(
                "alphabet must be a sequence of token strings, not a string"
            )
        normalized = [cls._normalize_symbol(symbol) for symbol in alphabet]
        if not normalized:
            raise ValueError("alphabet must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("alphabet tokens must be unique after gap normalization")
        return normalized

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = EsmForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.vocab_dict = self.tokenizer.get_vocab()
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        missing = [
            name
            for name, value in (
                ("cls_token_id", self.cls_token_id),
                ("eos_token_id", self.eos_token_id),
                ("pad_token_id", self.pad_token_id),
                ("unk_token_id", self.unk_token_id),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "ESM tokenizer is missing required special token ids: "
                + ", ".join(missing)
            )

        special_tokens = set(getattr(self.tokenizer, "all_special_tokens", ()))
        forbidden = [token for token in self.alphabet if token in special_tokens]
        if forbidden:
            raise ValueError(
                "relaxed alphabet cannot contain model special tokens: "
                + ", ".join(forbidden)
            )

        token_ids = [self._token_id(token) for token in self.alphabet]
        if len(set(token_ids)) != len(token_ids):
            raise ValueError(
                "relaxed alphabet entries must map to distinct model tokens; "
                "use at most one unsupported-symbol column"
            )
        self._alphabet_token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.device,
        )

        embeddings = self.model.esm.embeddings.word_embeddings
        with torch.no_grad():
            self.embeddings_matrix = embeddings(self._alphabet_token_ids).detach()
            special_ids = torch.tensor(
                [self.cls_token_id, self.eos_token_id, self.pad_token_id],
                dtype=torch.long,
                device=self.device,
            )
            special_embeddings = embeddings(special_ids).detach()
        self._cls_embedding = special_embeddings[0]
        self._eos_embedding = special_embeddings[1]
        self._pad_embedding = special_embeddings[2]

        for owner in (
            self.model,
            getattr(self.model, "config", None),
            getattr(self.model, "esm", None),
            getattr(getattr(self.model, "esm", None), "embeddings", None),
        ):
            if owner is not None and hasattr(owner, "token_dropout"):
                owner.token_dropout = False

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        return self.unk_token_id if token_id is None else int(token_id)

    def _freeze_esm(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _effective_batch_size(self, batch_size: int | None) -> int:
        return (
            self.batch_size
            if batch_size is None
            else self._validate_batch_size(batch_size)
        )

    def _normalize_hard_sequences(self, sequences: HardSequences) -> list[str]:
        if isinstance(sequences, str):
            values = [sequences]
        elif isinstance(sequences, Sequence):
            values = list(sequences)
        else:
            raise TypeError("hard sequences must be a string or a sequence of strings")
        if any(not isinstance(sequence, str) for sequence in values):
            raise TypeError("hard sequences must contain only strings")
        for sequence in values:
            if not sequence:
                raise ValueError("protein sequences must not be empty")
            if any(character.isspace() for character in sequence):
                raise ValueError("protein sequences must not contain whitespace")
        return values

    def _normalize_relaxed_sequences(
        self,
        sequences: RelaxedSequences,
    ) -> list[torch.Tensor]:
        if isinstance(sequences, (np.ndarray, torch.Tensor)):
            values: list[RelaxedSequence] = [sequences]
        elif isinstance(sequences, Sequence) and not isinstance(sequences, str):
            values = list(sequences)
        else:
            raise TypeError(
                "relaxed sequences must be a matrix or a sequence of matrices"
            )

        matrices: list[torch.Tensor] = []
        for sequence in values:
            if not isinstance(sequence, (np.ndarray, torch.Tensor)):
                raise TypeError(
                    "relaxed sequences must contain only NumPy or Torch matrices"
                )
            matrix = (
                torch.as_tensor(sequence).detach().to(dtype=torch.float32, device="cpu")
            )
            if matrix.ndim != 2:
                raise ValueError(
                    "each relaxed sequence must have shape [length, alphabet]"
                )
            if matrix.shape[0] == 0:
                raise ValueError("relaxed sequences must not be empty")
            if matrix.shape[1] != len(self.alphabet):
                raise ValueError(
                    "relaxed sequence width must equal len(alphabet): "
                    f"{matrix.shape[1]} != {len(self.alphabet)}"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError("relaxed sequence probabilities must be finite")
            if torch.any(matrix < 0):
                raise ValueError("relaxed sequence probabilities must be non-negative")
            row_sums = matrix.sum(dim=1)
            if not torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("each relaxed sequence row must sum to one")
            matrices.append(matrix)
        return matrices

    def _hard_tokens(self, sequence: str) -> list[int]:
        residue_ids = [
            self._token_id(self._normalize_symbol(character)) for character in sequence
        ]
        return [self.cls_token_id, *residue_ids, self.eos_token_id]

    def _hard_batch_iterator(
        self,
        sequences: list[str],
        batch_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[int], tuple[int, ...]]]:
        sorted_inputs = sorted(
            enumerate(sequences),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        for start in range(0, len(sorted_inputs), batch_size):
            batch = sorted_inputs[start : start + batch_size]
            batch_indices, batch_sequences = zip(*batch)
            token_rows = [self._hard_tokens(sequence) for sequence in batch_sequences]
            lengths = [len(sequence) for sequence in batch_sequences]
            max_tokens = max(map(len, token_rows))
            input_ids = torch.full(
                (len(batch), max_tokens),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, token_ids in enumerate(token_rows):
                width = len(token_ids)
                input_ids[row, :width] = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask[row, :width] = 1
            yield input_ids, attention_mask, lengths, batch_indices

    def _relaxed_batch_iterator(
        self,
        sequences: list[torch.Tensor],
        batch_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[int], tuple[int, ...]]]:
        sorted_inputs = sorted(
            enumerate(sequences),
            key=lambda item: item[1].shape[0],
            reverse=True,
        )
        for start in range(0, len(sorted_inputs), batch_size):
            batch = sorted_inputs[start : start + batch_size]
            batch_indices, batch_sequences = zip(*batch)
            lengths = [sequence.shape[0] for sequence in batch_sequences]
            max_tokens = max(lengths) + 2
            input_embeddings = self._pad_embedding.repeat(len(batch), max_tokens, 1)
            attention_mask = torch.zeros(
                (len(batch), max_tokens),
                dtype=torch.long,
                device=self.device,
            )
            for row, sequence in enumerate(batch_sequences):
                probabilities = sequence.to(
                    device=self.device,
                    dtype=self.embeddings_matrix.dtype,
                )
                residue_embeddings = probabilities @ self.embeddings_matrix
                length = sequence.shape[0]
                input_embeddings[row, 0] = self._cls_embedding
                input_embeddings[row, 1 : length + 1] = residue_embeddings
                input_embeddings[row, length + 1] = self._eos_embedding
                attention_mask[row, : length + 2] = 1
            yield input_embeddings, attention_mask, lengths, batch_indices

    def _model_forward(
        self,
        model_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mode: str,
    ) -> Any:
        kwargs = {
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if mode == "hard":
            kwargs["input_ids"] = model_inputs
        else:
            kwargs["inputs_embeds"] = model_inputs
        return self.model(**kwargs)

    def forward_pass(
        self,
        relaxed_seqs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Any:
        """Run a low-level ESM pass over unwrapped relaxed positions.

        This method does not add special tokens. High-level embedding methods
        should normally be preferred because they apply the shared wrapping,
        padding, masking, and pooling contract.

        Parameters
        ----------
        relaxed_seqs : torch.Tensor
            Tensor with shape ``[length, alphabet]`` or
            ``[batch, length, alphabet]``.
        attention_mask : torch.Tensor, optional
            Mask with shape ``[batch, length]``. If omitted, all positions are
            attended.

        Returns
        -------
        output : object
            Hugging Face masked-language-model output.
        """
        tensor = torch.as_tensor(relaxed_seqs, dtype=torch.float32, device=self.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-1] != len(self.alphabet):
            raise ValueError(
                "relaxed_seqs must have shape [length, alphabet] or "
                "[batch, length, alphabet]"
            )
        if attention_mask is None:
            mask = torch.ones(tensor.shape[:2], dtype=torch.long, device=self.device)
        else:
            mask = torch.as_tensor(attention_mask, dtype=torch.long, device=self.device)
            if tuple(mask.shape) != tuple(tensor.shape[:2]):
                raise ValueError("attention_mask shape must match batch and length")
        inputs_embeds = tensor.to(self.embeddings_matrix.dtype) @ self.embeddings_matrix
        return self._model_forward(inputs_embeds, mask, mode="relaxed")

    def batch_iterator(
        self,
        sequences: HardSequences | RelaxedSequences,
        batch_size: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[int], tuple[int, ...]]]:
        """Yield model-ready length-sorted batches and original row indices.

        String inputs yield hard token-id batches. Matrix inputs yield relaxed
        input-embedding batches. Both include classifier/end tokens and right
        padding.

        Parameters
        ----------
        sequences : str, sequence of str, matrix, or sequence of matrices
            Homogeneous hard strings or relaxed probability matrices.
        batch_size : int, optional
            Batch size. If omitted, use the constructor setting.

        Yields
        ------
        model_inputs : torch.Tensor
            Hard token ids ``[B, T]`` or relaxed input embeddings ``[B, T, H]``.
        attention_mask : torch.Tensor
            Integer mask with one for classifier, residue, and end tokens and
            zero for right padding.
        original_lengths : list of int
            Residue counts excluding special tokens and padding.
        batch_indices : tuple of int
            Original caller positions for restoring input order.
        """
        effective = self._effective_batch_size(batch_size)
        if self._is_hard_input(sequences):
            values = self._normalize_hard_sequences(sequences)
            yield from self._hard_batch_iterator(values, effective)
        else:
            values = self._normalize_relaxed_sequences(sequences)
            yield from self._relaxed_batch_iterator(values, effective)

    @staticmethod
    def _is_hard_input(sequences: object) -> bool:
        if isinstance(sequences, str):
            return True
        if isinstance(sequences, Sequence) and not isinstance(
            sequences, (np.ndarray, torch.Tensor)
        ):
            values = list(sequences)
            return not values or isinstance(values[0], str)
        return False

    def get_seq_ohe(self, sequence: str) -> NDArray[np.float32]:
        """Encode a hard string in the configured relaxed alphabet.

        Parameters
        ----------
        sequence : str
            Non-empty protein sequence. ``"gap"`` alphabet entries and ``"-"``
            sequence characters refer to the same column. Symbols absent from
            the relaxed alphabet are rejected rather than silently zeroed.

        Returns
        -------
        encoding : ndarray
            Float32 one-hot matrix with shape ``[length, len(alphabet)]``.
        """
        values = self._normalize_hard_sequences(sequence)
        mapping = {token: index for index, token in enumerate(self.alphabet)}
        encoding = np.zeros((len(sequence), len(self.alphabet)), dtype=np.float32)
        for position, symbol in enumerate(values[0]):
            token = self._normalize_symbol(symbol)
            if token not in mapping:
                raise ValueError(
                    f"sequence symbol {symbol!r} is absent from the relaxed alphabet"
                )
            encoding[position, mapping[token]] = 1.0
        return encoding

    def _progress_iterator(
        self,
        iterator: Iterator,
        *,
        count: int,
        batch_size: int,
    ) -> Iterator:
        return tqdm(
            iterator,
            total=math.ceil(count / batch_size) if count else 0,
            desc="Embedding",
        )

    def _collect_embeddings(
        self,
        iterator: Iterator,
        *,
        count: int,
        batch_size: int,
        mode: str,
    ) -> NDArray[np.float32]:
        if count == 0:
            hidden_size = int(self.model.config.hidden_size)
            return np.empty((0, hidden_size), dtype=np.float32)
        features: list[torch.Tensor | None] = [None] * count
        batches = self._progress_iterator(
            iterator,
            count=count,
            batch_size=batch_size,
        )
        for model_inputs, attention_mask, lengths, batch_indices in batches:
            with torch.no_grad():
                output = self._model_forward(
                    model_inputs,
                    attention_mask,
                    mode=mode,
                )
                hidden_states = output.hidden_states[-1]
            for row, original_index in enumerate(batch_indices):
                length = lengths[row]
                features[original_index] = (
                    hidden_states[row, 1 : length + 1]
                    .mean(dim=0)
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                )
        return torch.stack(features).numpy()

    def embed_relaxed_seqs(
        self,
        sequences: RelaxedSequences,
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """Mean-pool relaxed sequence distributions with the ESM model.

        Parameters
        ----------
        sequences : matrix or sequence of matrices
            One NumPy/Torch matrix or a sequence of matrices. Every matrix has
            shape ``[length, len(alphabet)]`` and contains finite,
            non-negative rows that sum to one.
        batch_size : int, optional
            Batch size. If omitted, use the constructor setting.

        Returns
        -------
        embeddings : ndarray
            Float32 matrix with shape ``[n_sequences, hidden_size]`` in caller
            order.
        """
        values = self._normalize_relaxed_sequences(sequences)
        effective = self._effective_batch_size(batch_size)
        return self._collect_embeddings(
            self._relaxed_batch_iterator(values, effective),
            count=len(values),
            batch_size=effective,
            mode="relaxed",
        )

    def lm_output_probabilities(
        self,
        sequences: HardSequences | RelaxedSequences,
        batch_size: int | None = None,
    ) -> list[NDArray[np.float32]]:
        """Return per-residue model probabilities for the relaxed alphabet.

        Probabilities are selected from the full model softmax and are not
        renormalized after restricting columns to ``alphabet``.

        Parameters
        ----------
        sequences : str, sequence of str, matrix, or sequence of matrices
            Homogeneous hard strings or relaxed probability matrices.
        batch_size : int, optional
            Batch size. If omitted, use the constructor setting.

        Returns
        -------
        probabilities : list of ndarray
            Caller-ordered float32 matrices with shape
            ``[sequence_length, len(alphabet)]``.
        """
        effective = self._effective_batch_size(batch_size)
        if self._is_hard_input(sequences):
            values = self._normalize_hard_sequences(sequences)
            mode = "hard"
            iterator = self._hard_batch_iterator(values, effective)
        else:
            values = self._normalize_relaxed_sequences(sequences)
            mode = "relaxed"
            iterator = self._relaxed_batch_iterator(values, effective)
        output_probabilities: list[NDArray[np.float32] | None] = [None] * len(values)
        batches = self._progress_iterator(
            iterator,
            count=len(values),
            batch_size=effective,
        )
        for model_inputs, attention_mask, lengths, batch_indices in batches:
            with torch.no_grad():
                logits = self._model_forward(
                    model_inputs,
                    attention_mask,
                    mode=mode,
                ).logits
                probabilities = logits.softmax(dim=-1)
            for row, original_index in enumerate(batch_indices):
                length = lengths[row]
                selected = probabilities[
                    row,
                    1 : length + 1,
                    self._alphabet_token_ids,
                ]
                output_probabilities[original_index] = (
                    selected.detach().to(dtype=torch.float32, device="cpu").numpy()
                )
        return output_probabilities

    def embed_sequences(
        self,
        sequences: HardSequences,
        batch_size: int | None = None,
    ) -> NDArray[np.float32]:
        """Mean-pool hard protein strings with the ESM model.

        Parameters
        ----------
        sequences : str or sequence of str
            One non-empty protein string or a sequence of strings.
        batch_size : int, optional
            Batch size. If omitted, use the constructor setting.

        Returns
        -------
        embeddings : ndarray
            Float32 matrix with shape ``[n_sequences, hidden_size]`` in caller
            order.
        """
        values = self._normalize_hard_sequences(sequences)
        effective = self._effective_batch_size(batch_size)
        return self._collect_embeddings(
            self._hard_batch_iterator(values, effective),
            count=len(values),
            batch_size=effective,
            mode="hard",
        )

    def extract_features(
        self,
        sequences: HardSequences,
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """Delegate hard-string feature extraction to :meth:`embed_sequences`.

        Parameters
        ----------
        sequences : str or sequence of str
            One protein string or a sequence of strings.
        batch_size : int, default=32
            Inference batch size.

        Returns
        -------
        embeddings : ndarray
            Float32 matrix with shape ``[n_sequences, hidden_size]``.
        """
        return self.embed_sequences(sequences, batch_size)

    def save_embeddings(
        self,
        embeddings: ArrayLike,
        embedding_path: Path,
    ) -> None:
        """Save an embedding array in NumPy ``.npy`` format.

        Parameters
        ----------
        embeddings : array-like
            Embedding values to save.
        embedding_path : pathlib.Path
            Output path.
        """
        np.save(embedding_path, embeddings)

    def load_embeddings(self, embedding_path: Path) -> NDArray:
        """Load an embedding array from NumPy ``.npy`` format.

        Parameters
        ----------
        embedding_path : pathlib.Path
            Input path.

        Returns
        -------
        embeddings : ndarray
            Stored embedding values.
        """
        return np.load(embedding_path)
