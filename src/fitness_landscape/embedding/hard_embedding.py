import numpy as np
import torch

from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

SequenceLike = Union[str, np.ndarray, torch.Tensor, List[int]]


class ESMEmbedder:
    """
    Token-level ESM embedder that mirrors the soft embedding API but
    feeds discrete token ids via the standard Hugging Face workflow.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: Optional[str] = None,
        alphabet: List[str] = list("ACDEFGHIKLMNPQRSTVWY-") + ["<cls>", "<eos>", "<pad>", "<mask>"],
        batch_size: int = 1,
    ) -> None:
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.alphabet = alphabet
        self.batch_size = batch_size
        self._load_model()

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = EsmForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.vocab_dict = self.tokenizer.get_vocab()
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def _freeze_esm(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def _ensure_list(self, sequences: Union[SequenceLike, List[SequenceLike]]) -> List[SequenceLike]:
        if isinstance(sequences, list):
            return sequences
        return [sequences]

    def _sequence_length(self, sequence: SequenceLike) -> int:
        if isinstance(sequence, str):
            return len(sequence)
        tensor = torch.as_tensor(sequence)
        return tensor.shape[0]

    def _ensure_special_tokens(self, tokens: List[int]) -> List[int]:
        tokens = list(tokens)
        if self.cls_token_id is not None and (not tokens or tokens[0] != self.cls_token_id):
            tokens = [self.cls_token_id] + tokens
        if self.eos_token_id is not None and (not tokens or tokens[-1] != self.eos_token_id):
            tokens = tokens + [self.eos_token_id]
        return tokens

    def _token_length(self, tokens: List[int]) -> int:
        length = len(tokens)
        if self.cls_token_id is not None and tokens and tokens[0] == self.cls_token_id:
            length -= 1
        if self.eos_token_id is not None and tokens and tokens[-1] == self.eos_token_id:
            length -= 1
        return length

    def _sequence_to_tokens(self, sequence: SequenceLike) -> Tuple[List[int], int]:
        if isinstance(sequence, str):
            encoding = self.tokenizer(sequence, add_special_tokens=True, return_attention_mask=False)
            tokens = encoding["input_ids"]
            original_length = len(sequence)
        else:
            tensor = torch.as_tensor(sequence)
            if tensor.dim() == 2 and tensor.shape[-1] == len(self.alphabet):
                tensor = torch.argmax(tensor, dim=-1)
            tensor = tensor.view(-1).long()
            tokens = tensor.cpu().tolist()
            original_length = self._token_length(tokens)

        tokens = self._ensure_special_tokens(tokens)
        return tokens, original_length

    def batch_iterator(
        self,
        sequences: Union[SequenceLike, List[SequenceLike]],
        batch_size: Optional[int] = None,
    ):
        sequences_list = self._ensure_list(sequences)
        _batch_size = batch_size if batch_size is not None else self.batch_size

        sorted_with_indices = sorted(
            enumerate(sequences_list),
            key=lambda x: self._sequence_length(x[1]),
            reverse=True,
        )

        for i in range(0, len(sorted_with_indices), _batch_size):
            batch_with_indices = sorted_with_indices[i : i + _batch_size]
            batch_indices, batch_data = zip(*batch_with_indices)

            token_dicts = []
            original_lengths: List[int] = []

            for seq in batch_data:
                tokens, original_length = self._sequence_to_tokens(seq)
                token_dicts.append(
                    {
                        "input_ids": tokens,
                        "attention_mask": [1] * len(tokens),
                    }
                )
                original_lengths.append(original_length)

            padded = self.tokenizer.pad(
                token_dicts,
                padding=True,
                return_tensors="pt",
            )

            input_ids = padded["input_ids"].to(self.device)
            attention_mask = padded["attention_mask"].to(self.device)

            yield input_ids, attention_mask, original_lengths, batch_indices

    def forward_pass(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids)

        return self.model(
            input_ids=token_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
            return_dict=True,
        )

    def embed_relaxed_seqs(
        self,
        sequences: Union[SequenceLike, List[SequenceLike]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        sequences_list = self._ensure_list(sequences)
        features: List[Optional[torch.Tensor]] = [None] * len(sequences_list)

        iterator = list(self.batch_iterator(sequences, batch_size))

        for token_batch, mask_batch, original_lengths, batch_indices in tqdm(iterator, desc="Embedding"):
            with torch.no_grad():
                hidden_states = self.forward_pass(token_batch, mask_batch).hidden_states[-1]

            for j, original_idx in enumerate(batch_indices):
                length = original_lengths[j]
                embedding = hidden_states[j, 1 : length + 1].mean(dim=0).cpu()
                features[original_idx] = embedding

        stacked = torch.stack(features)
        return stacked.numpy()

    def lm_output_probabilities(
        self,
        sequences: Union[SequenceLike, List[SequenceLike]],
        batch_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        sequences_list = self._ensure_list(sequences)
        output_probabilities: List[Optional[np.ndarray]] = [None] * len(sequences_list)

        iterator = list(self.batch_iterator(sequences, batch_size))

        vocab_indices = [self.vocab_dict[aa] for aa in self.alphabet if aa in self.vocab_dict]

        for token_batch, mask_batch, original_lengths, batch_indices in tqdm(iterator, desc="Embedding"):
            with torch.no_grad():
                logits = self.forward_pass(token_batch, mask_batch).logits
                probabilities = logits.softmax(dim=-1)

            for j, original_idx in enumerate(batch_indices):
                length = original_lengths[j]
                probability = probabilities[j, 1 : length + 1, vocab_indices].cpu().numpy()
                output_probabilities[original_idx] = probability.astype(np.float32)

        return output_probabilities

    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        return self.embed_relaxed_seqs(sequences, batch_size)

    def extract_features(
        self,
        sequences: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        return self.embed_relaxed_seqs(sequences, batch_size)

    def save_embeddings(self, embeddings: np.ndarray, embedding_path: Path) -> None:
        np.save(embedding_path, embeddings)

    def load_embeddings(self, embedding_path: Path) -> np.ndarray:
        return np.load(embedding_path)
