import numpy as np
import torch

from numpy.typing import ArrayLike
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EsmForMaskedLM, AutoTokenizer
from typing import List, Optional, Union


class ESMEmbedder:
    """
    Class for embedding protein sequences using ESM models.
    Supports both relaxed sequences and one-hot encoded sequences.

    Attributes
    ----------
    model_name : str
        Name of the ESM model to use for embeddings.
    device : str
        Device to run the model on, either 'cuda' or 'cpu'.
    alphabet : List[str]
        List of amino acids expected for embedding.
    """

    def __init__(self,
                 model_name: str = "facebook/esm2_t6_8M_UR50D",
                 device: Optional[str] = None,
                 alphabet: List = list('ACDEFGHIKLMNPQRSTVWY-') + ['<cls>', '<eos>', '<pad>','<mask>'],
                 batch_size: int = 1) -> None:
        """
        Initialise PLM embedder class by initialising the provided
        model.

        Parameters
        ----------
        model_name : str
            HuggingFace PLM model name.

        device : str, default = None
            Device to use, if None will autoselect.

        alphabet : List
            Tokens expected for embedding.

        batch_size : int, default = 1
        """
        
        # device management
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # load model
        self.alphabet = alphabet
        self.model_name = model_name
        self.batch_size = batch_size
        self._load_model()

    def _load_model(self):
        """
        Load tokenizer and model given name. Tokenizer is used to
        construct an embedding matrix for use on relaxed sequences.
        """
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = EsmForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # set up model for inferece
        self.model.esm.token_dropout = False
        self.model.config.token_dropout = False
        self.model.esm.embeddings.token_dropout = False

        # embedding dict
        self.vocab_dict = self.tokenizer.get_vocab()
        sorted_vocab = [
            item[0] 
            for item in sorted(self.vocab_dict.items(), key=lambda x: x[1])
        ]
        token_ids = [self.vocab_dict[t] for t in sorted_vocab]
        token_ids_tensor = torch.tensor(token_ids, device=self.device)
        
        # extract esm embedding layer - required for embedding relaxed sequences
        embedding_layer = self.model.esm.embeddings.word_embeddings
        with torch.no_grad():
            all_embeddings = embedding_layer(token_ids_tensor)
        
        # convert to dict
        self.dense_vector_dict = {
            token: emb.detach().cpu().numpy()
            for token, emb in zip(sorted_vocab, all_embeddings)
        }
        
        # build embedding matrix for the vocab
        with torch.no_grad():
            self.embeddings_matrix = torch.stack(
                [all_embeddings[self.vocab_dict[aa]] for aa in self.alphabet],
                dim=0
            ).float().to(self.device)

    def _freeze_esm(self):
        """
        Freeze all parameters of the ESM model to prevent updates 
        during training.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def forward_pass(self,
                     relaxed_seqs: torch.Tensor,
                     attention_mask: torch.Tensor = None,
                     ) -> torch.Tensor:
        """
        Forward pass with relaxed amino acids at each position.

        Parameters
        ----------
        relaxed_seqs : torch.Tensor
            Array of shape [L, alphabet_size] or [B, L, alphabet_size] 
            containing distribution of AAs at each position.

        Returns
        -------
            torch.Tensor
                Final hidden layer from the model.
        """
        # ensure input has batch dimension
        if relaxed_seqs.dim() == 2:
            # add batch dimension if missing [L, A] -> [1, L, A]
            relaxed_seqs = relaxed_seqs.unsqueeze(0)
            
        # weighted embedding per position
        input_embeddings = torch.einsum(
            'bji,ik->bjk',
            relaxed_seqs,
            self.embeddings_matrix
        )

        # pass through esm
        out = self.model(
            inputs_embeds=input_embeddings,
            output_hidden_states=True,
            attention_mask=None if attention_mask is None else attention_mask,
            return_dict=True
        )

        return out
    
    def batch_iterator(self,
                   sequences: Union[np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]],
                   batch_size: Optional[int] = None):
        """
        Batch iterator for sequences of strings, numpy arrays, or torch tensors.
        Handles sorting for efficiency and yields batches of padded sequences, attention masks, original lengths, and batch indices.
        """

        if not isinstance(sequences, list):
            sequences = [sequences]

        _batch_size = batch_size if batch_size is not None else self.batch_size

        ohe_cls = torch.nn.functional.one_hot(torch.tensor(self.vocab_dict['<cls>']), num_classes=len(self.alphabet)).float()
        ohe_eos = torch.nn.functional.one_hot(torch.tensor(self.vocab_dict['<eos>']), num_classes=len(self.alphabet)).float()
        ohe_pad = torch.nn.functional.one_hot(torch.tensor(self.vocab_dict['<pad>']), num_classes=len(self.alphabet)).float()

        def get_len(seq):
            return len(seq) if isinstance(seq, str) else seq.shape[0]

        sorted_with_indices = sorted(enumerate(sequences), key=lambda x: get_len(x[1]), reverse=True)

        for i in range(0, len(sorted_with_indices), _batch_size):
            batch_with_indices = sorted_with_indices[i:i + _batch_size]
            batch_indices, batch_data = zip(*batch_with_indices)

            original_lengths = []
            processed_tensors = []

            for seq in batch_data:
                if isinstance(seq, str):
                    tensor = torch.from_numpy(self.get_seq_ohe(seq))
                else:
                    tensor = torch.as_tensor(seq, dtype=torch.float32)

                if tensor.shape[1] != len(self.alphabet):
                    tensor = torch.nn.functional.pad(tensor, (len(self.alphabet)-tensor.shape[1], 0), 'constant', 0)

                is_wrapped = (tensor.shape[0] > 1 and
                            torch.argmax(tensor[0]) == self.vocab_dict['<cls>'] and
                            torch.argmax(tensor[-1]) == self.vocab_dict['<eos>'])

                if is_wrapped:
                    original_lengths.append(tensor.shape[0] - 2)
                    processed_tensors.append(tensor)
                else:
                    original_lengths.append(tensor.shape[0])
                    processed_tensors.append(torch.cat([ohe_cls.unsqueeze(0), tensor, ohe_eos.unsqueeze(0)], dim=0))

            max_len = max(t.shape[0] for t in processed_tensors)

            padded_batch = ohe_pad.repeat(len(batch_data), max_len, 1)
            for j, t in enumerate(processed_tensors):
                padded_batch[j, :t.shape[0], :] = t

            actual_lengths = torch.tensor([t.shape[0] for t in processed_tensors])
            attention_mask = torch.arange(max_len)[None, :] < actual_lengths[:, None]

            yield padded_batch.to(self.device), attention_mask.to(self.device), original_lengths, batch_indices
    
    def get_seq_ohe(self, sequence: str) -> np.ndarray:
        """
        Get the OHE for a protein sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence string to get the OHE of.

        Returns
        -------
        np.ndarray
            OHE of sequence.
        """
        seq_len = len(sequence)
        alphabet_size = len(self.alphabet)
        encoding = np.zeros((seq_len, alphabet_size), dtype=np.float32)
        
        for i, aa in enumerate(sequence):
            if aa in self.alphabet:
                idx = self.alphabet.index(aa)
                encoding[i, idx] = 1.0
        
        return encoding
    
    def embed_relaxed_seqs(self,
                        sequences: Union[np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]],
                        batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Embeds sequences of strings, numpy arrays, or torch tensors.
        Handles sorting for efficiency and reorders the output to match the input order.

        Parameters
        ----------
        sequences : Union[np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]]
            A single sequence or a list of sequences to embed.

        batch_size : int, optional
            The batch size for processing. Defaults to the one set in the constructor.

        Returns
        -------
        np.ndarray
            Extracted features [number_of_sequences, embedding_dimension].
        """
        sequences_as_list = sequences if isinstance(sequences, list) else [sequences]
        features = [None] * len(sequences_as_list)

        iterator = list(self.batch_iterator(sequences, batch_size))

        for seq_batch, mask_batch, original_lengths, batch_indices in tqdm(iterator, desc="Embedding"):
            with torch.no_grad():
                hidden_states = self.forward_pass(seq_batch, mask_batch)
                hidden_states = hidden_states.hidden_states[-1]

            for j, original_idx in enumerate(batch_indices):
                length = original_lengths[j]
                embedding = hidden_states[j, 1:length + 1].cpu()
                embedding = embedding.mean(axis=0)
                features[original_idx] = embedding
        
        return np.array(features)
    
    def lm_output_probabilities(self,
                        sequences: Union[np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]],
                        batch_size: Optional[int] = None) -> List[np.ndarray]:
        """
        Embeds sequences of strings, numpy arrays, or torch tensors.
        Handles sorting for efficiency and reorders the output to match the input order.

        Parameters
        ----------
        sequences : Union[np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]]
            A single sequence or a list of sequences to embed.

        batch_size : int, optional
            The batch size for processing. Defaults to the one set in the constructor.

        Returns
        -------
        List[np.ndarray]
            Extracted probabilities list of arrays, each of shape [sequence_length, alphabet_size].
        """
        sequences_as_list = sequences if isinstance(sequences, list) else [sequences]
        output_probabilities = [None] * len(sequences_as_list)

        iterator = list(self.batch_iterator(sequences, batch_size))

        for seq_batch, mask_batch, original_lengths, batch_indices in tqdm(iterator, desc="Embedding"):
            with torch.no_grad():
                out = self.forward_pass(seq_batch, mask_batch)
                probabilities = out.logits.softmax(dim=-1)

            for j, original_idx in enumerate(batch_indices):
                length = original_lengths[j]
                probability = probabilities[j, 1:length + 1].cpu().numpy()
                # extract only vocab positions
                probability = probability[:, [self.vocab_dict[aa] for aa in self.alphabet]]
                # ensure the output is float32
                output_probabilities[original_idx] = probability.astype(np.float32)
        
        return output_probabilities
    
    def embed_sequences(self,
                        sequences: List[str],
                        batch_size: Optional[int] = None) -> np.ndarray:
            """
            Alias for embed_relaxed_seqs for embedding string sequences.
            """
            return self.embed_relaxed_seqs(sequences, batch_size)
        
    def extract_features(self,
                       sequences: List[str],
                       batch_size: int = 32) -> np.ndarray:
        """
        Alias for embed_relaxed_seqs for consistency.
        """
        return self.extract_features(sequences, batch_size)

    def save_embeddings(self, 
                        embeddings: np.ndarray, 
                        embedding_path: Path) -> None:
        """
        Save embeddings.
        
        Parameters:
        ----------
        embeddings : np.ndarray
            Embeddings to save.
            
        embedding_path : Path
            Path to load the embeddings from
        """
        np.save(embedding_path, embeddings)

    def load_embeddings(self,
                        embedding_path: Path) -> np.ndarray:
        """
        Load saved embeddings.
        
        Parameters:
        ----------
        embedding_path : Path
            Path to load the embeddings from
            
        Returns:
        --------
        np.ndarray
            Loaded embeddings
        """
        return np.load(embedding_path)