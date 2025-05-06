import numpy as np
import torch

from numpy.typing import ArrayLike
from pathlib import Path
from tqdm import tqdm
from transformers import EsmForMaskedLM, AutoTokenizer
from typing import List, Optional


class ESMEmbedder:

    def __init__(self,
                 model_name: str,
                 device: Optional[str] = None,
                 alphabet: str = 'ACDEFGHIKLMNPQRSTVWY',
                 exclude_cls_eos: bool = True) -> None:
        '''
        Initialise PLM embedder class by initialising the provided
        model.

        Parameters:
        -----------
        model_name : str
            HuggingFace PLM model name.

        device : str, default = None
            Device to use, if None will autoselect.

        alphabet : str
            Amino acids expected for embedding.

        exclude_cls_eos : bool
            Whether the forward pass will add <cls> and <eos> tokens to
            the embeddings.
        '''
        # device management
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # load model
        self.alphabet = [aa for aa in alphabet]
        self.model_name = model_name
        self._load_model()

        self.exclude_cls_eos = exclude_cls_eos

    def _load_model(self):
        '''
        Load tokenizer and model given name. Tokenizer is used to
        construct an embedding matrix for use on relaxed sequences.
        '''
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = EsmForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # set up model for inferece
        self.model.esm.token_dropout = False
        self.model.config.token_dropout = False
        self.model.esm.embeddings.token_dropout = False

        # embedding dict
        vocab_dict = self.tokenizer.get_vocab()
        sorted_vocab = [
            item[0] 
            for item in sorted(vocab_dict.items(), key=lambda x: x[1])
        ]
        token_ids = [vocab_dict[t] for t in sorted_vocab]
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
                [all_embeddings[vocab_dict[aa]] for aa in self.alphabet],
                dim=0
            ).float().to(self.device)

    def _freeze_esm(self):
        '''
        Freeze all parameters of the ESM model to prevent updates 
        during training.
        '''
        for param in self.model.parameters():
            param.requires_grad = False

    def forward_pass(self, relaxed_seqs: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass with relaxed amino acids at each position.

        Parameters:
        -----------
        relaxed_seqs : torch.Tensor
            Array of shape [L * alphabet_size] or [B * L * alphabet_size] 
            containing distribution of AAs at each position.

        Returns:
        --------
            torch.Tensor
                Final hidden layer from the model.
        '''
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

        # prepend <cls> and append <eos>
        if self.exclude_cls_eos:
            # prepare special token vectors
            cls_vec = torch.from_numpy(
                self.dense_vector_dict["<cls>"]
            ).to(self.device).float().unsqueeze(0).unsqueeze(0)
            eos_vec = torch.from_numpy(
                self.dense_vector_dict["<eos>"]
            ).to(self.device).float().unsqueeze(0).unsqueeze(0)
            
            # repeat vectors for batch size if needed
            batch_size = input_embeddings.shape[0]
            if batch_size > 1:
                cls_vec = cls_vec.repeat(batch_size, 1, 1)
                eos_vec = eos_vec.repeat(batch_size, 1, 1)
                
            # concatenate along sequence dimension
            input_embeddings = torch.cat([cls_vec, input_embeddings, eos_vec], dim=1)

        # pass through esm
        out = self.model(
            inputs_embeds=input_embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        return out.hidden_states[-1]
    
    def get_seq_ohe(self, sequence: str) -> np.ndarray:
        '''
        Get the OHE for a protein sequence.

        Parameters:
        -----------
        sequence : str
            Protein sequence string to get the OHE of.

        Returns:
        --------
        np.ndarray
            OHE of sequence.
        '''
        seq_len = len(sequence)
        alphabet_size = len(self.alphabet)
        encoding = np.zeros((seq_len, alphabet_size), dtype=np.float32)
        
        for i, aa in enumerate(sequence):
            if aa in self.alphabet:
                idx = self.alphabet.index(aa)
                encoding[i, idx] = 1.0
        
        return encoding
    
    def embed_relaxed_seqs(self,
                           relaxed_seqs: ArrayLike,
                           batch_size: int = 32) -> np.ndarray:
        '''
        Embed relaxed (or one-hot encoded) sequences directly.

        Parameters:
        -----------
        relaxed_seqs : List[torch.Tensor]
            List of relaxed sequence tensors.

        batch_size : int
            Batch size for processing.

        Returns:
        --------
        np.ndarray
            Extracted features.
        '''
        if isinstance(relaxed_seqs, np.ndarray):
            relaxed_seqs = torch.from_numpy(relaxed_seqs).float()

        features = []

        # process relaxed sequences
        for i in tqdm(range(0, len(relaxed_seqs), batch_size)):
            batch_seqs = relaxed_seqs[i : i + batch_size]
            batch_features = []
            for seq_tensor in batch_seqs:
                with torch.no_grad():
                    # ensure tensor is on correct device
                    if not seq_tensor.device == self.device:
                        seq_tensor = seq_tensor.to(self.device)
                    # compute the hidden state using the forward pass
                    hidden_state = self.forward_pass(seq_tensor)
                    hidden_state.squeeze_(0)
                    batch_features.append(hidden_state.cpu().numpy())
            features.extend(batch_features)
        
        return np.array(features).astype(np.float32)
    
    def extract_features(self,
                         sequences: List[str],
                         batch_size: int = 32) -> np.ndarray:
        '''
        Extract features from protein sequences.

        Parameters:
        -----------
        sequences : List[str]
            List of protein sequences.

        batch_size : int
            Batch size for processing.

        Returns:
        --------
        np.ndarray
            Extracted features.
        '''
        features = []

        # process sequences
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i : i + batch_size]
            batch_features = []
            for sequence in batch_sequences:
                with torch.no_grad():
                    # convert to one-hot encoding
                    sequence_ohe = torch.from_numpy(
                        self.get_seq_ohe(sequence)
                    ).to(self.device)
                    # compute the hidden state using the forward pass
                    sequence_hidden_state = self.forward_pass(sequence_ohe)
                    sequence_hidden_state.squeeze_(0)
                    batch_features.append(sequence_hidden_state.cpu().numpy())
            features.extend(batch_features)
        
        return np.array(features).astype(np.float32)
        
    def embed_sequences(self,
                       sequences: List[str],
                       batch_size: int = 32) -> np.ndarray:
        '''
        Alias for extract_features for consistency with embed_relaxed_seqs.
        
        Parameters:
        -----------
        sequences : List[str]
            List of protein sequences.
            
        batch_size : int
            Batch size for processing.
            
        Returns:
        --------
        np.ndarray
            Extracted features.
        '''
        return self.extract_features(sequences, batch_size)

    def save_embeddings(self, 
                        embeddings: np.ndarray, 
                        embedding_path: Path) -> None:
        '''
        Save embeddings.
        
        Parameters:
        -----------
        embeddings : np.ndarray
            Embeddings to save.
            
        embedding_path : Path
            Path to load the embeddings from
        '''
        # save embeddings
        np.save(embedding_path, embeddings)

    def load_embeddings(self, embedding_path: Path) -> np.ndarray:
        '''
        Load saved embeddings.
        
        Parameters:
        ----------
        embedding_path : Path
            Path to load the embeddings from
            
        Returns:
        --------
        np.ndarray
            Loaded embeddings
        '''
        return np.load(embedding_path)