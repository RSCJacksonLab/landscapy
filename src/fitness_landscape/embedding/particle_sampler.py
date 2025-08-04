from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import networkx as nx
from sklearn.manifold import TSNE
import torch.nn as nn
from softalign import align_soft_sequences
import umap
from .soft_embedding import ESMEmbedder
from .._const import PROT_20

def pad_sequences(sequences: List[torch.Tensor],
                  pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    
    """
    num_classes = sequences[0].shape[-1]
    pad_ohe = torch.nn.functional.one_hot(
        torch.tensor(pad_idx, device=sequences[0].device),
        num_classes=num_classes
    ).float()
    
    padded_batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(s) for s in sequences], device=padded_batch.device)
    max_len = padded_batch.size(1)
    mask = torch.arange(max_len, device=padded_batch.device) >= lengths[:, None]
    
    if mask.any():
        padded_batch[mask] = pad_ohe
        
    return padded_batch, mask

def remove_padding(sequences: torch.Tensor,
                   pad_idx: int) -> List[torch.Tensor]:
    """
    
    """
    pad_ohe = torch.nn.functional.one_hot(
        torch.tensor(pad_idx, device=sequences.device),
        num_classes=sequences.shape[-1]
    ).float()
    is_pad = torch.all(sequences == pad_ohe, dim=-1)
    lengths = (~is_pad).sum(dim=1)
    return [sequences[i, :lengths[i]] for i in range(len(sequences))]

def parse_to_pssm_tensor(file_path: Path,
                         alphabet: str = PROT_20+['-']) -> torch.Tensor:
    """
    
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    hmm_start_idx = next(i for i, line in enumerate(lines) if line.startswith('HMM'))

    header_str = lines[hmm_start_idx].strip()
    data_start_line_idx = hmm_start_idx + 1
    while 'm->m' not in lines[data_start_line_idx]:
        header_str += " " + lines[data_start_line_idx].strip()
        data_start_line_idx += 1
    
    file_amino_acids = header_str.split()[1:]
    num_file_aas = len(file_amino_acids)

    all_scores = []
    for line in lines[data_start_line_idx:]:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            numbers = [p for p in parts[1:] if re.match(r'^-?\d+\.\d+$', p)]
            all_scores.extend(numbers)
    
    data_rows = [
        [float(s) for s in all_scores[i:i + num_file_aas]] 
        for i in range(0, len(all_scores), num_file_aas)
    ]
    
    scores_tensor = torch.tensor(data_rows, dtype=torch.float32)

    exp_scores = torch.exp(-scores_tensor)
    initial_pssm = exp_scores / exp_scores.sum(dim=1, keepdim=True)


    source_map = {aa: i for i, aa in enumerate(file_amino_acids)}

    num_positions = initial_pssm.shape[0]
    final_pssm = torch.zeros(num_positions, len(alphabet), dtype=torch.float32)

    for target_idx, char in enumerate(alphabet):
        if char in source_map:
            source_idx = source_map[char]
            final_pssm[:, target_idx] = initial_pssm[:, source_idx]
            
    return final_pssm

def score_entropy(tensor: torch.Tensor,
                  padding_mask: torch.Tensor,
                  epsilon: float = 1e-9) -> torch.Tensor:
    """
    
    """
    positional_entropy = -torch.sum(tensor * torch.log2(tensor.clamp(min=epsilon)), dim=-1)
    positional_entropy[padding_mask] = 0.0
    total_entropy_per_sequence = positional_entropy.sum(dim=-1)
    sequence_lengths = (~padding_mask).sum(dim=-1)
    average_entropy = total_entropy_per_sequence / sequence_lengths.clamp(min=1)
    
    return average_entropy

def compare_entropy(tensor1: torch.Tensor,
                    tensor2: torch.Tensor,
                    padding_mask1: torch.Tensor,
                    padding_mask2: torch.Tensor) -> torch.Tensor:
    """
    
    """
    entropy1 = score_entropy(tensor1, padding_mask1)
    entropy2 = score_entropy(tensor2, padding_mask2)
    return entropy1 - entropy2

class Sampler(ABC):
    """
    
    """
    def __init__(self,
                 temperature: float = 1.0) -> None:
        
        if not (temperature > 0):
            raise ValueError("Temperature must be a positive number.")
        self.temperature = temperature

    @abstractmethod
    def _core_sample(self,
                    probabilities: torch.Tensor) -> torch.Tensor:
        """
        
        """
        raise NotImplementedError("Subclasses must implement the '_core_sample' method.")

    def sample(self,
               probabilities: torch.Tensor,
               pad_idx: int) -> torch.Tensor:
        """
        
        """
        original_shape = probabilities.shape
        num_tokens = original_shape[-1]
        device = probabilities.device
        
        padding_mask = torch.argmax(probabilities, dim=-1) == pad_idx
        flat_padding_mask = padding_mask.view(-1)

        probs_flat = probabilities.view(-1, num_tokens)
        unpadded_probs = probs_flat[~flat_padding_mask].clone()

        if self.temperature != 1.0:
            unpadded_probs.pow_(1.0 / self.temperature)

        processed_unpadded = self._core_sample(unpadded_probs)

        pad_vector = torch.nn.functional.one_hot(
            torch.tensor(pad_idx, device=device), num_classes=num_tokens
        ).float()
        
        output_flat = pad_vector.expand(probs_flat.shape[0], -1).clone()
        output_flat[~flat_padding_mask] = processed_unpadded
        return output_flat.view(original_shape)

class TopPSampler(Sampler):
    """
    
    """
    def __init__(self,
                 temperature: float = 1.0,
                 top_p: float = 0.9) -> None:
        
        super().__init__(temperature)
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0.")
        self.top_p = top_p

    def _core_sample(self,
                     probabilities: torch.Tensor) -> torch.Tensor:
        """
        
        """
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cumulative_probs - sorted_probs > self.top_p
        sorted_probs[nucleus] = 0.0

        sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1)

        sampled_indices = torch.gather(sorted_indices, -1, sampled_sorted_indices)

        return nn.functional.one_hot(
            sampled_indices.squeeze(-1), num_classes=probabilities.shape[-1]
        ).float()
        
class TrajectoryGraph:
    """
    
    """
    def __init__(self):
        self.G = nx.DiGraph()
        self.node_counter = 0

    def add_node(self,
                 **attrs: Any) -> int:
        """
        
        """
        node_id = self.node_counter
        self.G.add_node(node_id, **attrs)
        self.node_counter += 1
        return node_id

    def add_edge(self,
                 parent_id: int,
                 child_id: int,
                 **attrs: Any):
        """
        
        """
        self.G.add_edge(parent_id, child_id, **attrs)

    def get_node_data(self,
                      node_id: int) -> Dict[str, Any]:
        """
        
        """
        return self.G.nodes[node_id]

    def get_tips_with_attribute(self,
                                attr_name: str) -> List[Tuple[int, float]]:
        """
        
        """
        all_tips = [n for n, d in self.G.out_degree() if d >= 0]
        return [(tip, self.G.nodes[tip][attr_name]) for tip in all_tips]

class ParentSelector:
    """
    
    """
    def __init__(self,
                 max_state_size: int):
        
        self.max_state_size = max_state_size

    def select(self,
               candidates_with_weights: List[Tuple[int, float]]) -> List[int]:
        """

        """

        candidates, weights = zip(*candidates_with_weights)
        num_to_select = min(self.max_state_size, len(candidates))

        scores = -torch.tensor(weights, dtype=torch.float)
        probabilities = torch.softmax(scores, dim=0)

        selected_indices = torch.multinomial(probabilities, num_to_select, replacement=False)
        return [candidates[i] for i in selected_indices]

class SequenceGenerator:
    """
    
    """
    def __init__(self,
                 embedder: ESMEmbedder,
                 sampler: Sampler,
                 batch_size: int,
                 alphabet: List = PROT_20 + ['-']):
        self.embedder = embedder
        self.sampler = sampler
        self.batch_size = batch_size
        self.pad_idx = self.embedder.alphabet.index('-')
        self.alphabet = alphabet

    def process_sequences(self,
                          sequences: List[str],
                          _emb_arr_key: str = "emb_arr") -> List[Dict[str, Any]]:
        """
        
        """
        representations = self.embedder.embed_relaxed_seqs(sequences, self.batch_size)
        probabilities = self.embedder.lm_output_probabilities(sequences, self.batch_size)
        probabilities = [torch.tensor(prob, dtype=torch.float32) for prob in probabilities]
        padded_probs, pad_mask = pad_sequences(probabilities, self.pad_idx)
        entropies = score_entropy(padded_probs, pad_mask)

        #TODO: probably a cleaner way to do this...
        processed_data = []
        processed_sequences = []
        for seq, rep, prob, entropy in zip(sequences, representations, probabilities, entropies):
            
            #Drop gaps
            sequence_str = get_ohe_seq(seq).replace("-","")
            if not any(char not in self.alphabet for char in sequence_str):
                # Create a BaseNumpySequence object.
                sequence_obj = BaseNumpySequence(list(sequence_str), alphabet=self.alphabet.remove("-"))
                
                # Collect sequence for validation.
                processed_sequences.append(sequence_obj)
                processed_data.append({
                    'sequence': sequence_obj, # Store the object instead of the tensor
                    f'{_emb_arr_key}': rep,
                    'lm_output': prob,
                    'lm_entropy': entropy.item()
                })
        return processed_data, processed_sequences

    def generate_children(self,
                          parent_probs: List[torch.Tensor],
                          n_samples: int) -> List[Dict[str, Any]]:
        """
        
        """
        padded_probs, pad_mask = pad_sequences(parent_probs, self.pad_idx)
        expanded_probs = padded_probs.repeat_interleave(n_samples, dim=0)
        
        children_padded = self.sampler.sample(expanded_probs, self.pad_idx)
        children_unpadded = remove_padding(children_padded, self.pad_idx)
        
        return self.process_sequences(children_unpadded)[0]
    
class SequenceSpaceAttractor:
    """
    
    """
    def __init__(self,
                 embedder: ESMEmbedder,
                 hmm_file: Path = None,
                 embedding_attribute: str = 'emb_array'):
        self.embedder = embedder
        self.embedding_attribute = embedding_attribute
        self.hmm_pssm = parse_to_pssm_tensor(hmm_file)
        self.attractor = torch.tensor(self.embedder.embed_relaxed_seqs(self.hmm_pssm).squeeze())
        self.embedder = None

    def compare(self,
                G: nx.DiGraph,
                initial: bool = False) -> float:
        """
        
        """
        nodes_with_repr = [n for n, d in G.nodes(data=True) if 'attractor_sim' not in d]
        representations = np.array([G.nodes[n][self.embedding_attribute] for n in nodes_with_repr])
        similarities = torch.cosine_similarity(
            torch.tensor(representations, dtype=torch.float32),
            self.attractor.unsqueeze(0).expand(len(nodes_with_repr), -1)
        )
        if initial:
            self.seed_mean = similarities.mean().item()
            self.seed_std = similarities.std().item()

        for node, sim in zip(nodes_with_repr, similarities):
            G.nodes[node]['attractor_sim'] = sim.item()
        return G

    
class EvolutionParticleSampler:
    """
    
    """
    def __init__(self,
                 generator: SequenceGenerator,
                 selector: ParentSelector,
                 n_samples: int,
                 traj_length: int,
                 attractor = None):
        
        self.generator = generator
        self.selector = selector
        self.n_samples = n_samples
        self.traj_length = traj_length
        self.current_state_nodes = []
        self.G = nx.DiGraph()
        self.node_counter = 0
        if attractor is not None:
            self.attractor = attractor

    def initialize(self,
                   seed_sequences: List[str]):
        """
        
        """
        processed_seeds, _ = self.generator.process_sequences(seed_sequences)
        for seed_data in processed_seeds:
            node_id = self.node_counter
            self.G.add_node(node_id, **seed_data)
            self.node_counter += 1
            self.current_state_nodes.append(node_id)
        if hasattr(self, 'attractor'):
            self.G = self.attractor.compare(self.G, initial=True)

    def _step(self) -> bool:
        """
        
        """
        if hasattr(self, 'attractor'):
            candidates = [(node, self.G.nodes[node]['lm_entropy']) for node in self.G.nodes if self.G.nodes[node]['attractor_sim'] < self.attractor.seed_mean * 2]
        else:
            candidates = [(node, self.G.nodes[node]['lm_entropy']) for node in self.G.nodes]
        
        parent_nodes = self.selector.select(candidates)
        if not parent_nodes:
            return False
        
        self.current_state_nodes = parent_nodes
        parent_probs = [self.G.nodes[p]['lm_output'] for p in parent_nodes]
        children_data = self.generator.generate_children(parent_probs, self.n_samples)
        
        expanded_parent_nodes = [p for p in parent_nodes for _ in range(self.n_samples)]
        
        for parent_id, child_data in zip(expanded_parent_nodes, children_data):
            child_id = self.node_counter
            self.G.add_node(child_id, **child_data)
            self.node_counter += 1
            
            parent_entropy = self.G.nodes[parent_id]['lm_entropy']
            child_entropy = child_data['lm_entropy']
            self.G.add_edge(parent_id, child_id, delta_entropy=(parent_entropy - child_entropy))

        if hasattr(self, 'attractor'):
            self.G = self.attractor.compare(self.G)
        
        return True

    def run(self):
        """
        
        """
        for i in range(self.traj_length):
            if not self._step():
                break
        print("Evolution finished.")

#TODO: refactor to vis module.
def _plot_graph(G: nx.DiGraph,
               attractor: SequenceSpaceAttractor = None,

               output_file: Path = None):

    """
    
    """
    nodes_with_repr = [n for n, d in G.nodes(data=True) if 'representation' in d]
    if len(nodes_with_repr) < 2:
        print("Not enough nodes with representations to plot.")
        return

    representations = np.array([G.nodes[n]['representation'] for n in nodes_with_repr])
    all_reps = list(representations)

    combined_reps = np.array(all_reps)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced_combined = reducer.fit_transform(combined_reps)
    
    reduced_real = reduced_combined[:len(representations)]
    pos = {node: reduced_real[i] for i, node in enumerate(nodes_with_repr)}

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 14))
    ax = plt.gca()


    valid_edges = [edge for edge in G.edges() if edge[0] in pos and edge[1] in pos]
    nx.draw_networkx_edges(G, pos, edgelist=valid_edges, width=1.2, alpha=1.0, 
                           edge_color=[G.edges[edge].get('delta_entropy', 0) for edge in valid_edges],
                           edge_cmap=plt.cm.viridis, edge_vmin=-1, edge_vmax=1, arrows= True)
    
    threshold = 2 * attractor.seed_mean if attractor else float('inf')
    inliers = {n: d for n, d in G.nodes(data=True) if n in pos and d.get('attractor_sim', 0) < threshold}
    outliers = {n: d for n, d in G.nodes(data=True) if n in pos and d.get('attractor_sim', 0) >= threshold}
    
    if inliers:
        inlier_nodes, inlier_pos = list(inliers.keys()), {k: pos[k] for k in inliers}
        node_colors = [d['lm_entropy'] for d in inliers.values()]
        nodes_plot = nx.draw_networkx_nodes(G, inlier_pos, nodelist=inlier_nodes, node_size=50, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9)
        plt.colorbar(nodes_plot, ax=ax, shrink=0.8, label='Node LM Entropy (Lower is better)')

    if outliers:
        outlier_nodes, outlier_pos = list(outliers.keys()), {k: pos[k] for k in outliers}
        nx.draw_networkx_nodes(G, outlier_pos, nodelist=outlier_nodes, node_size=90, node_color='red', alpha=0.8)

    plt.title("UMAP Visualization of Evolutionary Trajectory", fontsize=18)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        plt.show()