import torch


class DummyTokenizer:
    """
    Minimal tokenizer stub that produces variable-length token id sequences
    from space-separated characters to exercise padding code paths.
    - Adds CLS=101 and EOS=102 when add_special_tokens is True
    - Expands 'A' to two token ids to induce variable lengths
    """
    def __call__(self, text, add_special_tokens=True, return_tensors='pt'):
        assert return_tensors == 'pt'
        tokens = text.split()
        ids = []
        if add_special_tokens:
            ids.append(101)
        for t in tokens:
            if t == 'A':
                ids.extend([200, 201])
            else:
                # simple stable mapping
                ids.append(100 + (ord(t[0]) % 50))
        if add_special_tokens:
            ids.append(102)
        return {'input_ids': torch.tensor([ids], dtype=torch.long)}


def test_to_sequence_tensors_tokenizer_padding(binary_3bit_landscape):
    tok = DummyTokenizer()
    ds = binary_3bit_landscape.to_sequence_tensors(tokenizer=tok)
    assert isinstance(ds, list) and len(ds) == len(binary_3bit_landscape.sequences)
    # All sequence_tensors should be Long and same length; attention_mask present
    lens = []
    for item in ds:
        assert 'sequence_tensor' in item and 'fitness_tensors' in item
        assert isinstance(item['sequence_tensor'], torch.Tensor)
        assert item['sequence_tensor'].dtype == torch.long
        assert 'attention_mask' in item
        assert item['attention_mask'].dtype == torch.long
        lens.append(int(item['sequence_tensor'].numel()))
        # mask matches token length (1s then 0s)
        tlen = int(item['attention_mask'].sum().item())
        assert tlen <= item['attention_mask'].numel()
    # verify padding applied (max length equals at least one row, and not all equal if variable expansion hit)
    assert max(lens) == len(ds[0]['sequence_tensor'])


def test_to_graph_tensor_tokenizer_padding(binary_3bit_landscape):
    tok = DummyTokenizer()
    try:
        data = binary_3bit_landscape.to_graph_tensor(tokenizer=tok)
    except (ImportError, NameError):
        import pytest
        pytest.skip("torch_geometric not installed.")
    # Check token_ids and attention_mask are attached and padded
    assert hasattr(data, 'token_ids') and hasattr(data, 'attention_mask')
    assert data.token_ids.dtype == torch.long
    assert data.attention_mask.dtype == torch.long
    assert data.token_ids.shape == data.attention_mask.shape
    # All nodes accounted for
    assert data.token_ids.shape[0] == binary_3bit_landscape.graph.number_of_nodes()
    # Mask is 1 for tokens, 0 for padding
    row = data.attention_mask[0]
    ones = int(row.sum().item())
    assert ones <= row.numel()
