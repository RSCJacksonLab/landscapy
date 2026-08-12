"""Opt-in integration coverage for the pinned smallest public ESM model."""

from __future__ import annotations

import os

import numpy as np
import pytest

from fitness_landscape.embedding.esm import DEFAULT_ESM_MODEL, ESMEmbedder


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("LANDSCAPY_RUN_ESM_INTEGRATION") != "1",
    reason="set LANDSCAPY_RUN_ESM_INTEGRATION=1 to load the pinned ESM model",
)
def test_pinned_small_esm_model_embedding_contract():
    embedder = ESMEmbedder(
        model_name=DEFAULT_ESM_MODEL,
        device="cpu",
        alphabet=["A", "C", "-"],
        batch_size=2,
    )

    embeddings = embedder.embed_sequences(["AC", "A-"], batch_size=2)

    assert embeddings.shape == (2, 320)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert not np.array_equal(embeddings[0], embeddings[1])
