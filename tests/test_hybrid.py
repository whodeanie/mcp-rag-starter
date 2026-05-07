"""Tests for hybrid retrieval."""

import pytest

from mcp_rag_starter.bm25 import BM25Retriever
from mcp_rag_starter.config import VectorStoreConfig
from mcp_rag_starter.hybrid import HybridRetriever
from mcp_rag_starter.vector_store import FAISSVectorStore


def test_hybrid_search():
    """Test hybrid search with mock data."""
    # Create mock chunks
    chunks = [
        {"content": "machine learning is great", "source": "doc1.pdf", "start_idx": 0, "end_idx": 1},
        {"content": "deep learning networks", "source": "doc1.pdf", "start_idx": 2, "end_idx": 3},
        {"content": "neural networks process data", "source": "doc1.pdf", "start_idx": 4, "end_idx": 5},
    ]

    # Initialize retrievers
    vector_store = FAISSVectorStore(VectorStoreConfig())
    bm25 = BM25Retriever()

    hybrid = HybridRetriever(vector_store, bm25)

    # Index chunks
    embeddings = [[0.1, 0.2, 0.3, 0.4]] * 3  # Mock embeddings
    vector_store.add(embeddings, chunks)
    bm25.index(chunks)

    # Search
    query_embedding = [0.15, 0.25, 0.35, 0.45]
    results = hybrid.search("machine learning", query_embedding, k=2)

    assert len(results) <= 2
    assert all("metadata" in r for r in results)
    assert all("score" in r for r in results)


def test_hybrid_no_results():
    """Test hybrid search with empty store."""
    vector_store = FAISSVectorStore()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(vector_store, bm25)

    results = hybrid.search("query", [0.1, 0.2], k=5)
    assert len(results) == 0
