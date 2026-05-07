"""FAISS vector store for semantic search."""

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from mcp_rag_starter.config import VectorStoreConfig


class FAISSVectorStore:
    """FAISS-based vector store with persistence."""

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize vector store.

        Args:
            config: Vector store configuration.
        """
        self.config = config or VectorStoreConfig()
        self.persist_dir = Path(self.config.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.persist_dir / "index.faiss"
        self.metadata_path = self.persist_dir / "metadata.json"

        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: list[dict] = []
        self.vector_count = 0

        self._load()

    def add(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        """Add vectors and metadata to store.

        Args:
            vectors: List of embedding vectors.
            metadata: List of metadata dicts corresponding to vectors.
        """
        if not vectors or not metadata:
            return

        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have same length")

        vectors_array = np.array(vectors, dtype=np.float32)

        if self.index is None:
            # Initialize index with dimension from first vector
            dim = vectors_array.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(vectors_array)
        self.metadata.extend(metadata)
        self.vector_count = self.index.ntotal

        self._persist()

    def search(self, query_vector: list[float], k: int = 5) -> list[dict]:
        """Search for nearest neighbors.

        Args:
            query_vector: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of dicts with 'metadata' and 'distance' keys, sorted by distance.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(dist),
                })

        return results

    def clear(self) -> None:
        """Clear all vectors and metadata."""
        self.index = None
        self.metadata = []
        self.vector_count = 0
        self._persist()

    def _persist(self) -> None:
        """Save index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load(self) -> None:
        """Load index and metadata from disk if they exist."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.vector_count = self.index.ntotal

        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self.metadata = json.load(f)
