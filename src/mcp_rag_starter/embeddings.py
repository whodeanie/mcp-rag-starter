"""Embedding generation using sentence transformers."""

from typing import Optional

from sentence_transformers import SentenceTransformer

from mcp_rag_starter.config import EmbeddingsConfig


class EmbeddingEngine:
    """Wrapper around sentence-transformers for embedding generation."""

    def __init__(self, config: Optional[EmbeddingsConfig] = None):
        """Initialize embedding engine.

        Args:
            config: Embeddings configuration.
        """
        self.config = config or EmbeddingsConfig()
        self.model = SentenceTransformer(
            self.config.model_name,
            cache_folder=self.config.cache_dir,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if not text:
            return []

        embeddings = self.model.encode([text], convert_to_numpy=True)
        return embeddings[0].tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Dimension of embeddings produced by this model.
        """
        return self.model.get_sentence_embedding_dimension()
