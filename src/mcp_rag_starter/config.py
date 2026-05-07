"""Configuration loader and validation using Pydantic."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings model."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model name",
    )
    cache_dir: str = Field(default=".cache", description="Cache directory for models")


class VectorStoreConfig(BaseModel):
    """Configuration for FAISS vector store."""

    persist_dir: str = Field(default=".vector_store", description="Directory for persisted index")
    dim: int = Field(default=384, description="Embedding dimension (must match model output)")


class BM25Config(BaseModel):
    """Configuration for BM25 retrieval."""

    k1: float = Field(default=1.5, description="BM25 k1 parameter")
    b: float = Field(default=0.75, description="BM25 b parameter")


class RerankerConfig(BaseModel):
    """Configuration for cross encoder reranker."""

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross encoder model name",
    )
    batch_size: int = Field(default=32, description="Batch size for reranking")
    cache_dir: str = Field(default=".cache", description="Cache directory for models")


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    max_chunk_size: int = Field(default=512, description="Maximum chunk size in tokens")
    overlap: int = Field(default=50, description="Overlap between chunks in tokens")
    respect_headers: bool = Field(default=True, description="Respect document headers as boundaries")


class RAGConfig(BaseModel):
    """Top-level RAG configuration."""

    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    top_k_hybrid: int = Field(default=20, description="Top K results from hybrid search before reranking")
    top_k_reranked: int = Field(default=5, description="Final top K results after reranking")
    knowledge_base_dir: str = Field(default="./knowledge_base", description="Directory containing PDFs")

    @field_validator("top_k_hybrid", "top_k_reranked")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


def load_config(config_path: Optional[str] = None) -> RAGConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses config.yaml in current directory.

    Returns:
        RAGConfig: Validated configuration object.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML is invalid.
        ValueError: If configuration validation fails.
    """
    if config_path is None:
        config_path = "config.yaml"

    config_file = Path(config_path)
    if not config_file.exists():
        # Return default config if file doesn't exist
        return RAGConfig()

    with open(config_file) as f:
        config_dict = yaml.safe_load(f) or {}

    return RAGConfig(**config_dict)
