"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from mcp_rag_starter.config import RAGConfig, load_config


def test_load_default_config():
    """Test loading default configuration."""
    config = load_config("/nonexistent/path.yaml")
    assert isinstance(config, RAGConfig)
    assert config.top_k_hybrid == 20
    assert config.top_k_reranked == 5


def test_load_custom_config():
    """Test loading custom YAML configuration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_dict = {
            "top_k_hybrid": 30,
            "top_k_reranked": 10,
            "knowledge_base_dir": "/custom/path",
        }
        yaml.dump(config_dict, f)
        config_path = f.name

    try:
        config = load_config(config_path)
        assert config.top_k_hybrid == 30
        assert config.top_k_reranked == 10
        assert config.knowledge_base_dir == "/custom/path"
    finally:
        Path(config_path).unlink()


def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        RAGConfig(top_k_hybrid=-1)

    with pytest.raises(ValueError):
        RAGConfig(top_k_reranked=0)


def test_config_defaults():
    """Test configuration defaults."""
    config = RAGConfig()

    assert config.embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.chunking.max_chunk_size == 512
    assert config.chunking.overlap == 50
