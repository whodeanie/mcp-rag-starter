"""Tests for document chunking."""

import pytest

from mcp_rag_starter.chunking import TextChunker
from mcp_rag_starter.config import ChunkingConfig


def test_chunk_basic():
    """Test basic chunking."""
    config = ChunkingConfig(max_chunk_size=20, overlap=5)
    chunker = TextChunker(config)

    text = "This is a test. This is another sentence. And yet another one here."
    chunks = chunker.chunk(text, source="test.txt")

    assert len(chunks) > 0
    assert all("content" in c for c in chunks)
    assert all("source" in c for c in chunks)
    assert all(c["source"] == "test.txt" for c in chunks)


def test_chunk_empty():
    """Test chunking empty text."""
    chunker = TextChunker()
    chunks = chunker.chunk("", source="empty.txt")
    assert len(chunks) == 0


def test_chunk_multiline():
    """Test chunking multiline text."""
    config = ChunkingConfig(max_chunk_size=30, overlap=5)
    chunker = TextChunker(config)

    text = "Line one\nLine two\nLine three\nLine four\nLine five"
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk["content"]) > 0


def test_chunk_with_headers():
    """Test chunking with header respect."""
    config = ChunkingConfig(max_chunk_size=50, overlap=5, respect_headers=True)
    chunker = TextChunker(config)

    text = "# Header One\nSome content here\n## Header Two\nMore content"
    chunks = chunker.chunk(text)

    assert len(chunks) > 0


def test_chunk_metadata():
    """Test that metadata is properly set."""
    chunker = TextChunker()
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunker.chunk(text, source="doc.pdf")

    assert all("start_idx" in c for c in chunks)
    assert all("end_idx" in c for c in chunks)
    assert all(c["source"] == "doc.pdf" for c in chunks)
