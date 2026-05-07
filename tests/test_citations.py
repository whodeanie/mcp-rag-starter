"""Tests for citation tracking."""

from mcp_rag_starter.citations import create_cited_response, extract_citations, format_citations


def test_extract_citations():
    """Test citation extraction."""
    result = {
        "metadata": {
            "source": "doc.pdf",
            "start_idx": 10,
            "end_idx": 20,
            "content": "This is the actual content from the document",
        }
    }

    citation = extract_citations(result)

    assert citation.source == "doc.pdf"
    assert citation.start_idx == 10
    assert citation.end_idx == 20
    assert "content" in citation.content_preview


def test_format_citations():
    """Test citation formatting."""
    results = [
        {
            "metadata": {
                "source": "doc1.pdf",
                "start_idx": 0,
                "end_idx": 5,
                "content": "First source content",
            }
        },
        {
            "metadata": {
                "source": "doc2.pdf",
                "start_idx": 10,
                "end_idx": 15,
                "content": "Second source content",
            }
        },
    ]

    formatted = format_citations(results)

    assert "doc1.pdf" in formatted
    assert "doc2.pdf" in formatted
    assert "[1]" in formatted
    assert "[2]" in formatted


def test_format_citations_empty():
    """Test citation formatting with empty results."""
    formatted = format_citations([])
    assert formatted == ""


def test_create_cited_response():
    """Test creating a response with citations."""
    results = [
        {
            "metadata": {
                "source": "source.pdf",
                "start_idx": 0,
                "end_idx": 10,
                "content": "Some content",
            }
        }
    ]

    response = create_cited_response("This is the answer", results)

    assert response["answer"] == "This is the answer"
    assert "citations" in response
    assert response["source_count"] == 1
