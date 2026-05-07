"""Citation tracking and source attribution."""

from dataclasses import dataclass


@dataclass
class Citation:
    """Citation metadata for a result."""

    source: str
    start_idx: int
    end_idx: int
    content_preview: str


def extract_citations(result: dict) -> Citation:
    """Extract citation information from a result.

    Args:
        result: Result dict with metadata key.

    Returns:
        Citation object with source and position info.
    """
    metadata = result.get("metadata", {})
    content = metadata.get("content", "")
    preview = content[:100] + "..." if len(content) > 100 else content

    return Citation(
        source=metadata.get("source", "unknown"),
        start_idx=metadata.get("start_idx", 0),
        end_idx=metadata.get("end_idx", 0),
        content_preview=preview,
    )


def format_citations(results: list[dict]) -> str:
    """Format citations for display.

    Args:
        results: List of result dicts.

    Returns:
        Formatted citation string.
    """
    if not results:
        return ""

    citations = []
    for i, result in enumerate(results, 1):
        citation = extract_citations(result)
        citations.append(
            f"[{i}] {citation.source} "
            f"(lines {citation.start_idx}-{citation.end_idx}): "
            f"{citation.content_preview}"
        )

    return "\n".join(citations)


def create_cited_response(answer: str, results: list[dict]) -> dict:
    """Create a response with citations.

    Args:
        answer: Answer text.
        results: Supporting results.

    Returns:
        Dict with answer and citations.
    """
    return {
        "answer": answer,
        "citations": format_citations(results),
        "source_count": len(results),
    }
