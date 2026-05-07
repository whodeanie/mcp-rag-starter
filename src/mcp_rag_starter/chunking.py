"""Header-aware semantic document chunking."""

from typing import Optional

from mcp_rag_starter.config import ChunkingConfig


class TextChunker:
    """Intelligently chunk text while respecting structural boundaries."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config or ChunkingConfig()
        self.max_chunk_size = self.config.max_chunk_size
        self.overlap = self.config.overlap
        self.respect_headers = self.config.respect_headers

    def chunk(self, text: str, source: str = "") -> list[dict]:
        """Chunk text into semantically coherent segments.

        Respects structural boundaries (headers, paragraphs) when configured.
        Returns chunks with metadata for citation tracking.

        Args:
            text: Text to chunk.
            source: Source identifier (e.g., filename or page number).

        Returns:
            List of chunk dicts with keys: 'content', 'source', 'start_idx', 'end_idx'.
        """
        if not text:
            return []

        chunks = []
        lines = text.split("\n")
        current_chunk = []
        current_size = 0
        start_idx = 0

        for idx, line in enumerate(lines):
            line_tokens = len(line.split())

            # Check if adding this line would exceed max chunk size
            would_exceed = current_size + line_tokens > self.max_chunk_size

            # Identify structural boundaries (simple heuristic)
            is_header = self.respect_headers and (
                line.strip().startswith("#") or
                line.strip().isupper() or
                (len(line.strip()) < 50 and line.strip() and line.strip()[0].isupper())
            )

            if would_exceed and current_chunk:
                # Save current chunk
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        "content": chunk_text,
                        "source": source,
                        "start_idx": start_idx,
                        "end_idx": idx - 1,
                    })

                # Start overlap
                overlap_count = 0
                overlap_lines = []
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_count += len(current_chunk[i].split())
                    overlap_lines.insert(0, current_chunk[i])
                    if overlap_count >= self.overlap:
                        break

                current_chunk = overlap_lines
                current_size = overlap_count
                start_idx = max(0, idx - len(overlap_lines))

            current_chunk.append(line)
            current_size += line_tokens

            # Break on headers if configured
            if is_header and current_chunk[:-1]:
                chunk_text = "\n".join(current_chunk[:-1]).strip()
                if chunk_text:
                    chunks.append({
                        "content": chunk_text,
                        "source": source,
                        "start_idx": start_idx,
                        "end_idx": idx - 1,
                    })
                current_chunk = [line]
                current_size = line_tokens
                start_idx = idx

        # Save final chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "source": source,
                    "start_idx": start_idx,
                    "end_idx": len(lines) - 1,
                })

        return chunks
