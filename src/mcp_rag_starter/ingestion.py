"""PDF ingestion and preprocessing."""

from pathlib import Path

from pypdf import PdfReader

from mcp_rag_starter.chunking import TextChunker
from mcp_rag_starter.config import ChunkingConfig


def load_pdf(pdf_path: str | Path) -> str:
    """Load text from PDF file.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        Extracted text from PDF.

    Raises:
        FileNotFoundError: If PDF does not exist.
        ValueError: If PDF cannot be read.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        text_pages = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                text_pages.append(f"[Page {page_num}]\n{text}")
        return "\n\n".join(text_pages)
    except Exception as e:
        raise ValueError(f"Failed to read PDF {pdf_path}: {e}") from e


def ingest_pdf(pdf_path: str | Path, chunking_config: ChunkingConfig | None = None) -> list[dict]:
    """Ingest a PDF and return chunks with metadata.

    Args:
        pdf_path: Path to PDF file.
        chunking_config: Chunking configuration.

    Returns:
        List of chunks with content, source, and position metadata.

    Raises:
        FileNotFoundError: If PDF does not exist.
        ValueError: If PDF cannot be read.
    """
    pdf_path = Path(pdf_path)
    text = load_pdf(pdf_path)

    chunker = TextChunker(chunking_config)
    chunks = chunker.chunk(text, source=pdf_path.name)

    return chunks


def ingest_directory(
    directory: str | Path,
    chunking_config: ChunkingConfig | None = None,
) -> list[dict]:
    """Ingest all PDFs in a directory.

    Args:
        directory: Directory containing PDF files.
        chunking_config: Chunking configuration.

    Returns:
        List of all chunks from all PDFs.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_chunks = []
    for pdf_file in directory.glob("*.pdf"):
        try:
            chunks = ingest_pdf(pdf_file, chunking_config)
            all_chunks.extend(chunks)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Failed to ingest {pdf_file}: {e}")

    return all_chunks
