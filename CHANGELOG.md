# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-15

### Added
- Initial release of mcp-rag-starter
- PDF ingestion with intelligent header-aware chunking
- FAISS-based vector search with persistence
- BM25 lexical search for rare terms and identifiers
- Reciprocal rank fusion (RRF) for hybrid retrieval
- Cross-encoder reranking with ms-marco-MiniLM-L-6-v2
- Citation tracking and source attribution
- MCP server with three tools: ingest_pdf, query, get_stats
- YAML-based configuration with Pydantic validation
- Evaluation harness with 20 hand-crafted QA pairs
- Comprehensive test suite covering chunking, hybrid retrieval, and config
- GitHub Actions CI for linting and testing
- Example PDF generated from Federalist Papers (public domain)
