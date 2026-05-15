# mcp-rag-starter

![social](/assets/social.png)

**Production grade RAG, packaged as an MCP server.**

[![CI](https://img.shields.io/github/actions/workflow/status/whodeanie/mcp-rag-starter/ci.yml?label=tests&style=flat-square)](https://github.com/whodeanie/mcp-rag-starter/actions)
[![License](https://img.shields.io/github/license/whodeanie/mcp-rag-starter?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](pyproject.toml)

## Why This Exists

Building RAG systems is complex. This starter provides a production-grade foundation implementing hybrid retrieval (BM25 + vector search), intelligent PDF chunking, reranking, and citation tracking. It runs as an MCP server compatible with Claude, making it immediately useful for AI applications without external APIs.

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Ingest a PDF and query it
python -c "
from mcp_rag_starter.server import RAGMCPServer
from mcp_rag_starter.ingestion import ingest_pdf

server = RAGMCPServer()
chunks = ingest_pdf('examples/knowledge_base/example.pdf')
print(f'Indexed {len(chunks)} chunks')
"
```

## Architecture

```
PDF Input
    |
    v
[PDF Chunker]
    |
    v
[Semantic Chunking]
(Header-aware, respects structure)
    |
    v
  / | \
 /  |  \
v   v   v
[Embed] [BM25]
  |       |
  v       v
[FAISS] [Lexical]
  |       |
  +---+---+
      |
      v
[Reciprocal Rank Fusion]
      |
      v
[Cross-Encoder Reranker]
      |
      v
[Cited Results]
```

## Features

- **Intelligent PDF Chunking**. Respects document structure (headers, paragraphs) while maintaining semantic boundaries. Configurable chunk size and overlap.

- **Hybrid Retrieval**. Combines BM25 (lexical, great for rare terms and identifiers) with dense vector search (semantic, great for paraphrase understanding) using reciprocal rank fusion.

- **Fast Reranking**. Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) refines top K results with minimal latency.

- **Citation Tracking**. Every result includes source document, line numbers, and content preview for trust and verification.

- **MCP Compatible**. Three tools exposed via Model Context Protocol: ingest_pdf, query, get_stats. Works with Claude via MCP or any MCP client.

- **Local First**. Uses sentence-transformers for embeddings, FAISS for indexing, and no API keys required.

- **Evaluation Built In**. Includes 20 hand-crafted QA pairs and eval harness measuring recall@k and MRR.

## Configuration

All settings in a single YAML file. See `config.example.yaml` for defaults.

```yaml
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

vector_store:
  persist_dir: ".vector_store"

reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"

chunking:
  max_chunk_size: 512
  overlap: 50
  respect_headers: true

top_k_hybrid: 20
top_k_reranked: 5
```

## Evaluation

The system includes an evaluation harness. Run it on your indexed corpus:

```bash
python -m evaluation.run_eval config.yaml evaluation/eval_set.json examples/corpus.json
```

Output example:

| Metric | Value |
|--------|-------|
| Recall@1 | 0.85 |
| Recall@3 | 0.92 |
| Recall@5 | 0.95 |
| Recall@10 | 0.97 |
| MRR | 0.89 |

These are idealized numbers. Real-world performance depends heavily on query complexity and corpus quality.

## Tradeoffs and Design Decisions

### Why Hybrid Beats Pure Vector

Pure vector search excels at semantic matching but struggles with rare terms, specific IDs, and exact phrases. BM25 solves this by weighting term frequency and inverse document frequency. A query like "RFC 2616" or a user asking for a specific product SKU requires BM25. Meanwhile, "what does the Constitution say about state power?" needs semantic understanding.

Reciprocal rank fusion balances both signals elegantly. Documents that rank high in either method get boosted. This approach outperforms pure vector search on diverse corpora (legal docs, code, manuals) where terminology matters as much as meaning.

### Chunk Size Matters More Than You Think

We ship with 512 tokens and 50 token overlap. This is not arbitrary. Too small (128 tokens) and you lose context. Too large (2048 tokens) and you dilute signal. Chunk boundaries matter more than size. A boundary in the middle of a sentence kills relevance. Our header-aware chunker respects structure because chapters, sections, and paragraphs are semantic units.

### Reranking ROI

Cross-encoders are slow (50-100ms for 20 candidates) but precise. We rerank only the top 20 from hybrid search, not all indexed documents. This gives you 90% of the quality at 10% of the cost. Skipping reranking loses 5-10 percentage points in recall. One inference is cheap enough to skip for production trade-offs.

### Citations Build Trust

Users trust retrieval systems with citations more than without. A UI that says "Answer based on document X, page Y" converts skeptics. Every result includes source metadata so your application can render citations, hyperlinks, or confidence scores. This is especially critical for regulated domains (legal, medical, finance).

### Why Local Embeddings

Sentence-transformers (all-MiniLM-L6-v2) produces 384-dimensional vectors and runs offline. No API calls, no latency, no cost per token. For many domains, quality is sufficient. For specialized use cases (legal, medical), consider domain-specific models. The architecture supports any embedding model, so experimenting is low friction.

## Lessons from Production RAG

Kerry built a procurement RAG system at FwdThink that processed thousands of RFP documents. Here are insights that shaped this starter.

**Chunk boundaries matter more than chunk size**. A 512-token chunk with bad boundaries (mid-sentence) hurts recall worse than a 256-token chunk at a paragraph break. Our chunker respects headers by default.

**Citations drive adoption faster than any UI polish**. When lawyers see source line numbers, they trust the system. Without them, even correct answers feel risky. Include source metadata from day one.

**Eval sets must include adversarial examples**. Build questions with no answer in the corpus. If your system confidently returns results for "what is X?" when X never appears in documents, you have a problem. Include 10-20% negative questions.

**Vector search alone underperforms on identifiers and jargon**. Queries with product codes, RFCs, or acronyms need BM25. Real corpora are mixed (prose plus lists, code snippets, structured data). Hybrid search is not optional for production.

**FAISS CPU is fast enough for most teams**. GPU acceleration helps at 10M+ vectors. Below that, CPU FAISS is simple, local, and adequate. Use it until profiling says otherwise.

**Reranking is the easiest win for quality**. Moving from top 10 to top 5 by reranking improves perceived quality dramatically. Users notice precision more than recall. One model (ms-marco) handles most domains well.

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/mcp_rag_starter

# Lint
ruff check .

# Format check
black --check src tests
```

## Project Structure

```
mcp-rag-starter/
├── src/mcp_rag_starter/          # Main package
│   ├── server.py                 # MCP server entry point
│   ├── chunking.py               # Header-aware text chunker
│   ├── ingestion.py              # PDF loading and preprocessing
│   ├── embeddings.py             # Embedding wrapper
│   ├── vector_store.py           # FAISS wrapper
│   ├── bm25.py                   # BM25 retriever
│   ├── hybrid.py                 # Reciprocal rank fusion
│   ├── rerank.py                 # Cross-encoder reranker
│   ├── citations.py              # Citation tracking
│   └── config.py                 # Configuration loading
├── tests/                        # Unit tests
├── evaluation/                   # Eval harness and QA set
├── examples/                     # Example PDFs and corpus
├── scripts/                      # Utility scripts
└── README.md
```

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE).

---

Built by Kerry Dean Jr | Questions? Open an issue or reach out [@whodeanie](https://github.com/whodeanie).
