# mcp-rag-starter

![social](/assets/social.png)

**A local-first RAG starter packaged as an MCP server.**

[![CI](https://img.shields.io/github/actions/workflow/status/whodeanie/mcp-rag-starter/ci.yml?label=tests&style=flat-square)](https://github.com/whodeanie/mcp-rag-starter/actions)
[![License](https://img.shields.io/github/license/whodeanie/mcp-rag-starter?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](pyproject.toml)

## Why This Exists

Building RAG systems is complex. This starter is a reference implementation for the core pieces teams usually need early: hybrid retrieval (BM25 + vector search), PDF chunking, optional reranking, and citation metadata. It runs as an MCP server compatible with Claude and keeps the stack local-first so it can be studied, forked, and extended without external API keys.

## What this is not

This is not a finished enterprise RAG platform. It does not include tenant isolation, production auth, durable multi-user storage, hosted observability, queue-backed ingestion, access-control enforcement, or a managed vector database. Treat it as a working starter and architecture reference, not a drop-in compliance-ready service.

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

- **Source Metadata**. Every result includes source document, line numbers, and a content preview.

- **MCP Compatible**. Three tools exposed via Model Context Protocol: ingest_pdf, query, get_stats. Works with Claude via MCP or any MCP client.

- **Local First**. Uses sentence-transformers for embeddings, FAISS for indexing, and no API keys required.

- **Evaluation Starter**. Includes a small QA fixture set and eval harness measuring recall@k and MRR.

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

The system includes a small evaluation harness. Run it on a corpus JSON that
matches your own documents:

```bash
python -m evaluation.run_eval config.yaml evaluation/eval_set.json examples/corpus.json
```

The script reports recall@1, recall@3, recall@5, recall@10, and MRR. This
repository ships fixture questions so the harness is easy to inspect, but it
does not ship a benchmark-grade corpus or claim default quality numbers.

## Tradeoffs and Design Decisions

### Why Hybrid Retrieval

Pure vector search excels at semantic matching but struggles with rare terms, specific IDs, and exact phrases. BM25 solves this by weighting term frequency and inverse document frequency. A query like "RFC 2616" or a user asking for a specific product SKU requires BM25. Meanwhile, "what does the Constitution say about state power?" needs semantic understanding.

Reciprocal rank fusion balances both signals without assuming the BM25 and
vector scores live on the same scale. Treat it as a strong default to evaluate,
not proof that hybrid search wins for every corpus.

### Chunk Size Tradeoffs

The default is 512 words with 50 words of overlap. That is a practical starting
point, not a universal rule. The chunker uses simple line and header heuristics
so readers can understand the behavior quickly.

### Reranking ROI

Cross-encoders are slower than first-pass retrieval but often more precise. This starter reranks only the top candidates from hybrid search, not all indexed documents. The exact latency and quality tradeoff depends on corpus size, hardware, and model choice, so measure it against your own eval set before relying on it.

### Source Metadata

Every result includes source metadata so an application can render citations,
links, or confidence notes. In regulated domains, this is only one piece of the
work; you still need policy, access control, audit, and review workflows around it.

### Why Local Embeddings

Sentence-transformers (all-MiniLM-L6-v2) produces 384-dimensional vectors and
runs offline. It avoids API calls and per-token fees, but local inference still
has a latency and hardware cost. Specialized use cases may need a different
embedding model.

## Lessons That Shaped This Starter

This starter is informed by production retrieval work, but the repository itself
is intentionally smaller. These are design notes, not claims that the starter
already includes every production control.

**Chunk boundaries matter more than chunk size**. A chunk with bad boundaries
can hurt recall worse than a smaller chunk at a paragraph break. The starter
uses simple structure-aware chunking by default.

**Source metadata changes the review conversation**. Even when a retrieval
result is correct, people need to see where it came from. Include source
metadata from day one.

**Eval sets must include adversarial examples**. Build questions with no answer in the corpus. If your system confidently returns results for "what is X?" when X never appears in documents, you have a problem. Include 10-20% negative questions.

**Vector search alone can miss identifiers and jargon**. Queries with product
codes, RFCs, or acronyms often need lexical search. Real corpora are mixed
prose, lists, code snippets, and structured data.

**FAISS CPU is a useful local default**. It keeps setup simple while you learn
whether the rest of the retrieval pipeline is worth hardening.

**Reranking is worth measuring early**. Cross-encoders can improve precision,
but they add latency. Measure the tradeoff against your own fixture set.

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
