"""Evaluation harness for RAG system. Measures recall at K and MRR."""

import json
import sys
from pathlib import Path

from mcp_rag_starter.bm25 import BM25Retriever
from mcp_rag_starter.config import load_config
from mcp_rag_starter.embeddings import EmbeddingEngine
from mcp_rag_starter.hybrid import HybridRetriever
from mcp_rag_starter.rerank import CrossEncoderReranker
from mcp_rag_starter.vector_store import FAISSVectorStore


def load_eval_set(eval_path: str) -> list[dict]:
    """Load evaluation question-answer pairs.

    Args:
        eval_path: Path to eval_set.json.

    Returns:
        List of QA pairs.
    """
    with open(eval_path) as f:
        return json.load(f)


def load_corpus(corpus_path: str) -> list[dict]:
    """Load corpus for evaluation.

    Args:
        corpus_path: Path to corpus JSON file.

    Returns:
        List of documents.
    """
    with open(corpus_path) as f:
        return json.load(f)


def compute_recall_at_k(results: list[dict], answer_keywords: list[str], k: int) -> bool:
    """Check if answer keywords appear in top K results.

    Args:
        results: Retrieved results.
        answer_keywords: Keywords that should appear in relevant results.
        k: Position to evaluate recall at.

    Returns:
        True if any answer keyword found in top K results.
    """
    top_k = results[:k]
    top_k_text = " ".join([r.get("metadata", {}).get("content", "") for r in top_k])

    return any(keyword.lower() in top_k_text.lower() for keyword in answer_keywords)


def run_evaluation(
    config_path: str | None = None,
    eval_set_path: str | None = None,
    corpus_path: str | None = None,
) -> dict:
    """Run RAG evaluation.

    Args:
        config_path: Path to config YAML.
        eval_set_path: Path to eval_set.json.
        corpus_path: Path to corpus JSON.

    Returns:
        Dictionary with evaluation metrics.
    """
    if eval_set_path is None:
        eval_set_path = "evaluation/eval_set.json"
    if corpus_path is None:
        corpus_path = "examples/corpus.json"

    config = load_config(config_path)
    eval_set = load_eval_set(eval_set_path)

    # Try to load corpus
    corpus = []
    if Path(corpus_path).exists():
        corpus = load_corpus(corpus_path)
    else:
        print(f"Warning: Corpus file not found at {corpus_path}")

    # Initialize components
    embedder = EmbeddingEngine(config.embeddings)
    vector_store = FAISSVectorStore(config.vector_store)
    bm25 = BM25Retriever(config.bm25)
    reranker = CrossEncoderReranker(config.reranker)
    hybrid = HybridRetriever(vector_store, bm25)

    # Index corpus if available
    if corpus:
        contents = [c.get("content", "") for c in corpus]
        embeddings = embedder.embed(contents)
        vector_store.add(embeddings, corpus)
        bm25.index(corpus)

    # Evaluate
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    mrr = 0.0

    for qa in eval_set:
        question = qa["question"]
        answer = qa["answer"]
        answer_keywords = answer.split()[:3]  # Use first 3 words as keywords

        if not corpus:
            continue

        # Search
        query_embedding = embedder.embed_single(question)
        hybrid_results = hybrid.search(
            question,
            query_embedding,
            k=min(10, config.top_k_hybrid),
        )

        reranked = reranker.rerank(
            question,
            hybrid_results,
            k=10,
        )

        # Compute metrics
        if compute_recall_at_k(reranked, answer_keywords, 1):
            recall_at_1 += 1
        if compute_recall_at_k(reranked, answer_keywords, 3):
            recall_at_3 += 1
        if compute_recall_at_k(reranked, answer_keywords, 5):
            recall_at_5 += 1
        if compute_recall_at_k(reranked, answer_keywords, 10):
            recall_at_10 += 1

        # MRR
        for rank, result in enumerate(reranked, 1):
            content = result.get("metadata", {}).get("content", "")
            if any(kw.lower() in content.lower() for kw in answer_keywords):
                mrr += 1.0 / rank
                break

    num_queries = len(eval_set) if corpus else 0

    metrics = {
        "total_queries": num_queries,
        "recall_at_1": recall_at_1 / num_queries if num_queries > 0 else 0.0,
        "recall_at_3": recall_at_3 / num_queries if num_queries > 0 else 0.0,
        "recall_at_5": recall_at_5 / num_queries if num_queries > 0 else 0.0,
        "recall_at_10": recall_at_10 / num_queries if num_queries > 0 else 0.0,
        "mrr": mrr / num_queries if num_queries > 0 else 0.0,
    }

    return metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    eval_set_path = sys.argv[2] if len(sys.argv) > 2 else None
    corpus_path = sys.argv[3] if len(sys.argv) > 3 else None

    metrics = run_evaluation(config_path, eval_set_path, corpus_path)
    print(json.dumps(metrics, indent=2))
