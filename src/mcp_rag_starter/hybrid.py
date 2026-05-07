"""Hybrid retrieval combining BM25 and vector search via reciprocal rank fusion."""

from mcp_rag_starter.bm25 import BM25Retriever
from mcp_rag_starter.vector_store import FAISSVectorStore


class HybridRetriever:
    """Combines BM25 and vector search using reciprocal rank fusion (RRF)."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_store: FAISS vector store instance.
            bm25_retriever: BM25 retriever instance.
            rrf_k: RRF parameter (higher means more weight to both methods).
        """
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """Search using hybrid approach (BM25 + vector search via RRF).

        Args:
            query: Text query string.
            query_embedding: Embedding vector for query.
            k: Number of final results to return.

        Returns:
            List of dicts with 'metadata', 'score', and 'rank' keys.
        """
        # Get results from both methods
        vector_results = self.vector_store.search(query_embedding, k=k * 2)
        bm25_results = self.bm25.search(query, k=k * 2)

        # Create lookup tables for RRF scores
        rrf_scores = {}

        # Add vector search scores via RRF
        for rank, result in enumerate(vector_results):
            doc_id = self._get_doc_id(result["metadata"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)

        # Add BM25 scores via RRF
        for rank, result in enumerate(bm25_results):
            doc_id = self._get_doc_id(result["metadata"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (self.rrf_k + rank + 1)

        # Create metadata lookup
        all_metadata = {}
        for result in vector_results + bm25_results:
            doc_id = self._get_doc_id(result["metadata"])
            all_metadata[doc_id] = result["metadata"]

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            results.append({
                "metadata": all_metadata[doc_id],
                "score": float(score),
                "rank": rank + 1,
            })

        return results

    @staticmethod
    def _get_doc_id(metadata: dict) -> str:
        """Generate unique ID for document based on metadata.

        Args:
            metadata: Metadata dict from chunk.

        Returns:
            Unique document identifier.
        """
        source = metadata.get("source", "unknown")
        start = metadata.get("start_idx", 0)
        return f"{source}:{start}"
