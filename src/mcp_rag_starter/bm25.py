"""BM25 retrieval for lexical search."""

from typing import Optional

from rank_bm25 import BM25Okapi

from mcp_rag_starter.config import BM25Config


class BM25Retriever:
    """BM25-based lexical retriever."""

    def __init__(self, config: Optional[BM25Config] = None):
        """Initialize BM25 retriever.

        Args:
            config: BM25 configuration.
        """
        self.config = config or BM25Config()
        self.corpus: list[str] = []
        self.metadata: list[dict] = []
        self.bm25: Optional[BM25Okapi] = None

    def index(self, documents: list[dict]) -> None:
        """Index documents for BM25 retrieval.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys.
        """
        self.corpus = []
        self.metadata = []

        for doc in documents:
            content = doc.get("content", "")
            self.corpus.append(content)
            self.metadata.append(doc)

        if self.corpus:
            tokenized_corpus = [doc.split() for doc in self.corpus]
            self.bm25 = BM25Okapi(
                tokenized_corpus,
                k1=self.config.k1,
                b=self.config.b,
            )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for relevant documents using BM25.

        Args:
            query: Query string.
            k: Number of results to return.

        Returns:
            List of dicts with 'metadata' and 'score' keys.
        """
        if self.bm25 is None or not self.corpus:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "metadata": self.metadata[idx],
                    "score": float(scores[idx]),
                })

        return results
