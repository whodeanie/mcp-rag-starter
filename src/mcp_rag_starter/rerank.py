"""Cross-encoder reranking for result refinement."""

from typing import Optional

from cross_encoder import CrossEncoder

from mcp_rag_starter.config import RerankerConfig


class CrossEncoderReranker:
    """Rerank results using a cross-encoder model."""

    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize reranker.

        Args:
            config: Reranker configuration.
        """
        self.config = config or RerankerConfig()
        self.model = CrossEncoder(
            self.config.model_name,
            cache_folder=self.config.cache_dir,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        k: int = 5,
    ) -> list[dict]:
        """Rerank candidate results using cross-encoder.

        Args:
            query: Original query string.
            candidates: List of candidate dicts with 'metadata' key.
            k: Number of final results to return.

        Returns:
            Reranked list of dicts with added 'rerank_score' key.
        """
        if not candidates:
            return []

        # Extract candidate texts
        texts = [cand["metadata"].get("content", "") for cand in candidates]

        # Compute relevance scores
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)

        # Add scores to candidates
        scored_candidates = [
            {**cand, "rerank_score": float(score)}
            for cand, score in zip(candidates, scores)
        ]

        # Sort by rerank score and return top k
        reranked = sorted(
            scored_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )[:k]

        return reranked
