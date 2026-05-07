"""MCP server for RAG with ingest, query, and stats tools."""

import json
from pathlib import Path
from typing import Any

from mcp import Server, types
from mcp.types import Tool, TextContent, ToolResult

from mcp_rag_starter.bm25 import BM25Retriever
from mcp_rag_starter.citations import create_cited_response
from mcp_rag_starter.config import RAGConfig, load_config
from mcp_rag_starter.embeddings import EmbeddingEngine
from mcp_rag_starter.hybrid import HybridRetriever
from mcp_rag_starter.ingestion import ingest_directory, ingest_pdf
from mcp_rag_starter.rerank import CrossEncoderReranker
from mcp_rag_starter.vector_store import FAISSVectorStore


class RAGMCPServer:
    """MCP server for retrieval augmented generation."""

    def __init__(self, config_path: str | None = None):
        """Initialize RAG MCP server.

        Args:
            config_path: Path to YAML config file.
        """
        self.config = load_config(config_path)
        self.server = Server("mcp-rag-starter")

        # Initialize components
        self.embedder = EmbeddingEngine(self.config.embeddings)
        self.vector_store = FAISSVectorStore(self.config.vector_store)
        self.bm25 = BM25Retriever(self.config.bm25)
        self.reranker = CrossEncoderReranker(self.config.reranker)
        self.hybrid = HybridRetriever(
            self.vector_store,
            self.bm25,
            rrf_k=60,
        )

        self.indexed_chunks: list[dict] = []

        # Register tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register MCP tools."""

        @self.server.call_tool()
        async def ingest_pdf_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
            """Ingest a PDF document into the knowledge base."""
            if name != "ingest_pdf":
                return ToolResult(
                    content=[TextContent(type="text", text="Unknown tool")],
                    is_error=True,
                )

            pdf_path = arguments.get("pdf_path")
            if not pdf_path:
                return ToolResult(
                    content=[TextContent(type="text", text="pdf_path required")],
                    is_error=True,
                )

            try:
                chunks = ingest_pdf(pdf_path, self.config.chunking)
                self.indexed_chunks.extend(chunks)

                # Index chunks
                contents = [c["content"] for c in chunks]
                embeddings = self.embedder.embed(contents)

                self.vector_store.add(embeddings, chunks)
                self.bm25.index(chunks)

                return ToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Successfully ingested {pdf_path}: {len(chunks)} chunks indexed",
                        )
                    ]
                )
            except Exception as e:
                return ToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    is_error=True,
                )

        @self.server.call_tool()
        async def query_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
            """Query the knowledge base."""
            if name != "query":
                return ToolResult(
                    content=[TextContent(type="text", text="Unknown tool")],
                    is_error=True,
                )

            query = arguments.get("query")
            if not query:
                return ToolResult(
                    content=[TextContent(type="text", text="query required")],
                    is_error=True,
                )

            try:
                # Embed query
                query_embedding = self.embedder.embed_single(query)

                # Hybrid search
                hybrid_results = self.hybrid.search(
                    query,
                    query_embedding,
                    k=self.config.top_k_hybrid,
                )

                # Rerank
                reranked = self.reranker.rerank(
                    query,
                    hybrid_results,
                    k=self.config.top_k_reranked,
                )

                # Format response
                response = create_cited_response("", reranked)
                return ToolResult(
                    content=[TextContent(type="text", text=json.dumps(response, indent=2))]
                )
            except Exception as e:
                return ToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    is_error=True,
                )

        @self.server.call_tool()
        async def get_stats_tool(name: str, arguments: dict[str, Any]) -> ToolResult:
            """Get knowledge base statistics."""
            if name != "get_stats":
                return ToolResult(
                    content=[TextContent(type="text", text="Unknown tool")],
                    is_error=True,
                )

            stats = {
                "total_chunks": len(self.indexed_chunks),
                "vector_store_size": self.vector_store.vector_count,
                "embedding_model": self.config.embeddings.model_name,
                "reranker_model": self.config.reranker.model_name,
                "config": {
                    "top_k_hybrid": self.config.top_k_hybrid,
                    "top_k_reranked": self.config.top_k_reranked,
                    "chunk_size": self.config.chunking.max_chunk_size,
                },
            }

            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(stats, indent=2))]
            )

        # Add tools to server
        self.server.add_tool(
            Tool(
                name="ingest_pdf",
                description="Ingest a PDF document into the knowledge base",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Path to PDF file",
                        }
                    },
                    "required": ["pdf_path"],
                },
            )
        )

        self.server.add_tool(
            Tool(
                name="query",
                description="Query the knowledge base",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query string",
                        }
                    },
                    "required": ["query"],
                },
            )
        )

        self.server.add_tool(
            Tool(
                name="get_stats",
                description="Get knowledge base statistics",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            )
        )

    async def run(self) -> None:
        """Run the MCP server."""
        async with self.server:
            pass
