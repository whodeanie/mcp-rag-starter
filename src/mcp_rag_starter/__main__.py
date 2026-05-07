"""Entry point for MCP RAG server."""

import asyncio
import sys
from pathlib import Path

from mcp_rag_starter.server import RAGMCPServer


async def main() -> None:
    """Run the MCP RAG server."""
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    server = RAGMCPServer(config_path)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
