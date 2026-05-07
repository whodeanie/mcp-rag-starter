# Contributing to mcp-rag-starter

Thank you for your interest in contributing to mcp-rag-starter. We welcome contributions from the community.

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install with dev dependencies: `pip install -e ".[dev]"`

## Code Quality

We maintain high standards for code quality:

- Run `ruff check .` to lint your code
- Run `black src tests` to format code
- Run `pytest` to ensure tests pass

All pull requests must pass linting and tests.

## Testing

Write tests for new features in the `tests/` directory:

```bash
pytest
pytest --cov=src/mcp_rag_starter
```

## Documentation

Update docstrings using Google style. All public functions must have docstrings.

## Commits

Keep commits atomic and focused. Use clear commit messages describing what changed and why.

## Pull Requests

Include a clear description of:
- What problem does this solve
- How does it solve it
- Any breaking changes
- How to test the change

## License

By contributing, you agree your code will be licensed under the MIT license.
