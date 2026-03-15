# Contributing to MCP Memory Service

This project provides semantic memory and persistent storage for AI assistants through the Model Context Protocol.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Be respectful. Focus on constructive criticism and collaborative problem-solving. No harassment or discrimination.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- [uv](https://docs.astral.sh/uv/) package manager

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mcp-memory-service.git
   cd mcp-memory-service
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Verify installation**:
   ```bash
   uv run ruff check src/ tests/
   uv run pytest -x -m "not slow"
   ```

5. **Run the service**:
   ```bash
   uv run memory server
   ```

6. **Test with MCP Inspector** (optional):
   ```bash
   npx @modelcontextprotocol/inspector uv run memory server
   ```

### Docker Setup

```bash
docker compose up -d  # Starts Qdrant + FalkorDB + memory service
```

## Development Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Branch prefixes: `feature/`, `fix/`, `docs/`, `test/`, `refactor/`

### 2. Make Your Changes

- Follow the coding standards below
- Add/update tests as needed
- Keep commits focused and atomic

### 3. Test Your Changes

```bash
# Quality gates (all must pass)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest -x -m "not slow"

# Full suite
uv run pytest tests/

# With coverage
uv run pytest --cov=mcp_memory_service tests/
```

### 4. Commit Your Changes

Semantic commit messages:
```bash
git commit -m "feat: add memory export functionality"
git commit -m "fix: resolve timezone handling in memory search"
git commit -m "test: add coverage for storage backends"
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

### 5. Create a Pull Request

Open a PR to `27b-io/mcp-memory-service` (not the upstream `doobidoo/` repo) with:
- Clear title describing the change
- Description of what and why
- Reference to any related issues

## Coding Standards

### Python Style

- **Formatter/Linter**: Ruff, 129 character line length
- **Type checker**: basedpyright
- Use type hints for all function signatures
- Absolute imports only

### Code Organization

```python
# Import order (enforced by Ruff)
import standard_library
import third_party_libraries
from mcp_memory_service import local_modules

# Type hints
async def process_memory(content: str) -> dict[str, Any]:
    """Process and store memory content."""
    ...
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Never silently fail

## Testing Requirements

### Writing Tests

- Place tests in `tests/` directory
- Name test files with `test_` prefix
- Include both positive and negative test cases

Example:
```python
import pytest
from mcp_memory_service.models.memory import Memory

def test_memory_model_validation():
    """Test Memory model validates required fields."""
    memory = Memory(content="test content", content_hash="abc123")
    assert memory.content == "test content"
    assert memory.salience_score == 0.0
```

### Test Coverage

- Focus on critical paths and edge cases
- Test error handling scenarios
- Include integration tests where appropriate

## Submitting Changes

### Pull Request Guidelines

1. **PR Title**: Use semantic format (e.g., "feat: add batch memory operations")

2. **PR Checklist**:
   - [ ] Tests pass locally (`uv run pytest -x -m "not slow"`)
   - [ ] Linting passes (`uv run ruff check src/ tests/`)
   - [ ] Formatting passes (`uv run ruff format --check src/ tests/`)
   - [ ] No sensitive data exposed

## Reporting Issues

### Bug Reports

Include:
1. OS, Python version, MCP Memory Service version
2. Steps to reproduce (minimal example)
3. Expected vs actual behavior
4. Error messages/stack traces

### Feature Requests

Describe the problem you're solving, your proposed solution, and alternatives considered.

---

Questions? Open an [issue](https://github.com/27b-io/mcp-memory-service/issues) or [discussion](https://github.com/27b-io/mcp-memory-service/discussions).
