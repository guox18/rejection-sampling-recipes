# Contributing Guide

Thank you for your interest in contributing to this project!

## Development Environment

```bash
# Using uv (recommended)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

## Code Standards

- Use `ruff` for linting and formatting
- Code comments, docstrings, and commit messages should be in English
- Type hints are recommended

```bash
# Check
ruff check .
ruff format --check .

# Auto-fix
ruff check --fix .
ruff format .
```

## Contribution Types

### 1. Adding a New Verifier

See [src/verifier/README.md](src/verifier/README.md) for details.

**Key Steps**:
1. Implement the `BaseVerifier` interface
2. Register with `@register_verifier("name")`
3. Add test data in `tests/fixtures/model_outputs.json`
4. Add test class in `tests/test_verifier.py`

### 2. Adding a New Formatter

Similar to Verifier, refer to implementations in `src/formatter/`.

### 3. Bug Fixes

1. Create an Issue to describe the problem first
2. Fork and fix
3. Add tests covering the fix scenario
4. Submit a PR

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_verifier.py -v

# With coverage
pytest tests/ -v --cov=src
```

## Submitting a PR

Fork this repository, create a feature branch, and submit a Pull Request.

### Commit Convention

```
feat: New feature
fix: Bug fix
docs: Documentation update
test: Test related
refactor: Code refactoring
```

## Project Structure

```
rejection-sampling-recipes/
├── src/
│   ├── sampler/        # Samplers (OpenAI API, vLLM)
│   ├── verifier/       # Verifiers ← Most common contribution area
│   ├── formatter/      # Formatters (SFT, DPO)
│   ├── utils/          # Utility functions
│   └── pipeline.py     # Main pipeline
├── tests/
│   ├── fixtures/       # Test data
│   └── test_*.py       # Test files
├── configs/            # Hydra configurations
└── transforms/         # Data transform functions
```

## Feedback

- Bug reports: Create an Issue with reproduction steps
- Feature requests: Create an Issue describing the use case
