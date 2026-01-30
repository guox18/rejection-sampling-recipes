# Contributing to Rejection Sampling Recipes

Thank you for your interest in contributing to Rejection Sampling Recipes! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Adding a New Recipe](#adding-a-new-recipe)

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for everyone.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rejection-sampling-recipes.git
   cd rejection-sampling-recipes
   ```
3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/guox18/rejection-sampling-recipes.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Using uv (recommended)
uv sync --extra dev

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Pre-commit Hooks (Optional)

We recommend setting up pre-commit hooks to automatically check code style:

```bash
pre-commit install
```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes and commit them with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

3. Keep your branch up to date with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Please ensure your code passes all checks before submitting.

### Running Linters

```bash
# Check for linting issues
uvx ruff check .

# Auto-fix linting issues where possible
uvx ruff check --fix .

# Check formatting
uvx ruff format --check .

# Apply formatting
uvx ruff format .
```

### Style Guidelines

- **Line length**: 100 characters maximum
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with isort (handled by Ruff)
- **Type hints**: Encouraged for function signatures
- **Docstrings**: Required for public classes and functions

### Example

```python
def process_item(self, item: dict) -> dict:
    """
    Process a single data item.

    Args:
        item: Input data item with 'messages' field.

    Returns:
        Processed item with additional metadata.

    Raises:
        ValueError: If item is missing required fields.
    """
    if "messages" not in item:
        raise ValueError("Item must contain 'messages' field")
    
    # Processing logic here
    return item
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_pipeline.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both positive and negative test cases

Example:

```python
import pytest
from src.base import Stage

class TestStage:
    def test_process_item_success(self):
        """Test successful item processing."""
        # Test implementation
        pass

    def test_process_item_missing_field(self):
        """Test error handling for missing fields."""
        with pytest.raises(ValueError):
            # Test implementation
            pass
```

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass and code style checks succeed
2. Update documentation if needed
3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Create a Pull Request on GitHub
5. Fill in the PR template with:
   - Description of changes
   - Related issue numbers (if any)
   - Testing performed
   - Screenshots (if UI changes)

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Write clear PR titles and descriptions
- Respond to review feedback promptly
- Squash commits if requested

## Adding a New Recipe

When adding a new recipe, follow this structure:

### 1. Create Recipe Directory

```
recipes/
└── your_recipe_name/
    ├── __init__.py
    ├── config.py          # Configuration dataclass
    ├── recipe.py          # Main recipe class
    ├── tools.py           # Helper functions and clients
    └── entrypoint/
        ├── run.py         # Python entry point
        └── run.sh         # Shell script entry point
```

### 2. Implement Required Files

**config.py**:
```python
from dataclasses import dataclass, field

@dataclass
class YourConfig:
    """Configuration for YourRecipe."""
    
    # Model settings
    model_name: str = "your-model"
    temperature: float = 0.7
    
    # Processing settings
    batch_size: int = 32
```

**recipe.py**:
```python
from src.base import BaseRecipe, Stage
from .config import YourConfig

class YourStage(Stage):
    """Your processing stage."""
    
    def __init__(self, config: YourConfig):
        self.config = config
    
    def process_item(self, item: dict) -> dict:
        # Your processing logic
        return item

class YourRecipe(BaseRecipe):
    """Your recipe implementation."""
    
    def stages(self) -> list[Stage]:
        return [
            YourStage(self.config),
            # Add more stages as needed
        ]
```

### 3. Add Tests

Create test files in `tests/` for your recipe:
```python
# tests/test_your_recipe.py
from recipes.your_recipe_name.recipe import YourRecipe, YourConfig

def test_your_recipe():
    config = YourConfig()
    recipe = YourRecipe(config)
    # Test implementation
```

### 4. Update Documentation

- Add your recipe to the README.md
- Include usage examples
- Document any special requirements

## Questions?

If you have questions or need help, please:

1. Check existing [issues](https://github.com/guox18/rejection-sampling-recipes/issues)
2. Open a new issue with the "question" label
3. Be as specific as possible about your question

Thank you for contributing!
