# AGENTS.md Guide for Rejection Sampling Recipes

This AGENTS.md file provides guidance for code agents working with this codebase.

## Core Project Structure

- `/src`: This contains the core framework code
  - `base.py`: Base classes `Stage` and `BaseRecipe` that define the processing pipeline architecture
  - `pipeline.py`: Pipeline execution engine
  - `/utils`: Utility modules for data I/O and helper functions
- `/recipes`: This contains different recipe implementations for data synthesis
  - Each recipe is a self-contained module with its own config, stages, and tools
  - Examples: `/sft`
- `/scripts`: Utility scripts for preprocessing and service deployment
  - `preprocess_images.py`: Image preprocessing utilities
  - `/launch_serve`: Scripts for model service deployment
- `/tests`: Test files for the framework
  - `/mock`: Mock data for testing

## Core Concepts

### Stage
A `Stage` is a single processing step in the pipeline. Stages inherit from the `Stage` base class and implement `process_item(item: dict) -> dict`.

**Three execution modes:**
- **Sync mode** (default): Sequential processing
- **Async mode** (`@Stage.async_mode`): Concurrent processing with asyncio
- **Threaded mode** (`@Stage.threaded_mode`): Concurrent processing with thread pool

### Recipe
A `Recipe` defines a sequence of stages. Recipes inherit from `BaseRecipe` and implement `stages() -> list[Stage]`.

### Pipeline
The pipeline engine executes recipes on datasets with features like batching, error handling, and checkpoint/resume.

## Coding Conventions

- Keep each recipe self-contained in its own directory with `recipe.py`, `config.py`, `tools.py`
- Implement stages by inheriting from `Stage` and implementing `process_item()`
- Use decorators `@Stage.async_mode` or `@Stage.threaded_mode` for concurrent processing
- Never modify framework internal fields: `_resume_id`, `_failed`, `_error`, `_traceback`
- Follow the existing project structure when adding new recipes

## Adding a New Recipe

When adding a new recipe:
1. Create a new directory under `/recipes` with your recipe name
2. Implement the required files:
   - `recipe.py`: Main recipe class inheriting from `BaseRecipe`
   - `config.py`: Configuration dataclass for the recipe
   - `tools.py`: Helper functions and clients
   - `entrypoint/run.py` and `run.sh`: Entry points for running the recipe
3. Define stages by inheriting from `Stage` and implementing `process_item()`
4. Choose appropriate execution mode based on workload (sync/async/threaded)


## Code Style

- Code style is enforced with `ruff`
- Line length: 100 characters
- Run `ruff format .` to format code
- Run `ruff check .` to check for issues
- Install quality tools: `pip install -e ".[dev]"`

