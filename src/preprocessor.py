"""
Data Preprocessor.

Handles data loading, transformation, and validation.
"""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path


class DataPreprocessor:
    """Preprocesses input data into the required format."""

    REQUIRED_FIELDS = {"id", "messages", "metadata"}

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        transform: str | None = None,
    ):
        """
        Initialize data preprocessor.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file (data/input.jsonl)
            transform: Transform function spec (e.g., "transforms/gsm8k.py:transform")
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.transform_fn = self._load_transform(transform) if transform else None

    def _load_transform(self, transform_spec: str) -> Callable:
        """
        Load a transform function from a module.

        Args:
            transform_spec: "path/to/module.py:function_name"

        Returns:
            Callable transform function
        """
        if ":" not in transform_spec:
            raise ValueError(
                f"Invalid transform spec: '{transform_spec}'. "
                "Expected format: 'path/to/module.py:function_name'"
            )

        module_path, func_name = transform_spec.rsplit(":", 1)
        module_path = Path(module_path)

        if not module_path.exists():
            raise FileNotFoundError(f"Transform module not found: {module_path}")

        # Load module dynamically
        spec = importlib.util.spec_from_file_location("transform_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, func_name):
            raise AttributeError(f"Function '{func_name}' not found in {module_path}")

        return getattr(module, func_name)

    def process(self) -> list[dict]:
        """
        Process input data.

        If output already exists, loads from there (resume scenario).
        Otherwise, processes input and writes to output.

        Returns:
            List of processed items
        """
        # Check if output already exists (resume scenario)
        if self.output_path.exists():
            print(f"Loading existing preprocessed data from {self.output_path}")
            return self._load_jsonl(self.output_path)

        # Process input
        print(f"Processing input data from {self.input_path}")

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        items = []
        warnings = []

        for idx, raw_item in enumerate(self._iter_jsonl(self.input_path)):
            # Apply transform if specified
            if self.transform_fn:
                item = self.transform_fn(raw_item)
                if item is None:
                    continue  # Skip this item
            else:
                item = raw_item

            # Validate format
            validation_errors = self._validate_item(item, idx)
            if validation_errors:
                for error in validation_errors:
                    warnings.append(f"Item {idx}: {error}")
                continue

            # Check for answer in metadata
            if "answer" not in item.get("metadata", {}):
                warnings.append(
                    f"Item {idx} (id={item.get('id')}): "
                    "No 'answer' in metadata, verification may fail"
                )

            items.append(item)

        # Print warnings
        if warnings:
            print(f"⚠️  {len(warnings)} warnings during preprocessing:")
            for warning in warnings[:10]:  # Show first 10
                print(f"   {warning}")
            if len(warnings) > 10:
                print(f"   ... and {len(warnings) - 10} more")

        # Write output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(self.output_path, items)
        print(f"Wrote {len(items)} items to {self.output_path}")

        return items

    def _validate_item(self, item: dict, idx: int) -> list[str]:
        """
        Validate an item has required fields.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        missing = self.REQUIRED_FIELDS - set(item.keys())
        if missing:
            errors.append(f"Missing required fields: {missing}")
            return errors

        # Check id is string
        if not isinstance(item["id"], str):
            errors.append(f"'id' must be string, got {type(item['id']).__name__}")

        # Check messages format
        messages = item.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            errors.append("'messages' must be a non-empty list")
        else:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"messages[{i}] must be a dict")
                elif "role" not in msg or "content" not in msg:
                    errors.append(f"messages[{i}] must have 'role' and 'content'")

        # Check metadata is dict
        if not isinstance(item.get("metadata"), dict):
            errors.append("'metadata' must be a dict")

        return errors

    def _iter_jsonl(self, path: Path):
        """Iterate over JSONL file."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_jsonl(self, path: Path) -> list[dict]:
        """Load all items from JSONL file."""
        return list(self._iter_jsonl(path))

    def _write_jsonl(self, path: Path, items: list[dict]) -> None:
        """Write items to JSONL file."""
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
