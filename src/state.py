"""
State Manager for checkpoint/resume support.

Tracks progress and allows resuming from the last checkpoint.
"""

import json
from datetime import datetime
from pathlib import Path


class StateManager:
    """Manages pipeline state for checkpoint/resume."""

    def __init__(self, work_dir: str | Path):
        """
        Initialize state manager.

        Args:
            work_dir: Working directory for the experiment
        """
        self.work_dir = Path(work_dir)
        self.state_file = self.work_dir / "state.json"
        self.state = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load existing state or create new one."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                print(f"Loaded existing state from {self.state_file}")
                return state

        # Create new state
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_shards": [],
            "current_shard": None,
            "total_items": 0,
            "processed_items": 0,
            "status": "initialized",
        }

    def save(self) -> None:
        """Save state to file."""
        self.state["updated_at"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def mark_shard_started(self, shard_idx: int) -> None:
        """Mark a shard as started."""
        self.state["current_shard"] = shard_idx
        self.state["status"] = "running"
        self.save()

    def mark_shard_completed(self, shard_idx: int, num_items: int) -> None:
        """Mark a shard as completed."""
        if shard_idx not in self.state["completed_shards"]:
            self.state["completed_shards"].append(shard_idx)
            self.state["processed_items"] += num_items
        self.state["current_shard"] = None
        self.save()

    def is_shard_completed(self, shard_idx: int) -> bool:
        """Check if a shard has been completed."""
        return shard_idx in self.state["completed_shards"]

    def set_total_items(self, total: int) -> None:
        """Set total number of items."""
        self.state["total_items"] = total
        self.save()

    def mark_completed(self) -> None:
        """Mark the entire pipeline as completed."""
        self.state["status"] = "completed"
        self.state["completed_at"] = datetime.now().isoformat()
        self.save()

    def mark_failed(self, error: str) -> None:
        """Mark the pipeline as failed."""
        self.state["status"] = "failed"
        self.state["error"] = error
        self.save()

    @property
    def completed_shards(self) -> list[int]:
        """Get list of completed shard indices."""
        return self.state["completed_shards"]

    @property
    def is_completed(self) -> bool:
        """Check if the pipeline is completed."""
        return self.state["status"] == "completed"

    def get_progress(self) -> str:
        """Get human-readable progress string."""
        processed = self.state["processed_items"]
        total = self.state["total_items"]
        pct = (processed / total * 100) if total > 0 else 0
        return f"{processed}/{total} items ({pct:.1f}%)"
