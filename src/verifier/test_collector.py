"""
Verifier Test Collector.

Collects model outputs and compares rule-based verifier with LLM judge
to validate coverage. Designed for MCQ verification.

Usage:
    from src.verifier import VerifierTestCollector, MCQRLVRVerifier, MCQLLMJudgeVerifier

    collector = VerifierTestCollector(
        rule_verifier=MCQRLVRVerifier(),
        llm_verifier=MCQLLMJudgeVerifier(base_url="...", model="..."),
    )
    results = collector.compare_batch(samples)
    collector.report(results)
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .base import BaseVerifier


@dataclass
class ComparisonResult:
    """Result of comparing rule-based and LLM-based verification."""

    sample_id: str
    response: str
    ground_truth: str
    question: str
    # Rule-based results
    rule_score: float
    rule_extracted: str | None
    # LLM judge results
    llm_score: float
    # Comparison
    is_consistent: bool
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class VerifierTestCollector:
    """
    Collects and compares rule-based vs LLM-based verification results.

    Supports caching LLM judge results to avoid redundant API calls.
    """

    def __init__(
        self,
        rule_verifier: BaseVerifier,
        llm_verifier: BaseVerifier,
        cache_path: Path | str | None = None,
    ):
        """
        Initialize collector.

        Args:
            rule_verifier: Rule-based verifier (e.g., MCQRLVRVerifier)
            llm_verifier: LLM-based verifier (e.g., MCQLLMJudgeVerifier)
            cache_path: Optional path to cache LLM judge results.
                        If provided, results are persisted and reused.
        """
        self.rule_verifier = rule_verifier
        self.llm_verifier = llm_verifier

        # Cache for LLM judge results: {hash: score}
        self._cache: dict[str, float] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache_dirty = False

        if self._cache_path and self._cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        """Load LLM judge cache from disk."""
        try:
            with open(self._cache_path) as f:
                self._cache = json.load(f)
            print(f"📂 Loaded {len(self._cache)} cached LLM judge results")
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save LLM judge cache to disk."""
        if not self._cache_path or not self._cache_dirty:
            return

        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved {len(self._cache)} LLM judge results to cache")
            self._cache_dirty = False
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")

    def _cache_key(self, response: str, metadata: dict) -> str:
        """Generate cache key for a response + metadata pair."""
        # Include question and answer in the hash for uniqueness
        content = json.dumps(
            {
                "response": response,
                "question": metadata.get("question", ""),
                "answer": metadata.get("answer", ""),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_llm_score(self, response: str, metadata: dict) -> float:
        """
        Get LLM judge score, using cache if available.

        Args:
            response: Model response
            metadata: Metadata dict

        Returns:
            LLM judge score (0.0 or 1.0)
        """
        cache_key = self._cache_key(response, metadata)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Call LLM judge
        score = self.llm_verifier.verify(response, metadata)

        # Cache result
        self._cache[cache_key] = score
        self._cache_dirty = True

        return score

    def compare_single(
        self,
        response: str,
        metadata: dict,
        sample_id: str = "unknown",
    ) -> ComparisonResult:
        """
        Compare rule-based and LLM-based verification for a single sample.

        Args:
            response: Model's response string
            metadata: Dict with 'answer' (required), 'question' (optional)
            sample_id: Identifier for this sample

        Returns:
            ComparisonResult with both verification results
        """
        try:
            # Rule-based verification
            rule_extracted = self.rule_verifier.extract_answer(response)
            rule_score = self.rule_verifier.verify(response, metadata)

            # LLM judge verification (with caching)
            llm_score = self._get_llm_score(response, metadata)

            # Check consistency
            is_consistent = rule_score == llm_score

            return ComparisonResult(
                sample_id=sample_id,
                response=response,
                ground_truth=metadata.get("answer", ""),
                question=metadata.get("question", ""),
                rule_score=rule_score,
                rule_extracted=rule_extracted,
                llm_score=llm_score,
                is_consistent=is_consistent,
            )

        except Exception as e:
            return ComparisonResult(
                sample_id=sample_id,
                response=response,
                ground_truth=metadata.get("answer", ""),
                question=metadata.get("question", ""),
                rule_score=0.0,
                rule_extracted=None,
                llm_score=0.0,
                is_consistent=True,  # Both failed
                error=str(e),
            )

    def compare_batch(
        self,
        samples: list[dict],
        response_key: str = "response",
        id_key: str = "id",
        verbose: bool = True,
    ) -> list[ComparisonResult]:
        """
        Compare verification results for a batch of samples.

        Args:
            samples: List of dicts, each containing:
                - response_key: Model's response
                - 'answer': Ground truth answer
                - 'question': Original question (optional)
                - id_key: Sample identifier (optional)
            response_key: Key for response in sample dict
            id_key: Key for sample ID in sample dict
            verbose: Whether to print progress

        Returns:
            List of ComparisonResult
        """
        results = []

        for i, sample in enumerate(samples):
            sample_id = sample.get(id_key, f"sample_{i}")

            if verbose:
                print(f"  [{i + 1}/{len(samples)}] {sample_id}...", end=" ", flush=True)

            response = sample.get(response_key, "")
            metadata = {
                "answer": sample.get("answer", ""),
                "question": sample.get("question", ""),
            }

            result = self.compare_single(response, metadata, sample_id)
            results.append(result)

            if verbose:
                if result.error:
                    print(f"❌ Error: {result.error[:50]}...")
                elif result.is_consistent:
                    status = "✓" if result.rule_score == 1.0 else "✗"
                    print(f"✓ ({result.rule_extracted}) [{status}]")
                else:
                    print(f"⚠️  INCONSISTENT! rule={result.rule_score} vs llm={result.llm_score}")

        # Save cache after batch
        self._save_cache()

        return results

    def report(self, results: list[ComparisonResult], max_response_len: int = 500) -> str:
        """
        Generate a human-readable report.

        Args:
            results: List of ComparisonResult
            max_response_len: Max length of response to show in inconsistent samples

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("VERIFIER COMPARISON REPORT")
        lines.append("=" * 60)

        # Summary stats
        total = len(results)
        errors = sum(1 for r in results if r.error)
        successful = total - errors
        consistent = sum(1 for r in results if r.is_consistent and not r.error)
        inconsistent = sum(1 for r in results if not r.is_consistent and not r.error)

        rule_correct = sum(1 for r in results if r.rule_score == 1.0 and not r.error)
        llm_correct = sum(1 for r in results if r.llm_score == 1.0 and not r.error)

        lines.append(f"\nTotal samples: {total}")
        lines.append(f"Errors: {errors}")
        lines.append(f"Successful: {successful}")
        lines.append(f"  - Consistent: {consistent}")
        lines.append(f"  - Inconsistent: {inconsistent}")

        if successful > 0:
            lines.append("\nAccuracy:")
            lines.append(
                f"  - Rule-based: {rule_correct}/{successful} ({100 * rule_correct / successful:.1f}%)"
            )
            lines.append(
                f"  - LLM Judge:  {llm_correct}/{successful} ({100 * llm_correct / successful:.1f}%)"
            )

            # Consistency rate
            lines.append(
                f"\nConsistency rate: {consistent}/{successful} ({100 * consistent / successful:.1f}%)"
            )

        # Inconsistent samples for review
        inconsistent_results = [r for r in results if not r.is_consistent and not r.error]
        if inconsistent_results:
            lines.append("\n" + "=" * 60)
            lines.append("INCONSISTENT SAMPLES (for manual review)")
            lines.append("=" * 60)

            for r in inconsistent_results:
                lines.append(f"\n--- {r.sample_id} ---")
                lines.append(f"Ground Truth: {r.ground_truth}")
                lines.append(f"Rule: extracted='{r.rule_extracted}', score={r.rule_score}")
                lines.append(f"LLM:  score={r.llm_score}")
                lines.append(
                    f"Question: {r.question[:200]}..."
                    if len(r.question) > 200
                    else f"Question: {r.question}"
                )

                response_display = r.response[:max_response_len]
                if len(r.response) > max_response_len:
                    response_display += "..."
                lines.append(f"Response:\n{response_display}")

        return "\n".join(lines)

    def save_results(self, results: list[ComparisonResult], path: Path | str) -> None:
        """
        Save results to JSON file.

        Args:
            results: List of ComparisonResult
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "collected_at": datetime.now().isoformat(),
            "total_samples": len(results),
            "consistent_count": sum(1 for r in results if r.is_consistent),
            "inconsistent_count": sum(1 for r in results if not r.is_consistent),
            "results": [r.to_dict() for r in results],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(results)} results to {path}")
