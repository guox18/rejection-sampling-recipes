#!/usr/bin/env python3
"""
[DEV TOOL] Verifier Robustness Tester

This is a DEVELOPER tool for testing the robustness of rule-based verifiers.

Problem:
    Different models output answers in various formats (\\boxed{A}, \\boxed{\\textbf{A}},
    \\boxed{\\mathrm{A}}, etc.). Rule-based verifiers may miss some edge cases.

Solution:
    This tool compares rule-based verifier (mcq-rlvr) against LLM judge (mcq-llm-as-judge)
    to identify inconsistencies. Inconsistent cases are flagged for human review.

Workflow:
    1. Sample questions from a dataset
    2. Generate model responses (calls API)
    3. Compare rule-based vs LLM judge results
    4. Report inconsistencies for manual review
    5. Update rule-based verifier if needed

Setup:
    cp configs/dev/test_verifier.example.yaml configs/dev/test_verifier.yaml
    # Edit test_verifier.yaml with your actual API keys

Usage:
   uv run python scripts/dev_test_mcq_verifier.py --config configs/dev/test_verifier.yaml -i data/Nemotron-Post-Training-Dataset-v2_formatted/datasets/train_30.jsonl

Output:
    - Console: Summary + inconsistent samples for review
    - JSON: Full results saved to tests/fixtures/collected/
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from openai import OpenAI

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.verifier import MCQLLMJudgeVerifier, MCQRLVRVerifier, VerifierTestCollector  # noqa: E402

# =============================================================================
# Banner
# =============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🔧 VERIFIER ROBUSTNESS TESTER                            ║
║                                                                              ║
║  Developer tool for testing rule-based verifier against LLM judge.          ║
║  Inconsistent cases will be flagged for human review.                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Default config path
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dev" / "test_verifier.yaml"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a model endpoint."""

    name: str
    base_url: str
    model_id: str
    api_key: str = "dummy"
    temperature: float = 0.6
    max_tokens: int = 4096
    verify_ssl: bool = True


@dataclass
class Config:
    """Full configuration for the test tool."""

    # Sampling
    sample_size: int = 10
    seed: int = 42

    # Models
    models: dict[str, ModelConfig] = None
    default_model: str = None
    test_models: list[str] = None  # List of models to test (if not specified via CLI)

    # LLM Judge
    judge_base_url: str = None
    judge_model: str = None
    judge_api_key: str = "dummy"

    # Paths
    output_dir: Path = None
    results_file: str = "verifier_test_results.json"
    cache_file: str = "llm_judge_cache.json"

    def __post_init__(self):
        if self.models is None:
            self.models = {}
        if self.test_models is None:
            self.test_models = []
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "tests" / "fixtures" / "collected"

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.results_file

    @property
    def cache_path(self) -> Path:
        return self.output_dir / self.cache_file


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("   Please copy the example config:")
        print(f"   cp {config_path.with_suffix('.example.yaml')} {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Parse models
    models = {}
    for key, model_cfg in raw.get("models", {}).items():
        models[key] = ModelConfig(
            name=model_cfg.get("name", key),
            base_url=model_cfg["base_url"],
            model_id=model_cfg["model_id"],
            api_key=model_cfg.get("api_key", "dummy"),
            temperature=model_cfg.get("temperature", 0.6),
            max_tokens=model_cfg.get("max_tokens", 4096),
            verify_ssl=model_cfg.get("verify_ssl", True),
        )

    # Parse paths
    paths = raw.get("paths", {})
    output_dir = paths.get("output_dir", "tests/fixtures/collected")
    if not Path(output_dir).is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Parse judge config
    judge = raw.get("llm_judge", {})

    return Config(
        sample_size=raw.get("sampling", {}).get("size", 10),
        seed=raw.get("sampling", {}).get("seed", 42),
        models=models,
        default_model=raw.get("default_model"),
        test_models=raw.get("test_models", []),
        judge_base_url=judge.get("base_url"),
        judge_model=judge.get("model"),
        judge_api_key=judge.get("api_key", "dummy"),
        output_dir=Path(output_dir),
        results_file=paths.get("results_file", "verifier_test_results.json"),
        cache_file=paths.get("cache_file", "llm_judge_cache.json"),
    )


# =============================================================================
# Data Loading
# =============================================================================


def load_questions_from_jsonl(
    path: Path, sample_size: int | None = None, seed: int = 42
) -> list[dict]:
    """
    Load questions from jsonl file.

    Expected format per line:
        {
            "id": "...",
            "messages": [{"role": "user", "content": "..."}],
            "metadata": {"answer": "A", ...}
        }

    Args:
        path: Path to jsonl file
        sample_size: If set, randomly sample this many questions
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: id, question, answer
    """
    questions = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # Extract question from messages
            question = ""
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                    break

            # Extract answer from metadata
            metadata = item.get("metadata", {})
            answer = metadata.get("answer", "")

            if question and answer:
                questions.append(
                    {
                        "id": item.get("id", f"sample_{len(questions)}"),
                        "question": question,
                        "answer": answer,
                    }
                )

    print(f"📂 Loaded {len(questions)} questions from {path}")

    # Sample if requested
    if sample_size and sample_size < len(questions):
        random.seed(seed)
        questions = random.sample(questions, sample_size)
        print(f"🎲 Sampled {sample_size} questions (seed={seed})")

    return questions


# =============================================================================
# Data Generation
# =============================================================================


def create_client(config: ModelConfig) -> OpenAI:
    """Create OpenAI client for a model config."""
    if config.verify_ssl:
        return OpenAI(base_url=config.base_url, api_key=config.api_key)
    else:
        return OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            http_client=httpx.Client(verify=False),
        )


def generate_response(client: OpenAI, config: ModelConfig, question: str) -> str:
    """Generate a response from a model."""
    # Add instruction to use \boxed{} format
    prompt = question
    if "\\boxed{}" not in question.lower():
        prompt = question + "\n\nPlease put your final answer within \\boxed{}."

    response = client.chat.completions.create(
        model=config.model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    return response.choices[0].message.content or ""


def collect_responses(
    model_config: ModelConfig,
    questions: list[dict],
) -> list[dict]:
    """Collect responses from a model for given questions."""
    print(f"\n🤖 Generating responses from {model_config.name}...")

    client = create_client(model_config)
    samples = []

    for i, q in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] {q['id'][:20]}...", end=" ", flush=True)
        try:
            response = generate_response(client, model_config, q["question"])
            samples.append(
                {
                    "id": f"{q['id']}_{model_config.name}",
                    "question": q["question"],
                    "answer": q["answer"],
                    "response": response,
                    "model": model_config.name,
                }
            )
            print("✓")
        except Exception as e:
            print(f"❌ {e}")

    return samples


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="[DEV TOOL] Test rule-based verifier robustness against LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage: sample 10 questions, use default model
    python scripts/dev_test_mcq_verifier.py -i data/.../train_5.jsonl

    # Test with specific model
    python scripts/dev_test_mcq_verifier.py -i data/.../train_5.jsonl -n 50 -m deepseek-r1

    # Test multiple models at once
    python scripts/dev_test_mcq_verifier.py -i data/.../train_5.jsonl -m qwen,deepseek-r1,deepseek-v3

    # Use custom config
    python scripts/dev_test_mcq_verifier.py -i data/.../train_5.jsonl --config my_config.yaml

What to do with results:
    1. Check consistency rate
    2. Review inconsistent samples manually
    3. If rule-based missed valid formats, update MCQRLVRVerifier regex
    4. Add regression tests to tests/test_verifier.py
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input jsonl file with questions (messages + metadata.answer format)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Config file path (default: {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--sample",
        "-n",
        type=int,
        help="Number of questions to sample (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for sampling (overrides config)",
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        help="Model(s) to test, comma-separated (e.g., qwen,deepseek-r1). Must be defined in config.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for results (overrides config)",
    )

    args = parser.parse_args()

    # Print banner
    print(BANNER)

    # Load config
    config = load_config(args.config)
    print(f"📄 Loaded config from {args.config}")

    # Apply CLI overrides
    sample_size = args.sample if args.sample is not None else config.sample_size
    seed = args.seed if args.seed is not None else config.seed
    output_path = args.output if args.output else config.output_path

    # Validate input
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    # Load questions
    questions = load_questions_from_jsonl(args.input, sample_size=sample_size, seed=seed)

    if not questions:
        print("❌ No valid questions found")
        sys.exit(1)

    # Select models to test
    model_names = []
    if args.models:
        # CLI: comma-separated model names
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    elif config.test_models:
        # Config: test_models list
        model_names = config.test_models
    elif config.default_model and config.default_model in config.models:
        # Fallback: default_model
        model_names = [config.default_model]
        print(f"ℹ️  Using default model: {config.default_model}")
    elif config.models:
        # Fallback: first available model
        model_names = [list(config.models.keys())[0]]
        print(f"ℹ️  Using first available model: {model_names[0]}")

    if not model_names:
        print("❌ No models defined in config")
        sys.exit(1)

    # Validate model names
    for name in model_names:
        if name not in config.models:
            print(f"❌ Model '{name}' not found in config")
            print(f"   Available models: {list(config.models.keys())}")
            sys.exit(1)

    print(f"🎯 Testing {len(model_names)} model(s): {', '.join(model_names)}")

    # Collect responses from all models
    all_samples = []
    for model_name in model_names:
        model_config = config.models[model_name]
        samples = collect_responses(model_config, questions)
        all_samples.extend(samples)

    if not all_samples:
        print("❌ No samples collected")
        return

    print(f"\n📊 Collected {len(all_samples)} samples from {len(model_names)} model(s)")

    # Initialize collector
    print("\n🔍 Comparing verifiers...")
    print("   Rule-based: MCQRLVRVerifier (fast, regex-based)")
    print(f"   LLM Judge:  {config.judge_model} @ {config.judge_base_url}")
    print(f"   Cache:      {config.cache_path}")

    collector = VerifierTestCollector(
        rule_verifier=MCQRLVRVerifier(),
        llm_verifier=MCQLLMJudgeVerifier(
            base_url=config.judge_base_url,
            model=config.judge_model,
            api_key=config.judge_api_key,
        ),
        cache_path=config.cache_path,
    )

    # Compare
    results = collector.compare_batch(all_samples, response_key="response", id_key="id")

    # Report
    print("\n" + collector.report(results))

    # Save
    collector.save_results(results, output_path)

    # Final guidance
    inconsistent_count = sum(1 for r in results if not r.is_consistent)
    if inconsistent_count > 0:
        print("\n" + "=" * 60)
        print("💡 NEXT STEPS")
        print("=" * 60)
        print("1. Review inconsistent samples above")
        print("2. If rule-based verifier missed valid formats:")
        print("   → Update regex in src/verifier/mcq_rlvr.py")
        print("3. Add regression tests to tests/test_verifier.py")
        print("=" * 60)


if __name__ == "__main__":
    main()
