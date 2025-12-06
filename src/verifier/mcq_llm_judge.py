"""
MCQ LLM-as-Judge Verifier.

Uses an LLM to judge whether the model's answer is correct.
Clips the thinking process before sending to judge.
"""

from openai import OpenAI

from ..utils.response_processor import clip_thinking
from .base import BaseVerifier
from .registry import register_verifier

# Default judge prompt template
DEFAULT_JUDGE_TEMPLATE = """You are a helpful assistant who evaluates the correctness of models' outputs.
Please judge whether the candidate's answer matches the standard answer.

Evaluation criteria:
1. The standard answer is definitely correct. You only need to judge if the candidate's answer matches it.
2. Answers may be expressed differently but mean the same thing.
3. For multiple choice questions, the candidate needs to select the correct option(s).
4. Ignore formatting differences like \\boxed{{}}.

Grade the answer as:
A: CORRECT
B: INCORRECT

Just return "A" or "B", nothing else.

<Question>
{question}
</Question>

<Standard Answer>
{gold_target}
</Standard Answer>

<Candidate's Answer>
{predicted_answer}
</Candidate's Answer>

Your judgment (A or B):"""


@register_verifier("mcq-llm-as-judge")
class MCQLLMJudgeVerifier(BaseVerifier):
    """
    MCQ verifier using LLM as judge.

    Clips the thinking process (e.g., <think>...</think>) before judging.
    More reliable than rule-based extraction for non-R1 models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 10,
        template: str | None = None,
        **kwargs,  # Accept and ignore unused kwargs for compatibility
    ):
        """
        Initialize LLM judge.

        Args:
            model: Judge model name
            base_url: API base URL (uses OPENAI_BASE_URL env var if None)
            api_key: API key (uses OPENAI_API_KEY env var if None)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Max tokens for judge response
            template: Custom judge prompt template
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.template = template or DEFAULT_JUDGE_TEMPLATE

        # Initialize OpenAI client (use dummy API key as default for local APIs)
        import os

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy"

        self.client = OpenAI(
            base_url=base_url,
            api_key=resolved_api_key,
        )

    def verify(self, response: str, metadata: dict) -> float:
        """
        Verify MCQ answer using LLM judge.

        Args:
            response: Model's response string
            metadata: Should contain:
                - 'answer' or 'gold_target': correct answer
                - 'question' or 'origin_question' (optional): original question

        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        # Extract ground truth and question
        gold_target = metadata.get("answer") or metadata.get("gold_target", "")
        question = metadata.get("question") or metadata.get("origin_question", "")

        # Clip thinking process (use shared utility)
        clipped_response = clip_thinking(response)

        # Build judge prompt
        prompt = self.template.format(
            question=question,
            gold_target=gold_target,
            predicted_answer=clipped_response,
        )

        # Call judge model
        try:
            judge_output = self._call_judge(prompt)
            is_correct = self._parse_judge_output(judge_output)
            return 1.0 if is_correct else 0.0
        except Exception as e:
            # Log error but return 0.0 (fail-safe)
            print(f"⚠️  LLM Judge error: {e}")
            return 0.0

    def _call_judge(self, prompt: str) -> str:
        """
        Call the judge model.

        Args:
            prompt: Complete judge prompt

        Returns:
            Judge model's response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def _parse_judge_output(self, output: str) -> bool:
        """
        Parse judge output to boolean.

        Args:
            output: Judge model's output

        Returns:
            True if correct, False otherwise
        """
        if not output:
            return False

        cleaned = output.strip().lower()

        # Check for correct indicators
        if cleaned in ("a", "correct"):
            return True
        elif cleaned in ("b", "incorrect"):
            return False
        else:
            # Unknown format, default to incorrect
            return False
