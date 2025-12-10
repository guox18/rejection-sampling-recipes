# Verifier 开发指南

本文档说明如何为项目贡献新的 Verifier。

## 目录结构

```
src/verifier/
├── __init__.py          # 导出所有 verifier
├── base.py              # BaseVerifier 抽象类
├── registry.py          # 注册器
├── mcq_rlvr.py          # MCQ 规则验证器 (R1 模型)
├── mcq_llm_judge.py     # MCQ LLM 判断器
└── README.md            # 本文档
```

## 实现步骤

### 1. 创建 Verifier 文件

```python
# src/verifier/my_verifier.py
from .base import BaseVerifier
from .registry import register_verifier


@register_verifier("my-verifier")  # 注册名称，用于配置文件
class MyVerifier(BaseVerifier):
    """
    My verifier description.

    Supports:
    - Format 1: xxx
    - Format 2: yyy
    """

    def __init__(self, **kwargs):
        """
        Initialize verifier.

        Accept **kwargs for compatibility - pipeline may pass unused config.
        """
        # 如果需要配置，从 kwargs 取出
        # self.some_option = kwargs.get("some_option", default_value)
        pass

    def verify(self, response: str, metadata: dict) -> float:
        """
        Verify response against ground truth.

        Args:
            response: Model's raw response
            metadata: Dict containing 'answer' key with ground truth

        Returns:
            Score: 1.0 for correct, 0.0 for incorrect
        """
        # 1. 提取标准答案
        ground_truth = metadata.get("answer")
        if not ground_truth:
            return 0.0

        # 2. 从 response 提取模型答案
        extracted = self.extract_answer(response)
        if not extracted:
            return 0.0

        # 3. 比较
        return 1.0 if extracted == ground_truth else 0.0

    def extract_answer(self, response: str) -> str | None:
        """Extract answer from response."""
        # 实现提取逻辑
        pass
```

### 2. 在 `__init__.py` 中导出

```python
# src/verifier/__init__.py
from .my_verifier import MyVerifier

__all__ = [
    # ... existing exports
    "MyVerifier",
]
```

### 3. 添加测试数据

在 `tests/fixtures/model_outputs.json` 中添加测试用例：

```json
{
  "my_verifier_responses": [
    {
      "id": "case_correct",
      "model_type": "xxx",
      "raw_response": "模型输出...",
      "expected": {
        "extracted_answer": "A"
      },
      "ground_truth": "A",
      "expected_score": 1.0
    },
    {
      "id": "case_wrong",
      "raw_response": "错误输出...",
      "ground_truth": "A",
      "expected_score": 0.0
    },
    {
      "id": "case_empty",
      "raw_response": "",
      "ground_truth": "A",
      "expected_score": 0.0
    }
  ]
}
```

### 4. 添加测试类

在 `tests/test_verifier.py` 中添加：

```python
class TestMyVerifier:
    """Tests for MyVerifier."""

    @pytest.fixture
    def verifier(self):
        return MyVerifier()

    def test_correct_answer(self, verifier):
        """正确答案得分 1.0"""
        score = verifier.verify("正确输出", {"answer": "A"})
        assert score == 1.0

    def test_wrong_answer(self, verifier):
        """错误答案得分 0.0"""
        score = verifier.verify("错误输出", {"answer": "A"})
        assert score == 0.0

    def test_empty_response(self, verifier):
        """空响应得分 0.0"""
        assert verifier.verify("", {"answer": "A"}) == 0.0

    def test_no_answer_in_metadata(self, verifier):
        """缺少标准答案得分 0.0"""
        assert verifier.verify("输出", {}) == 0.0

    def test_with_fixtures(self, verifier, model_outputs):
        """使用 fixtures 批量测试"""
        for item in model_outputs.get("my_verifier_responses", []):
            score = verifier.verify(
                item["raw_response"],
                {"answer": item["ground_truth"]}
            )
            assert score == item["expected_score"], f"Failed: {item['id']}"
```

## 测试要求

提交 PR 前确保：

- [ ] 至少 5 个测试用例
- [ ] 覆盖：正确/错误/空输入/缺少 metadata
- [ ] `pytest tests/test_verifier.py -v` 通过
- [ ] `ruff check src/verifier/` 通过

## 使用新 Verifier

配置文件中使用注册名称：

```yaml
verifier:
  type: my-verifier  # @register_verifier("my-verifier")
```

## 共享工具函数

如果需要处理 thinking 标签，使用共享工具：

```python
from ..utils import clip_thinking, split_response

# 去除思考过程，只保留最终答案
final_answer = clip_thinking(response)

# 分离思考和答案
thinking, answer = split_response(response)
```
