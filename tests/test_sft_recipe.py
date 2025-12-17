"""Tests for SFT Recipe."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recipes.sft import SFTConfig, SFTRecipe
from recipes.sft.recipe import FormatterStage, SamplerStage, VerifierStage
from src.pipeline import Pipeline


# ============================================================
# SFTConfig Tests
# ============================================================

class TestSFTConfig:
    """Tests for SFTConfig."""
    
    def test_default_values(self):
        """默认值测试."""
        config = SFTConfig()
        
        assert config.model == "gpt-4o-mini"
        assert config.n_samples == 16
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.pass_threshold == 1.0
        assert config.max_retries == 3
        assert config.semaphore_per_judger == 50
    
    def test_custom_values(self):
        """自定义值测试."""
        config = SFTConfig(
            model="gpt-4",
            n_samples=8,
            temperature=0.5,
            pass_threshold=0.8,
        )
        
        assert config.model == "gpt-4"
        assert config.n_samples == 8
        assert config.temperature == 0.5
        assert config.pass_threshold == 0.8
    
    def test_from_yaml(self, tmp_path):
        """从 YAML 加载配置."""
        yaml_content = """
model: test-model
n_samples: 4
temperature: 0.9
pass_threshold: 0.5
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)
        
        config = SFTConfig.from_yaml(str(yaml_path))
        
        assert config.model == "test-model"
        assert config.n_samples == 4
        assert config.temperature == 0.9
        assert config.pass_threshold == 0.5
    
    def test_to_yaml(self, tmp_path):
        """保存配置到 YAML."""
        config = SFTConfig(model="test-model", n_samples=8)
        yaml_path = tmp_path / "output.yaml"
        
        config.to_yaml(str(yaml_path))
        
        # 重新加载验证
        loaded = SFTConfig.from_yaml(str(yaml_path))
        assert loaded.model == "test-model"
        assert loaded.n_samples == 8


# ============================================================
# VerifierStage Tests (LLM Judge)
# ============================================================

class TestVerifierStage:
    """Tests for VerifierStage (LLM Judge)."""
    
    @pytest.fixture
    def verifier(self):
        """创建 VerifierStage."""
        config = SFTConfig()
        return VerifierStage(config)
    
    def test_verify_failed_item_passthrough(self, verifier):
        """测试失败 item 的透传."""
        # Mock OpenAI client
        verifier._openai_client = MagicMock()
        
        batch = [
            {"id": "1", "_failed": True, "_error": "Previous error"},
            {"id": "2", "responses": ["A"], "metadata": {"answer": "A"}, "messages": []},
        ]
        
        # Mock judge response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="A"))]
        verifier._openai_client.chat.completions.create.return_value = mock_response
        
        result = verifier.process(batch)
        
        assert len(result) == 2
        assert result[0]["_failed"] is True
        assert result[0]["_error"] == "Previous error"
        assert "rollouts" in result[1]
    
    def test_parse_judge_output(self, verifier):
        """测试 judge 输出解析."""
        # 正确答案
        assert verifier._parse_judge_output("A") is True
        assert verifier._parse_judge_output("a") is True
        assert verifier._parse_judge_output("correct") is True
        assert verifier._parse_judge_output("CORRECT") is True
        
        # 错误答案
        assert verifier._parse_judge_output("B") is False
        assert verifier._parse_judge_output("b") is False
        assert verifier._parse_judge_output("incorrect") is False
        assert verifier._parse_judge_output("INCORRECT") is False
        
        # 空或未知
        assert verifier._parse_judge_output("") is False
        assert verifier._parse_judge_output("maybe") is False
    
    def test_verify_with_mock_judge(self, verifier):
        """测试 LLM Judge 验证 (mock)."""
        # Mock OpenAI client
        verifier._openai_client = MagicMock()
        
        batch = [
            {
                "id": "1",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "metadata": {"answer": "4"},
                "responses": ["The answer is 4.", "The answer is 5."],
            }
        ]
        
        # Mock: 第一个正确, 第二个错误
        call_count = [0]
        def mock_create(*args, **kwargs):
            result = MagicMock()
            if call_count[0] == 0:
                result.choices = [MagicMock(message=MagicMock(content="A"))]  # correct
            else:
                result.choices = [MagicMock(message=MagicMock(content="B"))]  # incorrect
            call_count[0] += 1
            return result
        
        verifier._openai_client.chat.completions.create.side_effect = mock_create
        
        result = verifier.process(batch)
        
        assert len(result) == 1
        rollouts = result[0]["rollouts"]
        assert len(rollouts) == 2
        assert rollouts[0]["score"] == 1.0
        assert rollouts[1]["score"] == 0.0


# ============================================================
# FormatterStage Tests
# ============================================================

class TestFormatterStage:
    """Tests for FormatterStage."""
    
    @pytest.fixture
    def formatter(self):
        """创建 FormatterStage."""
        config = SFTConfig(pass_threshold=1.0)
        return FormatterStage(config)
    
    def test_format_success(self, formatter, sample_batch_with_rollouts):
        """测试格式化成功的情况."""
        result = formatter.process(sample_batch_with_rollouts)
        
        assert len(result) == 2
        
        # 检查格式化后的数据
        for item in result:
            assert "messages" in item
            assert "metadata" in item
            
            # messages 应该包含 assistant 回复
            messages = item["messages"]
            assert len(messages) == 2
            assert messages[-1]["role"] == "assistant"
            
            # metadata 应该有统计信息
            metadata = item["metadata"]
            assert "_uid" in metadata
            assert "n_passed" in metadata
            assert "n_total" in metadata
    
    def test_format_no_pass(self, formatter):
        """测试没有通过验证的情况."""
        batch = [
            {
                "id": "1",
                "messages": [{"role": "user", "content": "test"}],
                "metadata": {"answer": "A"},
                "rollouts": [
                    {"response": "wrong", "score": 0.0},
                    {"response": "also wrong", "score": 0.5},  # 小于 pass_threshold=1.0
                ],
            }
        ]
        
        result = formatter.process(batch)
        
        assert len(result) == 1
        assert result[0]["_failed"] is True
        assert "No response passed" in result[0]["_error"]
    
    def test_format_with_threshold(self):
        """测试自定义 pass_threshold."""
        config = SFTConfig(pass_threshold=0.5)
        formatter = FormatterStage(config)
        
        batch = [
            {
                "id": "1",
                "messages": [{"role": "user", "content": "test"}],
                "metadata": {},
                "rollouts": [
                    {"response": "partial", "score": 0.6},
                    {"response": "wrong", "score": 0.3},
                ],
            }
        ]
        
        result = formatter.process(batch)
        
        assert len(result) == 1
        # 成功的 item 不应该有 _failed 字段
        assert "_failed" not in result[0]
        assert result[0]["messages"][-1]["content"] == "partial"


# ============================================================
# SamplerStage Tests (Mocked)
# ============================================================

class TestSamplerStage:
    """Tests for SamplerStage (with mocked API)."""
    
    @pytest.fixture
    def sampler(self):
        """创建 SamplerStage."""
        config = SFTConfig(n_samples=2, max_retries=1)
        return SamplerStage(config)
    
    def test_is_async(self, sampler):
        """SamplerStage 应该是异步的."""
        assert sampler.is_async()
    
    @pytest.mark.asyncio
    async def test_process_with_mock(self, sampler, sample_batch):
        """测试 process 方法 (mock API)."""
        # Mock API 响应
        mock_responses = ["Response 1", "Response 2"]
        
        with patch.object(sampler, "_call_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_responses
            
            result = await sampler.process(sample_batch)
        
        assert len(result) == 2
        for item in result:
            assert "responses" in item
            assert item["responses"] == mock_responses
    
    @pytest.mark.asyncio
    async def test_process_api_error(self, sampler, sample_batch):
        """测试 API 错误处理."""
        with patch.object(sampler, "_call_api", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = RuntimeError("API Error")
            
            result = await sampler.process(sample_batch)
        
        assert len(result) == 2
        for item in result:
            assert item["_failed"] is True
            assert "API Error" in item["_error"]


# ============================================================
# SFTRecipe Tests
# ============================================================

class TestSFTRecipe:
    """Tests for SFTRecipe."""
    
    def test_recipe_stages(self):
        """测试 Recipe 返回的 stages."""
        config = SFTConfig()
        recipe = SFTRecipe(config)
        
        stages = recipe.stages()
        
        assert len(stages) == 3
        assert isinstance(stages[0], SamplerStage)
        assert isinstance(stages[1], VerifierStage)
        assert isinstance(stages[2], FormatterStage)
    
    def test_recipe_with_pipeline(self):
        """测试 Recipe 与 Pipeline 集成."""
        config = SFTConfig()
        recipe = SFTRecipe(config)
        
        pipeline = Pipeline(
            recipe=recipe,
            batch_size=10,
            concurrency=1,
            stage_concurrency={
                "SamplerStage": 5,
                "VerifierStage": 3,
                "FormatterStage": 1,
            },
        )
        
        assert pipeline.recipe == recipe
        assert len(pipeline._stages) == 3
