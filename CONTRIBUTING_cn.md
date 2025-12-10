# Contributing Guide

感谢你对项目的贡献！

## 开发环境

```bash
# 推荐使用 uv
uv sync --dev

# 或者 pip
pip install -e ".[dev]"
```

## 代码规范

- 使用 `ruff` 进行 lint 和格式化
- 代码注释、docstring、commit message 使用英文
- 推荐使用 type hints

```bash
# 检查
ruff check .
ruff format --check .

# 自动修复
ruff check --fix .
ruff format .
```

## 贡献类型

### 1. 新增 Verifier

详见 [src/verifier/README.md](src/verifier/README.md)

**核心步骤**：
1. 实现 `BaseVerifier` 接口
2. 使用 `@register_verifier("name")` 注册
3. 在 `tests/fixtures/model_outputs.json` 添加测试数据
4. 在 `tests/test_verifier.py` 添加测试类

### 2. 新增 Formatter

类似 Verifier，参考 `src/formatter/` 中的实现。

### 3. Bug 修复

1. 先创建 Issue 描述问题
2. Fork 并修复
3. 添加测试覆盖修复的场景
4. 提交 PR

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_verifier.py -v

# 带覆盖率
pytest tests/ -v --cov=src
```

## PR 提交

1. Fork 本仓库
2. 创建分支：`git checkout -b feat/my-feature`
3. 提交更改：`git commit -m "feat: add xxx"`
4. 推送：`git push origin feat/my-feature`
5. 创建 Pull Request

### Commit 规范

```
feat: 新功能
fix: 修复 bug
docs: 文档更新
test: 测试相关
refactor: 重构
```

## 项目结构

```
rejection-sampling-recipes/
├── src/
│   ├── sampler/        # 采样器 (OpenAI API, vLLM)
│   ├── verifier/       # 验证器 ← 最常贡献的模块
│   ├── formatter/      # 格式化器 (SFT, DPO)
│   ├── utils/          # 工具函数
│   └── pipeline.py     # 主流程
├── tests/
│   ├── fixtures/       # 测试数据
│   └── test_*.py       # 测试文件
├── configs/            # Hydra 配置
└── transforms/         # 数据转换函数
```

## 问题反馈

- Bug 报告：创建 Issue 并附上复现步骤
- 功能建议：创建 Issue 描述需求场景
