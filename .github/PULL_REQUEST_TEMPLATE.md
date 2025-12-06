## Description

Brief description of changes.

## Type of Change

- [ ] New verifier
- [ ] New formatter
- [ ] Bug fix
- [ ] Documentation

---

## Checklist for New Verifier

If adding a new verifier, please ensure:

- [ ] Inherits from `BaseVerifier`
- [ ] Uses `@register_verifier("name")` decorator
- [ ] Exported in `src/verifier/__init__.py`
- [ ] Added test fixtures in `tests/fixtures/model_outputs.json`
- [ ] Added test class in `tests/test_verifier.py` with:
  - [ ] `test_correct_answer` - correct answer scores 1.0
  - [ ] `test_wrong_answer` - wrong answer scores 0.0
  - [ ] `test_empty_response` - empty input scores 0.0
  - [ ] `test_no_answer_in_metadata` - missing metadata scores 0.0
  - [ ] At least 5 test cases total
- [ ] `ruff check` passes
- [ ] `pytest tests/` passes

## Test Results

```
paste pytest output here
```
