# Test Suite Documentation

This directory contains comprehensive tests for the newsgroups classification project, with a focus on validating token truncation correctness and evaluation pipeline integrity.

## Running Tests

### Run all tests
```bash
source .venv/bin/activate
uv run pytest tests/ -v
```

### Run specific test file
```bash
uv run pytest tests/test_openai_utils.py -v
uv run pytest tests/test_evaluate.py -v
```

### Run specific test class or function
```bash
uv run pytest tests/test_openai_utils.py::TestTruncateTexts -v
uv run pytest tests/test_openai_utils.py::TestTruncateTexts::test_token_count_consistency -v
```

## Test Files

### `test_openai_utils.py`
Tests for the `openai_utils` module, focusing on:

#### `TestTruncateTexts` - Token Truncation Tests
- **`test_truncate_short_text`**: Verifies short texts remain unchanged
- **`test_truncate_long_text`**: Ensures long texts are properly truncated
- **`test_truncate_exact_token_count`**: Validates token counts are within limits
- **`test_truncate_preserves_order`**: Checks text order is maintained
- **`test_truncate_batch_processing`**: Verifies batch and individual processing produce identical results
- **`test_truncate_empty_text`**: Tests handling of empty strings
- **`test_truncate_special_characters`**: Validates Unicode and special character handling
- **`test_truncate_different_max_lengths`**: Tests various max_length values
- **`test_token_count_consistency`** ⭐ **CRITICAL**: Verifies that:
  - `text → tokenize(truncate=True) → tokens`
  - `text → truncate_texts() → truncated_text → tokenize() → tokens`
  - Both produce **IDENTICAL** token sequences

This critical test ensures OpenAI embeddings use the exact same token count as the local BERT model.

#### `TestGetOpenAIEmbeddings` - Embedding Extraction Tests
- **`test_get_embeddings_basic`**: Tests basic embedding extraction (requires API key)
- **`test_get_embeddings_batch_size`**: Validates different batch sizes work correctly
- **`test_get_embeddings_with_truncation`**: Tests embeddings with truncated texts
- **`test_get_embeddings_mock`**: Tests with mocked API calls (no API key needed)

#### `TestIntegration` - Pipeline Integration Tests
- **`test_truncate_and_embed_pipeline`**: Tests complete truncate → embed workflow

### `test_evaluate.py`
Tests for the evaluation pipeline:

#### `TestGetEmbeddings` - Embedding Extraction from Models
- **`test_get_embeddings_with_projection`**: Tests embedding extraction with projection head
- **`test_get_embeddings_without_projection`**: Tests baseline embedding extraction
- **`test_get_embeddings_normalization`**: Verifies projected embeddings are L2-normalized

#### `TestEvaluateModel` - Model Evaluation Tests
- **`test_evaluate_model_basic`**: Tests basic model evaluation workflow
- **`test_evaluate_model_cache_loading`**: Verifies embedding caching works correctly
- **`test_evaluate_model_different_k`**: Tests evaluation with different k values for k-NN
- **`test_evaluate_openai_model`**: Tests OpenAI embedding evaluation path

#### `TestEvaluationAccuracy` - Accuracy Calculation Tests
- **`test_perfect_classification`**: Validates accuracy calculation with perfect predictions
- **`test_per_class_accuracy_calculation`**: Tests per-class accuracy computation logic

#### `TestEvaluationIntegration` - End-to-End Tests
- **`test_full_evaluation_pipeline_mock`**: Tests complete evaluation pipeline with mocked data

## Key Testing Insights

### Token Count Verification
The most critical aspect tested is ensuring that `truncate_texts()` produces texts that, when tokenized, result in the **exact same token sequence** as if we had truncated during tokenization. This is essential because:

1. Local BERT model tokenizes with `truncation=True, max_length=256`
2. OpenAI API receives the truncated text string
3. Both must process the **same tokens** for fair comparison

The `test_token_count_consistency` test validates this by comparing:
- Direct tokenization with truncation
- Tokenization of pre-truncated text

### Why This Matters
If truncation isn't done correctly, you could get:
- OpenAI processing more/fewer tokens than BERT
- Unfair accuracy comparisons
- Misleading evaluation results

The test suite ensures this doesn't happen.

## Test Coverage

- ✅ Token truncation correctness
- ✅ Token count consistency
- ✅ Embedding extraction (with mocks and real API)
- ✅ Batch processing
- ✅ Edge cases (empty texts, special characters, Unicode)
- ✅ Evaluation pipeline (embeddings, k-NN, accuracy)
- ✅ Caching mechanism
- ✅ Per-class accuracy calculations

## Dependencies

Tests require:
- `pytest>=8.0.0`
- `pytest-mock>=3.12.0`
- All project dependencies (transformers, torch, scikit-learn, etc.)

Install with:
```bash
uv pip install pytest pytest-mock
```

## Notes

- Tests that require OpenAI API access are marked with `@pytest.mark.skipif` and will be skipped if `.env` file is not present
- Mock tests provide coverage without requiring API access
- Integration tests use mocked data to avoid dependencies on actual datasets
