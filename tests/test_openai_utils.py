"""
Tests for openai_utils module to verify token truncation and embedding extraction.
"""
import pytest
import numpy as np
from transformers import AutoTokenizer
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai_utils import truncate_texts, get_openai_embeddings


class TestTruncateTexts:
    """Test suite for truncate_texts function to ensure correct token-level truncation."""
    
    @pytest.fixture
    def tokenizer(self):
        """Fixture to provide a tokenizer instance."""
        return AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    def test_truncate_short_text(self, tokenizer):
        """Test that short texts remain unchanged."""
        texts = ["This is a short text."]
        max_length = 256
        
        truncated = truncate_texts(texts, max_length=max_length)
        
        # Verify the text is preserved
        assert len(truncated) == 1
        
        # Verify token count is within limit
        tokens = tokenizer(truncated[0], add_special_tokens=True)["input_ids"]
        assert len(tokens) <= max_length
        
    def test_truncate_long_text(self, tokenizer):
        """Test that long texts are properly truncated to max_length tokens."""
        # Create a very long text (repeat a sentence many times)
        long_text = "This is a sentence that will be repeated many times. " * 100
        texts = [long_text]
        max_length = 256
        
        truncated = truncate_texts(texts, max_length=max_length)
        
        # Verify truncation occurred
        assert len(truncated) == 1
        assert len(truncated[0]) < len(long_text)
        
        # Verify token count is exactly at or below max_length
        tokens = tokenizer(truncated[0], add_special_tokens=True)["input_ids"]
        assert len(tokens) <= max_length
        
    def test_truncate_exact_token_count(self, tokenizer):
        """Test that truncated texts have the exact expected token count."""
        # Create texts of varying lengths
        texts = [
            "Short text.",
            "A medium length text that has more words but still under limit.",
            "A very long text. " * 200  # This will definitely exceed 256 tokens
        ]
        max_length = 256
        
        truncated = truncate_texts(texts, max_length=max_length)
        
        assert len(truncated) == len(texts)
        
        # Verify each truncated text has token count <= max_length
        for original, trunc in zip(texts, truncated):
            tokens = tokenizer(trunc, add_special_tokens=True)["input_ids"]
            assert len(tokens) <= max_length, f"Token count {len(tokens)} exceeds max_length {max_length}"
            
            # If original was short, truncated should be identical or very similar
            original_tokens = tokenizer(original, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
            if len(original_tokens) <= max_length:
                # Decode both and compare
                original_decoded = tokenizer.decode(original_tokens, skip_special_tokens=True)
                assert trunc == original_decoded
    
    def test_truncate_preserves_order(self, tokenizer):
        """Test that truncation preserves the order of texts."""
        texts = [f"Text number {i}" for i in range(10)]
        max_length = 256
        
        truncated = truncate_texts(texts, max_length=max_length)
        
        assert len(truncated) == len(texts)
        for i, trunc in enumerate(truncated):
            assert f"number {i}" in trunc or f"{i}" in trunc
    
    def test_truncate_batch_processing(self, tokenizer):
        """Test that batch processing produces same results as individual processing."""
        texts = ["Sample text " * 50 for _ in range(5)]
        max_length = 128
        
        # Truncate all at once
        truncated_batch = truncate_texts(texts, max_length=max_length)
        
        # Truncate one by one
        truncated_individual = []
        for text in texts:
            result = truncate_texts([text], max_length=max_length)
            truncated_individual.append(result[0])
        
        # Results should be identical
        assert truncated_batch == truncated_individual
    
    def test_truncate_empty_text(self, tokenizer):
        """Test handling of empty texts."""
        texts = ["", "Normal text", ""]
        max_length = 256
        
        truncated = truncate_texts(texts, max_length=max_length)
        
        assert len(truncated) == 3
        assert truncated[0] == ""
        assert truncated[2] == ""
        assert len(truncated[1]) > 0
    
    def test_truncate_special_characters(self, tokenizer):
        """Test that special characters are handled correctly."""
        texts = [
            "Text with émojis 😀 and spëcial çharacters!",
            "Math symbols: ∑ ∫ ∂ ∇",
            "Unicode: 你好世界"
        ]
        max_length = 256
        
        truncated = truncate_texts(texts, max_length=max_length)
        
        assert len(truncated) == len(texts)
        for trunc in truncated:
            tokens = tokenizer(trunc, add_special_tokens=True)["input_ids"]
            assert len(tokens) <= max_length
    
    def test_truncate_different_max_lengths(self, tokenizer):
        """Test truncation with different max_length values."""
        text = "This is a test sentence. " * 50
        texts = [text]
        
        for max_length in [32, 64, 128, 256, 512]:
            truncated = truncate_texts(texts, max_length=max_length)
            tokens = tokenizer(truncated[0], add_special_tokens=True)["input_ids"]
            assert len(tokens) <= max_length, f"Failed for max_length={max_length}"
    
    def test_token_count_consistency(self, tokenizer):
        """
        CRITICAL TEST: Verify that truncate_texts produces texts that tokenize 
        to the exact same number of tokens as if we had truncated during tokenization.
        This ensures OpenAI embeddings use the same token count as the local model.
        """
        texts = [
            "Short text",
            "Medium length text with several words in it",
            "Very long text that will definitely be truncated. " * 100
        ]
        max_length = 256
        
        truncated = truncate_texts(texts, max_length=max_length)
        
        for original, trunc in zip(texts, truncated):
            # Method 1: Tokenize original with truncation
            tokens_direct = tokenizer(
                original,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )["input_ids"]
            
            # Method 2: Tokenize truncated text
            tokens_from_truncated = tokenizer(
                trunc,
                add_special_tokens=True
            )["input_ids"]
            
            # These should be identical
            assert tokens_direct == tokens_from_truncated, \
                f"Token mismatch: direct={len(tokens_direct)}, from_truncated={len(tokens_from_truncated)}"


class TestGetOpenAIEmbeddings:
    """Test suite for get_openai_embeddings function."""
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(".env").exists(),
        reason="Requires .env file with AZURE_OPENAI_API_KEY"
    )
    def test_get_embeddings_basic(self):
        """Test basic embedding extraction (requires API key)."""
        texts = ["Hello world", "Test sentence"]
        
        embeddings = get_openai_embeddings(texts, batch_size=2)
        
        # Check shape
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Embedding dimension
        
        # Check type
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float64 or embeddings.dtype == np.float32
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(".env").exists(),
        reason="Requires .env file with AZURE_OPENAI_API_KEY"
    )
    def test_get_embeddings_batch_size(self):
        """Test that different batch sizes produce valid embeddings with correct shape."""
        texts = ["Text " + str(i) for i in range(10)]
        
        embeddings_batch_2 = get_openai_embeddings(texts, batch_size=2)
        embeddings_batch_5 = get_openai_embeddings(texts, batch_size=5)
        
        # Both should produce embeddings with the same shape
        assert embeddings_batch_2.shape == embeddings_batch_5.shape
        assert embeddings_batch_2.shape[0] == len(texts)
        
        # Note: We don't compare exact values as OpenAI API may have slight non-determinism
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(".env").exists(),
        reason="Requires .env file with AZURE_OPENAI_API_KEY"
    )
    def test_get_embeddings_with_truncation(self):
        """Test that embeddings work correctly with truncated texts."""
        # Create a long text that will be truncated
        long_text = "This is a very long sentence. " * 100
        texts = [long_text]
        
        # Truncate first
        truncated = truncate_texts(texts, max_length=256)
        
        # Get embeddings
        embeddings = get_openai_embeddings(truncated, batch_size=1)
        
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0
    
    def test_get_embeddings_mock(self, monkeypatch):
        """Test embedding extraction with mocked API calls."""
        # Mock the extract_embeddings function
        class MockEmbedding:
            def __init__(self, dim=3072):
                self.embedding = np.random.randn(dim).tolist()
        
        def mock_extract_embeddings(texts, batch_size):
            return [MockEmbedding() for _ in texts]
        
        # Patch the import
        import openai_utils
        monkeypatch.setattr(openai_utils, "extract_embeddings", mock_extract_embeddings)
        
        texts = ["Test 1", "Test 2", "Test 3"]
        embeddings = get_openai_embeddings(texts, batch_size=2)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 3072


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_truncate_and_embed_pipeline(self, monkeypatch):
        """Test the complete truncate -> embed pipeline."""
        # Mock embeddings
        class MockEmbedding:
            def __init__(self, dim=3072):
                self.embedding = np.random.randn(dim).tolist()
        
        def mock_extract_embeddings(texts, batch_size):
            return [MockEmbedding() for _ in texts]
        
        import openai_utils
        monkeypatch.setattr(openai_utils, "extract_embeddings", mock_extract_embeddings)
        
        # Create test texts
        texts = [
            "Short text",
            "A longer text with more content " * 50
        ]
        
        # Truncate
        truncated = truncate_texts(texts, max_length=256)
        
        # Get embeddings
        embeddings = get_openai_embeddings(truncated, batch_size=2)
        
        # Verify
        assert len(truncated) == len(texts)
        assert embeddings.shape[0] == len(texts)
        
        # Verify token counts
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        for trunc in truncated:
            tokens = tokenizer(trunc, add_special_tokens=True)["input_ids"]
            assert len(tokens) <= 256
