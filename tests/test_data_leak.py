"""
Test to check if OpenAI might be receiving UNTRUNCATED text (data leak).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai_utils import truncate_texts, get_openai_embeddings
from unittest.mock import patch, MagicMock


def test_openai_receives_truncated_text():
    """
    Test to verify that OpenAI actually receives truncated text, not the original.
    """
    # Create a very long text
    texts = [f"This is sentence number {i}. " * 200 for i in range(5)]
    
    # Track what text is actually sent to OpenAI
    captured_texts = []
    
    def mock_extract_embeddings(texts_received, batch_size):
        """Mock that captures what texts OpenAI receives."""
        captured_texts.extend(texts_received)
        # Return mock embeddings
        class MockEmb:
            def __init__(self):
                import numpy as np
                self.embedding = np.random.randn(3072).tolist()
        return [MockEmb() for _ in texts_received]
    
    # Patch the extract_embeddings function
    with patch('openai_utils.extract_embeddings', side_effect=mock_extract_embeddings):
        # This is what evaluate.py does
        truncated = truncate_texts(texts, max_length=256)
        embeddings = get_openai_embeddings(truncated, batch_size=2)
    
    print(f"\nNumber of texts: {len(texts)}")
    print(f"Number of truncated texts: {len(truncated)}")
    print(f"Number of texts sent to OpenAI: {len(captured_texts)}")
    
    # Check lengths
    for i, (original, trunc, sent) in enumerate(zip(texts, truncated, captured_texts)):
        print(f"\nText {i}:")
        print(f"  Original length: {len(original)} chars")
        print(f"  Truncated length: {len(trunc)} chars")
        print(f"  Sent to OpenAI length: {len(sent)} chars")
        print(f"  Truncated == Sent: {trunc == sent}")
        
        assert trunc == sent, f"Text {i}: Truncated text doesn't match what was sent to OpenAI!"
        assert len(trunc) < len(original), f"Text {i}: Truncated text is not shorter than original!"


def test_check_evaluate_py_flow():
    """
    Simulate exactly what evaluate.py does to check for data leaks.
    """
    print("\n" + "="*80)
    print("SIMULATING evaluate.py FLOW")
    print("="*80)
    
    # Simulate the test data
    test_texts = [
        "Short text",
        "A medium length text with some content",
        "A very long text that will definitely need truncation. " * 50
    ]
    
    print(f"\nOriginal text lengths (chars):")
    for i, text in enumerate(test_texts):
        print(f"  Text {i}: {len(text)} chars")
    
    # What evaluate.py does for OpenAI:
    print("\n--- Step 1: Truncate texts ---")
    truncated = truncate_texts(test_texts, max_length=256)
    
    print(f"\nTruncated text lengths (chars):")
    for i, text in enumerate(truncated):
        print(f"  Text {i}: {len(text)} chars")
    
    # Verify truncation happened
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    print(f"\nToken counts after truncation:")
    for i, text in enumerate(truncated):
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        print(f"  Text {i}: {len(tokens)} tokens (with special tokens)")
        assert len(tokens) <= 256, f"Text {i} has {len(tokens)} tokens, exceeds 256!"
    
    print("\n✅ All truncated texts have <= 256 tokens")
    print("✅ OpenAI receives properly truncated text")


if __name__ == "__main__":
    test_openai_receives_truncated_text()
    test_check_evaluate_py_flow()
