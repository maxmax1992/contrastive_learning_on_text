"""
Test to verify token count consistency between OpenAI and BERT embeddings.
This test checks for the bug where BERT might be using more tokens than OpenAI.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from openai_utils import truncate_texts


def test_bert_vs_openai_token_count():
    """
    CRITICAL BUG TEST: Verify that BERT and OpenAI use the same token count.
    
    The issue is:
    - OpenAI receives: truncate_texts(text) → decoded text WITHOUT special tokens
    - BERT receives: tokenizer(text, add_special_tokens=True) → includes [CLS] and [SEP]
    
    This means BERT can use up to 258 tokens (256 + [CLS] + [SEP]) while OpenAI uses only 256!
    """
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create a long text
    long_text = "This is a test sentence. " * 100
    
    # Method 1: What OpenAI receives (truncated text)
    truncated = truncate_texts([long_text], max_length=256)[0]
    
    # Tokenize the truncated text (what OpenAI would see)
    # OpenAI doesn't add special tokens, it just sees the raw text
    openai_tokens = tokenizer(truncated, add_special_tokens=False)["input_ids"]
    
    # Method 2: What BERT receives (original text with truncation during tokenization)
    bert_tokens_with_special = tokenizer(
        long_text,
        truncation=True,
        max_length=256,
        add_special_tokens=True
    )["input_ids"]
    
    # Method 3: What BERT actually processes in BertEmbedder
    # BertEmbedder uses: tokenizer(texts, truncation=True, max_length=256, add_special_tokens=True by default)
    bert_actual_tokens = tokenizer(
        long_text,
        truncation=True,
        max_length=256,
        add_special_tokens=True  # This is the default
    )["input_ids"]
    
    print(f"\nToken counts:")
    print(f"OpenAI (truncated text, no special tokens): {len(openai_tokens)}")
    print(f"BERT (with special tokens, max_length=256): {len(bert_tokens_with_special)}")
    print(f"BERT actual (what BertEmbedder uses): {len(bert_actual_tokens)}")
    
    # The bug: truncate_texts decodes with skip_special_tokens=True
    # So OpenAI gets text that was truncated to 256 tokens INCLUDING special tokens,
    # but then the special tokens are removed, leaving ~254 tokens
    # Meanwhile, BERT processes the original text with max_length=256 INCLUDING special tokens
    
    # Let's verify this
    truncated_with_special = tokenizer(
        truncated,
        add_special_tokens=True
    )["input_ids"]
    
    print(f"Truncated text when re-tokenized with special tokens: {len(truncated_with_special)}")
    
    # This should be <= 256 because truncate_texts truncated to 256 then removed special tokens
    assert len(truncated_with_special) <= 256, \
        f"Truncated text has {len(truncated_with_special)} tokens when re-tokenized with special tokens"
    
    # But BERT might be using the full 256 tokens on the original text
    # This is the bug!
    if len(bert_actual_tokens) > len(truncated_with_special):
        print(f"\n⚠️  BUG DETECTED: BERT uses {len(bert_actual_tokens)} tokens, "
              f"but OpenAI only sees {len(openai_tokens)} tokens of content!")
        print(f"Difference: {len(bert_actual_tokens) - len(truncated_with_special)} tokens")


def test_verify_openai_receives_less_information():
    """
    Test to verify that OpenAI might be receiving LESS information than BERT.
    """
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Create text that's exactly at the boundary
    # We want text that tokenizes to exactly 256 tokens with special tokens
    base_text = "word " * 300  # Create a long text
    
    # Tokenize to exactly 256 tokens including special tokens
    tokens_with_special = tokenizer(
        base_text,
        truncation=True,
        max_length=256,
        add_special_tokens=True
    )["input_ids"]
    
    # Now decode this (simulating what truncate_texts does)
    decoded = tokenizer.decode(tokens_with_special, skip_special_tokens=True)
    
    # Re-tokenize the decoded text
    reencoded_no_special = tokenizer(decoded, add_special_tokens=False)["input_ids"]
    reencoded_with_special = tokenizer(decoded, add_special_tokens=True)["input_ids"]
    
    print(f"\nOriginal tokens (with special): {len(tokens_with_special)}")
    print(f"After decode->encode (no special): {len(reencoded_no_special)}")
    print(f"After decode->encode (with special): {len(reencoded_with_special)}")
    
    # The issue: when we decode with skip_special_tokens=True and re-encode,
    # we lose the special tokens from the count
    # So if original was 256 tokens (including [CLS] and [SEP]),
    # the decoded text only represents 254 tokens of actual content
    
    assert len(reencoded_with_special) <= 256
    assert len(reencoded_no_special) <= 254  # Lost 2 special tokens
    
    print(f"\n⚠️  OpenAI receives only {len(reencoded_no_special)} tokens of content")
    print(f"   while BERT processes {len(tokens_with_special)} tokens total")


if __name__ == "__main__":
    test_bert_vs_openai_token_count()
    test_verify_openai_receives_less_information()
