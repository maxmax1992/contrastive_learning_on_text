"""
Test to compare embedding properties between BERT and OpenAI.
Check for normalization, dimensions, and other potential issues.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bert_embedder import BertEmbedder
from openai_utils import get_openai_embeddings, truncate_texts
from unittest.mock import patch


def test_bert_embedding_properties():
    """Test BERT embedding normalization and properties."""
    embedder = BertEmbedder()
    
    texts = ["This is a test sentence.", "Another test sentence here."]
    embeddings = embedder(texts)
    
    print("\n" + "="*80)
    print("BERT EMBEDDINGS")
    print("="*80)
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"L2 norms: {norms}")
    print(f"Are normalized (≈1.0): {np.allclose(norms, 1.0)}")
    
    # Check value ranges
    print(f"Min value: {embeddings.min()}")
    print(f"Max value: {embeddings.max()}")
    print(f"Mean value: {embeddings.mean()}")
    print(f"Std value: {embeddings.std()}")
    
    return embeddings


def test_openai_embedding_properties():
    """Test OpenAI embedding properties (with mock)."""
    
    def mock_extract_embeddings(texts, batch_size):
        """Mock OpenAI embeddings with realistic properties."""
        class MockEmb:
            def __init__(self):
                # OpenAI text-embedding-3-large has 3072 dimensions
                # and returns normalized embeddings
                emb = np.random.randn(3072)
                # Normalize
                emb = emb / np.linalg.norm(emb)
                self.embedding = emb.tolist()
        return [MockEmb() for _ in texts]
    
    with patch('openai_utils.extract_embeddings', side_effect=mock_extract_embeddings):
        texts = ["This is a test sentence.", "Another test sentence here."]
        truncated = truncate_texts(texts, max_length=256)
        embeddings = get_openai_embeddings(truncated, batch_size=2)
    
    print("\n" + "="*80)
    print("OPENAI EMBEDDINGS (MOCKED)")
    print("="*80)
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"L2 norms: {norms}")
    print(f"Are normalized (≈1.0): {np.allclose(norms, 1.0)}")
    
    # Check value ranges
    print(f"Min value: {embeddings.min()}")
    print(f"Max value: {embeddings.max()}")
    print(f"Mean value: {embeddings.mean()}")
    print(f"Std value: {embeddings.std()}")
    
    return embeddings


def test_compare_embedding_quality():
    """
    Compare embedding quality using a simple similarity test.
    """
    print("\n" + "="*80)
    print("EMBEDDING QUALITY COMPARISON")
    print("="*80)
    
    # Create test texts with known relationships
    texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",  # Similar to first
        "Python is a programming language.",  # Different topic
        "The dog ran in the park.",  # Different but similar structure
    ]
    
    embedder = BertEmbedder()
    bert_embs = embedder(texts)
    
    print("\nBERT Similarities (cosine):")
    print("Text 0 vs Text 1 (similar meaning):", np.dot(bert_embs[0], bert_embs[1]))
    print("Text 0 vs Text 2 (different topic):", np.dot(bert_embs[0], bert_embs[2]))
    print("Text 0 vs Text 3 (different content):", np.dot(bert_embs[0], bert_embs[3]))
    
    # Check if BERT can distinguish
    sim_similar = np.dot(bert_embs[0], bert_embs[1])
    sim_different = np.dot(bert_embs[0], bert_embs[2])
    
    if sim_similar > sim_different:
        print("✅ BERT correctly identifies similar texts")
    else:
        print("⚠️  BERT may have issues distinguishing texts")
        print(f"   Similar texts similarity: {sim_similar}")
        print(f"   Different texts similarity: {sim_different}")


def test_check_baseline_vs_trained_difference():
    """
    Check if there's actually a difference between baseline and trained models.
    """
    print("\n" + "="*80)
    print("CHECKING BASELINE VS TRAINED")
    print("="*80)
    
    from contrastive_nn import ContrastiveNN
    import torch
    
    model = ContrastiveNN()
    model.eval()
    
    texts = ["Test sentence one.", "Test sentence two."]
    
    # Get baseline embeddings (no projection)
    with torch.no_grad():
        baseline_embs = model.encoder(texts).cpu().numpy()
    
    # Get trained embeddings (with projection)
    with torch.no_grad():
        encoder_out = model.encoder(texts)
        projected = model.projection_head(encoder_out)
        projected_normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
        trained_embs = projected_normalized.cpu().numpy()
    
    print(f"Baseline shape: {baseline_embs.shape}")
    print(f"Trained shape: {trained_embs.shape}")
    
    # Check if projection head is actually doing anything
    # (if untrained, it might just be random)
    print(f"\nBaseline L2 norms: {np.linalg.norm(baseline_embs, axis=1)}")
    print(f"Trained L2 norms: {np.linalg.norm(trained_embs, axis=1)}")
    
    # Check similarity between the two representations
    # If they're very similar, the projection head isn't doing much
    from scipy.spatial.distance import cosine
    
    for i in range(len(texts)):
        # Can't directly compare different dimensions, but we can check
        # if the projection is just noise or actually learned
        pass
    
    print("\n⚠️  Note: If the model wasn't trained properly, the projection head")
    print("   might not improve over the baseline BERT embeddings.")


if __name__ == "__main__":
    test_bert_embedding_properties()
    test_openai_embedding_properties()
    test_compare_embedding_quality()
    test_check_baseline_vs_trained_difference()
