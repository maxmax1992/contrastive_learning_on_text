import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_embedder import BaseEmbedder


class TFIDFEmbedder(BaseEmbedder):
    """Embedder using TF-IDF vectorization."""

    def __init__(self, max_text_size: int | None = None):
        """
        Args:
            max_text_size: Maximum number of characters to use from each text.
                          If None, uses full text.
        """
        self.max_text_size = max_text_size
        self.vectorizer = TfidfVectorizer()
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "TF_IDF"

    def _truncate_texts(self, texts: list[str]) -> list[str]:
        """Truncate texts to max_text_size characters if specified."""
        if self.max_text_size is None:
            return texts
        return [text[:self.max_text_size] for text in texts]

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF vectorizer on training texts."""
        truncated = self._truncate_texts(texts)
        print(f"Fitting TF-IDF vectorizer on {len(truncated)} texts...")
        self.vectorizer.fit(truncated)
        self._is_fitted = True

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.

        Note: vectorizer must be fitted first via fit() or will auto-fit on first call.
        """
        truncated = self._truncate_texts(texts)
        print(f"Extracting TF-IDF embeddings for {len(truncated)} texts...")

        if self._is_fitted:
            embeddings_data = self.vectorizer.transform(truncated)
        else:
            embeddings_data = self.vectorizer.fit_transform(truncated)
            self._is_fitted = True

        vectors = np.asarray(embeddings_data.toarray(), dtype=np.float32)
        return vectors


if __name__ == "__main__":
    embedder = TFIDFEmbedder(max_text_size=500)
    texts = ["This is a test", "This is another test"]
    embedder.fit(texts)
    embeddings = embedder.embed(texts)
    print(f"Embedding shape: {embeddings.shape}")
    print(embeddings)
