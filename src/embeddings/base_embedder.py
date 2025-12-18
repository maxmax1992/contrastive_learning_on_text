from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this embedder (used as cache key)."""
        pass

    @abstractmethod
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass
