"""Document classifier using k-NN with TF-IDF or OpenAI embeddings."""

from .embedders import BaseEmbedder, TFIDFEmbedder, OpenAIEmbedder
from .train_knn import prepare_data, fit_model, eval

__all__ = [
    "BaseEmbedder",
    "TFIDFEmbedder",
    "OpenAIEmbedder",
    "prepare_data",
    "fit_model",
    "eval",
]
