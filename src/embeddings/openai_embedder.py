import os

import numpy as np
import tqdm
from openai import OpenAI

from .base_embedder import BaseEmbedder

MAX_CHARS = 5000


def _extract_openai_embeddings(texts: list[str], batch_size: int = 100):
    """Extract embeddings using OpenAI API."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_embeddings = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="OpenAI API"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
        )
        all_embeddings.extend(response.data)

    return all_embeddings


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI text-embedding-3-large API."""

    @property
    def name(self) -> str:
        return "OpenAI"

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        truncated = [text[:MAX_CHARS] for text in texts]

        print(f"Extracting OpenAI embeddings for {len(truncated)} texts...")
        embeddings_data = _extract_openai_embeddings(truncated, batch_size=batch_size)

        return np.array([item.embedding for item in embeddings_data])

