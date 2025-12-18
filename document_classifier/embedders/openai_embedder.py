import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from .base_embedder import BaseEmbedder

load_dotenv()


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using Azure OpenAI API."""

    def __init__(
        self,
        max_text_size: int | None = None,
        endpoint: str = "https://opneaipocs.openai.azure.com/",
        deployment: str = "text-embedding-3-large",
        api_version: str = "2024-12-01-preview",
    ):
        """
        Args:
            max_text_size: Maximum number of characters to use from each text.
                          If None, uses full text.
            endpoint: Azure OpenAI endpoint URL.
            deployment: Model deployment name.
            api_version: API version to use.
        """
        self.max_text_size = max_text_size
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

    @property
    def name(self) -> str:
        return "OpenAI"

    def _truncate_texts(self, texts: list[str]) -> list[str]:
        """Truncate texts to max_text_size characters if specified."""
        if self.max_text_size is None:
            return texts
        return [text[:self.max_text_size] for text in texts]

    def fit(self, texts: list[str]) -> None:
        """No-op for OpenAI embedder (pretrained model)."""
        pass

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Get embeddings from Azure OpenAI API."""
        truncated = self._truncate_texts(texts)
        print(f"Extracting OpenAI embeddings for {len(truncated)} texts...")

        all_embeddings = []
        for i in range(0, len(truncated), batch_size):
            batch = truncated[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment,
            )
            all_embeddings.extend(response.data)

        vectors = np.array([item.embedding for item in all_embeddings])
        return vectors


if __name__ == "__main__":
    embedder = OpenAIEmbedder(max_text_size=500)
    texts = ["first phrase", "second phrase", "third phrase"]
    embeddings = embedder.embed(texts)
    print(f"Embedding shape: {embeddings.shape}")
    print(embeddings[:, :5])  # Print first 5 dims
