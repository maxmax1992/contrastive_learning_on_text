import numpy as np
import tqdm
from transformers import AutoTokenizer
from .base_embedder import BaseEmbedder
from openai_embeddings import extract_embeddings


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI API."""

    def __init__(
        self,
        truncate_model: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
    ):
        self.truncate_model = truncate_model
        self.max_length = max_length

    @property
    def name(self) -> str:
        return "OpenAI"

    def _truncate_texts(self, texts: list[str]) -> list[str]:
        """Truncate texts using BERT tokenizer to ensure consistent input length."""
        print(f"Loading tokenizer {self.truncate_model}...")
        tokenizer = AutoTokenizer.from_pretrained(self.truncate_model)
        tokenizer.model_max_length = self.max_length

        print(f"Truncating {len(texts)} texts to {self.max_length} tokens...")
        truncated_texts = []

        batch_size = 1000
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
            batch = texts[i : i + batch_size]
            encodings = tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True
            )

            for input_ids in encodings["input_ids"]:
                decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                truncated_texts.append(decoded_text)

        return truncated_texts

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        # Truncate texts first
        truncated = self._truncate_texts(texts)

        print(f"Extracting OpenAI embeddings for {len(truncated)} texts...")
        embeddings_data = extract_embeddings(truncated, batch_size=batch_size)

        vectors = np.array([item.embedding for item in embeddings_data])
        return vectors
