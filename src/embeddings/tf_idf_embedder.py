import numpy as np
import tqdm
from transformers import AutoTokenizer
from .base_embedder import BaseEmbedder
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFEmbedder(BaseEmbedder):
    """Embedder using TF_IDF."""

    def __init__(
        self,
        model_name: str = "TF_IDF_embedder",
        truncate_model: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.truncate_model = truncate_model
        self.max_length = max_length
        self.vectorizer = TfidfVectorizer()

    @property
    def name(self) -> str:
        return "TF_IDF"

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
        truncated = self._truncate_texts(texts)
        print(f"Extracting TF_IDF embeddings for {len(truncated)} texts...")
        if getattr(self.vectorizer, "vocabulary_", None):
            embeddings_data = self.vectorizer.transform(truncated)
        else:
            embeddings_data = self.vectorizer.fit_transform(truncated)

        vectors = np.asarray(embeddings_data.toarray(), dtype=np.float32)  # type: ignore[attr-defined]
        return vectors

if __name__ == "__main__":
    embedder = TFIDFEmbedder(model_name="TF_IDF_embedder")
    texts = ["This is a test", "This is another test"]
    embeddings = embedder.embed(texts)
    print(embeddings)