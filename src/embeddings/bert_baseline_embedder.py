import torch
import numpy as np
import tqdm
from transformers import AutoTokenizer, AutoModel
from .base_embedder import BaseEmbedder


class BertBaselineEmbedder(BaseEmbedder):
    """Baseline BERT embedder without any fine-tuning."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
        device: torch.device | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length

        dtype = torch.float16 if self.device.type == "mps" else None
        model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
        self.model.to(self.device)

    @property
    def name(self) -> str:
        return "BERT_Baseline"

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({self.name})"):
                batch_texts = texts[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
