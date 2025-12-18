import torch
import numpy as np
import tqdm
from .base_embedder import BaseEmbedder
from contrastive_nn import ContrastiveNN

class ContrastiveEmbedder(BaseEmbedder):
    """Embedder using contrastive-trained model with projection head."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = ContrastiveNN()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "Contrastive_BERT"

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({self.name})"):
                batch_texts = texts[i : i + batch_size]

                # Encoder output
                embeddings = self.model.encoder(batch_texts)
                # Pass through projection head
                embeddings = self.model.projection_head(embeddings)
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
