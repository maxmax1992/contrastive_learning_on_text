import torch
from bert_embedder import TrainableBertEmbedder

class ContrastiveNN(torch.nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 768, output_dim: int = 768):
        super().__init__()
        self.encoder = TrainableBertEmbedder()
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, anchor, positive, negative):
        # Encode all three
        # Note: TrainableBertEmbedder expects list[str]
        # If inputs are batched lists of strings, we can pass them directly.
        
        z_anchor = self.projection_head(self.encoder(anchor))
        z_positive = self.projection_head(self.encoder(positive))
        z_negative = self.projection_head(self.encoder(negative))
        
        # Normalize embeddings to hypersphere
        z_anchor = torch.nn.functional.normalize(z_anchor, p=2, dim=-1)
        z_positive = torch.nn.functional.normalize(z_positive, p=2, dim=-1)
        z_negative = torch.nn.functional.normalize(z_negative, p=2, dim=-1)
        
        return z_anchor, z_positive, z_negative
