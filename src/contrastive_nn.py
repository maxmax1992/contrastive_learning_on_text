import torch
from bert_embedder import BertEmbedder

class ContrastiveNN(torch.nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 768, output_dim: int = 768):
        super().__init__()
        self.encoder = BertEmbedder()
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_1, x_2):
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        z_1 = self.projection_head(x_1)
        z_2 = self.projection_head(x_2)
        return z_1, z_2
