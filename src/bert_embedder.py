from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class BertEmbedder:
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        *,
        max_length: int = 256,
    ):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Override tokenizer max length so huge documents get truncated instead of
        # blowing up attention memory on MPS.
        self.tokenizer.model_max_length = max_length

        dtype = torch.float16 if device.type == "mps" else None
        model_kwargs = {"dtype": dtype} if dtype is not None else {}
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).eval()
        self.model.to(device)

    def __call__(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden) → CLS/BOS
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=-1
        ).cpu().numpy()
        return sentence_embeddings

class TrainableBertEmbedder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 256,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        
        self.model = AutoModel.from_pretrained(model_name)
        # We don't call .eval() or .to(device) here immediately if we want to manage it in the training loop,
        # but usually it's fine to leave it to the caller or do it here. 
        # However, for nn.Module, .to(device) is usually called by the user on the instance.
        
    def forward(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden) → CLS/BOS
        
        # Normalize? Usually contrastive loss works better with normalized vectors, 
        # but sometimes it's part of the loss or projection head. 
        # The original had normalization. Let's keep it but return tensor.
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=-1
        )
        return sentence_embeddings

if __name__ == "__main__":
    samples = [
        "The King is the emperor of the kingdom.",
        "The emperor is the king of the kingdom.",
        "I've recently completed my PhD in computer science."
    ]

    embedder = BertEmbedder()
    embeddings = embedder(samples)
    print(embeddings.shape)  # e.g., torch.Size([2, 768])

    # test the similarity between the embeddings
    dist_0_1 = np.linalg.norm(embeddings[0] - embeddings[1])
    dist_0_2 = np.linalg.norm(embeddings[0] - embeddings[2])
    dist_1_2 = np.linalg.norm(embeddings[1] - embeddings[2])
    print(f"Distance between 0 and 1: {dist_0_1}")
    print(f"Distance between 0 and 2: {dist_0_2}")
    print(f"Distance between 1 and 2: {dist_1_2}")
    assert dist_0_1 < dist_0_2 and dist_0_1 < dist_1_2, "Distance between 0 and 1 is not less than distance between 0 and 2 and distance between 1 and 2"

    print("Distances separated predictions:")
    emb_0 = embedder([samples[0]])
    emb_1 = embedder([samples[1]])
    emb_2 = embedder([samples[2]])
    print(f"Distance between 0 and 1: {np.linalg.norm(emb_0 - emb_1)}")
    print(f"Distance between 0 and 2: {np.linalg.norm(emb_0 - emb_2)}")
    print(f"Distance between 1 and 2: {np.linalg.norm(emb_1 - emb_2)}")