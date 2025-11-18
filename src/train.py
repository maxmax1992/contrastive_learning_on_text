from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch


def train(self, dataloader: DataLoader, optimizer: Optimizer, device: torch.device) -> None:
    self.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = self(batch)
        loss = self.loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(self, dataloader: DataLoader, device: torch.device) -> None:
    self.eval()
    total_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        output = self(batch)
        loss = self.loss_fn(output, batch)
        total_loss += loss.item()
    return total_loss / len(dataloader)