import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import tqdm
import numpy as np
from dataloader import SameClassContrastiveDataset, extract_dataset
from contrastive_nn import ContrastiveNN

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm.tqdm(dataloader, desc="Training"):
        # Unpack batch
        # Dataset returns: anchor_text, positive_text, negative_text, anchor_label
        # DataLoader collates them into lists (tuples) of strings
        anchor, positive, negative, _ = batch
        
        # Forward pass
        # ContrastiveNN expects lists of strings
        # We need to ensure they are lists, DataLoader might return tuples
        anchor = list(anchor)
        positive = list(positive)
        negative = list(negative)
        
        optimizer.zero_grad()
        
        z_anchor, z_positive, z_negative = model(anchor, positive, negative)
        
        loss = loss_fn(z_anchor, z_positive, z_negative)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    BATCH_SIZE = 8 # Small batch size for BERT fine-tuning usually
    LR = 2e-5
    EPOCHS = 3
    MARGIN = 1.0
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading dataset...")
    train_x, train_y, test_x, test_y = extract_dataset()
    
    # Create Dataset
    # Using a subset for quick testing/demonstration if needed, but here full dataset
    train_dataset = SameClassContrastiveDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    print("Initializing model...")
    model = ContrastiveNN()
    model.to(device)
    
    # Optimizer: Fine-tune all parameters or just head?
    # User asked: "finetunes Bert last layer or an adapter layer"
    # Currently TrainableBertEmbedder has the whole model. 
    # To freeze all but last layer:
    
    # Freeze all BERT parameters first
    for param in model.encoder.model.parameters():
        param.requires_grad = False
        
    # Unfreeze last layer (encoder.layer[-1] usually)
    # ModernBERT structure: model.layers[-1] or similar.
    # Let's inspect safely. If we can't find it easily by name, we can iterate.
    # But usually for AutoModel it's model.encoder.layer[-1] or model.layers[-1]
    
    # Let's try to unfreeze the last 2 layers of the encoder to be safe and effective
    # Assuming HuggingFace standard naming (encoder.layer or layers)
    # We can just unfreeze the last few parameters in the list
    
    print("Unfreezing last BERT layer...")
    bert_params = list(model.encoder.model.parameters())
    # Unfreeze last 20 tensors (roughly last layer + pooler if exists)
    for param in bert_params[-20:]:
        param.requires_grad = True
    
    params_to_optimize = [
        {'params': model.projection_head.parameters(), 'lr': LR},
        {'params': [p for p in model.encoder.model.parameters() if p.requires_grad], 'lr': LR * 0.1} # Lower LR for BERT
    ]
    
    print(f"Optimizing Projection Head and Last BERT Layer")
    
    optimizer = AdamW(params_to_optimize)
    loss_fn = torch.nn.TripletMarginLoss(margin=MARGIN)
    
    # Training Loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"contrastive_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()