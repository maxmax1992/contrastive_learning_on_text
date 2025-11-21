import argparse
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tqdm
from dataloader import extract_dataset
from contrastive_nn import ContrastiveNN

def get_embeddings(model, texts, batch_size, device, use_projection=True):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i : i + batch_size]
            
            # Encoder output
            embeddings = model.encoder(batch_texts)
            
            if use_projection:
                embeddings = model.projection_head(embeddings)
                # Normalize projected embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0)

def evaluate_model(model, index_x, index_y, query_x, query_y, batch_size, device, k, name="Model", cache_file=None):
    print(f"Evaluating {name}...")
    
    index_emb = None
    query_emb = None
    
    # Try loading from cache
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings for {name} from {cache_file}...")
        try:
            data = np.load(cache_file)
            index_emb = data['index_emb']
            query_emb = data['query_emb']
            print("Loaded embeddings successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}")
            index_emb = None
            query_emb = None

    if index_emb is None or query_emb is None:
        # Get Embeddings
        print(f"Computing index embeddings for {name}...")
        index_emb = get_embeddings(model, index_x, batch_size, device, use_projection=(name!="Baseline"))
        
        print(f"Computing query embeddings for {name}...")
        query_emb = get_embeddings(model, query_x, batch_size, device, use_projection=(name!="Baseline"))
        
        # Save to cache
        if cache_file:
            print(f"Saving embeddings for {name} to {cache_file}...")
            np.savez(cache_file, index_emb=index_emb, query_emb=query_emb)
    
    # k-NN
    print(f"Fitting k-NN (k={k})...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(index_emb, index_y)
    
    print("Predicting...")
    preds = knn.predict(query_emb)
    
    # Metrics
    acc = accuracy_score(query_y, preds)
    print(f"{name} Overall Accuracy: {acc:.4f}")
    
    # Per class accuracy
    class_correct = {}
    class_total = {}
    
    for pred, true in zip(preds, query_y):
        if true not in class_total:
            class_total[true] = 0
            class_correct[true] = 0
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
            
    accuracies = []
    classes = sorted(class_total.keys())
    for c in classes:
        accuracies.append(class_correct[c] / class_total[c])
        
    return acc, accuracies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading test data...")
    _, _, test_x, test_y = extract_dataset()
    
    # Split 80/20
    print("Splitting test data 80/20...")
    index_x, query_x, index_y, query_y = train_test_split(
        test_x, test_y, test_size=0.2, random_state=42, stratify=test_y
    )
    
    print(f"Index size: {len(index_x)}")
    print(f"Query size: {len(query_x)}")
    
    # Initialize Model
    model = ContrastiveNN()
    model.to(device)
    
    results = {}
    
    # 1. Evaluate Baseline (Raw BERT)
    # Cache file for baseline
    baseline_cache = ".baseline_embeddings.npz"
    
    acc_base, class_acc_base = evaluate_model(
        model, index_x, index_y, query_x, query_y, args.batch_size, device, args.k, 
        name="Baseline", cache_file=baseline_cache
    )
    results["Baseline"] = class_acc_base
    
    # 2. Evaluate Trained (if provided)
    if args.trained_model_path:
        print(f"Loading trained model from {args.trained_model_path}...")
        state_dict = torch.load(args.trained_model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        acc_trained, class_acc_trained = evaluate_model(
            model, index_x, index_y, query_x, query_y, args.batch_size, device, args.k, name="Trained"
        )
        results["Trained"] = class_acc_trained
        
    # Plotting
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    
    n_classes = len(class_acc_base)
    indices = np.arange(n_classes)
    width = 0.35
    
    plt.bar(indices - width/2, class_acc_base, width, label=f'Baseline (Acc: {acc_base:.3f})')
    
    if "Trained" in results:
        plt.bar(indices + width/2, results["Trained"], width, label=f'Trained (Acc: {acc_trained:.3f})')
        
    plt.xlabel('Class ID')
    plt.ylabel('Accuracy')
    plt.title('Per-class k-NN Accuracy (Test Set Split)')
    plt.legend()
    plt.xticks(indices)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_file = "evaluation_results.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    main()
