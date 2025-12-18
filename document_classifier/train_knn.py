"""
Train and evaluate k-NN classifier using TF-IDF or OpenAI embeddings.

Usage:
    python train_knn.py --embedder tfidf --max_text_size 1000
    python train_knn.py --embedder openai --max_text_size 1000
    python train_knn.py --embedder both --max_text_size 1000
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from embedders import BaseEmbedder, TFIDFEmbedder, OpenAIEmbedder


def prepare_data(
    texts: list[str],
    labels: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """
    Split data into train (index) and test (query) sets.

    Args:
        texts: List of text documents
        labels: List of corresponding labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_texts, test_texts, train_labels, test_labels)
    """
    print(f"Splitting data: {len(texts)} samples, test_size={test_size}")

    train_x, test_x, train_y, test_y = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    print(f"Train size: {len(train_x)}, Test size: {len(test_x)}")
    return train_x, test_x, train_y, test_y


def fit_model(
    embedder: BaseEmbedder,
    train_texts: list[str],
    train_labels: list[int],
    k: int = 5,
    batch_size: int = 32,
) -> tuple[KNeighborsClassifier, np.ndarray]:
    """
    Fit k-NN classifier on training data.

    Args:
        embedder: Embedder instance to use for vectorization
        train_texts: Training text documents
        train_labels: Training labels
        k: Number of neighbors for k-NN
        batch_size: Batch size for embedding computation

    Returns:
        Tuple of (fitted KNeighborsClassifier, training embeddings)
    """
    print(f"\n{'='*50}")
    print(f"Fitting model with {embedder.name} embedder (k={k})")
    print(f"{'='*50}")

    # Fit embedder on training data
    embedder.fit(train_texts)

    # Compute training embeddings
    print("Computing training embeddings...")
    train_embeddings = embedder.embed(train_texts, batch_size=batch_size)
    print(f"Training embeddings shape: {train_embeddings.shape}")

    # Fit k-NN classifier
    print(f"Fitting k-NN classifier (k={k})...")
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_embeddings, train_labels)

    return knn, train_embeddings


def eval(
    embedder: BaseEmbedder,
    knn: KNeighborsClassifier,
    test_texts: list[str],
    test_labels: list[int],
    batch_size: int = 32,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate k-NN classifier on test data.

    Args:
        embedder: Embedder instance (must be fitted)
        knn: Fitted KNeighborsClassifier
        test_texts: Test text documents
        test_labels: Test labels
        batch_size: Batch size for embedding computation
        class_names: Optional list of class names for reporting

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {embedder.name}...")

    # Compute test embeddings
    print("Computing test embeddings...")
    test_embeddings = embedder.embed(test_texts, batch_size=batch_size)
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # Predict
    print("Predicting...")
    predictions = knn.predict(test_embeddings)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\n{embedder.name} Overall Accuracy: {accuracy:.4f}")

    # Per-class accuracy
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    for pred, true in zip(predictions, test_labels):
        if true not in class_total:
            class_total[true] = 0
            class_correct[true] = 0
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    per_class_accuracy = []
    classes = sorted(class_total.keys())
    for c in classes:
        per_class_accuracy.append(class_correct[c] / class_total[c])

    # Classification report
    if class_names:
        print("\nClassification Report:")
        print(classification_report(test_labels, predictions, target_names=class_names))

    results = {
        "embedder": embedder.name,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "predictions": predictions.tolist(),
        "test_labels": test_labels,
    }

    return results


def plot_results(
    results: dict[str, dict[str, Any]],
    output_file: str = "evaluation_results.png",
) -> None:
    """Plot per-class accuracy comparison between models."""
    print("\nPlotting results...")

    if not results:
        print("No results to plot.")
        return

    first_result = next(iter(results.values()))
    n_classes = len(first_result["per_class_accuracy"])
    indices = np.arange(n_classes)

    num_models = len(results)
    total_width = 0.8
    bar_width = total_width / num_models

    plt.figure(figsize=(12, 6))

    for i, (name, result) in enumerate(results.items()):
        offset = i - (num_models - 1) / 2
        plt.bar(
            indices + offset * bar_width,
            result["per_class_accuracy"],
            bar_width,
            label=f'{name} (Acc: {result["accuracy"]:.3f})',
        )

    plt.xlabel("Class ID")
    plt.ylabel("Accuracy")
    plt.title("Per-class k-NN Accuracy")
    plt.legend()
    plt.xticks(indices)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


def save_model(
    embedder: BaseEmbedder,
    knn: KNeighborsClassifier,
    output_path: str,
) -> None:
    """Save embedder and k-NN model to disk."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    # Save k-NN model
    knn_path = path / f"{embedder.name}_knn.pkl"
    with open(knn_path, "wb") as f:
        pickle.dump(knn, f)
    print(f"Saved k-NN model to {knn_path}")

    # Save TF-IDF vectorizer if applicable
    if isinstance(embedder, TFIDFEmbedder):
        vectorizer_path = path / "tfidf_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(embedder.vectorizer, f)
        print(f"Saved TF-IDF vectorizer to {vectorizer_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate k-NN classifier with embeddings"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        choices=["tfidf", "openai", "both"],
        default="tfidf",
        help="Embedder to use",
    )
    parser.add_argument(
        "--max_text_size",
        type=int,
        default=None,
        help="Maximum text size in characters",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for k-NN",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Directory to save trained models",
    )
    args = parser.parse_args()

    # Sanity check with sample data
    print("Running sanity checks with sample data...")
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "The stock market showed strong gains today",
        "Scientists discovered a new species of butterfly",
        "The football team won the championship game",
        "Artificial neural networks mimic the human brain",
        "JavaScript is widely used for web development",
        "Economic indicators suggest growth ahead",
        "A rare bird was spotted in the national park",
    ]
    sample_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # 5 classes

    # Prepare data
    train_x, test_x, train_y, test_y = prepare_data(
        sample_texts,
        sample_labels,
        test_size=args.test_size,
        random_state=42,
    )

    # Build embedders list
    embedders: list[BaseEmbedder] = []
    if args.embedder in ["tfidf", "both"]:
        embedders.append(TFIDFEmbedder(max_text_size=args.max_text_size))
    if args.embedder in ["openai", "both"]:
        embedders.append(OpenAIEmbedder(max_text_size=args.max_text_size))

    # Train and evaluate each embedder
    all_results: dict[str, dict[str, Any]] = {}

    for embedder in embedders:
        # Fit model
        knn, _ = fit_model(
            embedder,
            train_x,
            train_y,
            k=min(args.k, len(train_x) - 1),  # k can't exceed training samples
            batch_size=args.batch_size,
        )

        # Evaluate
        results = eval(
            embedder,
            knn,
            test_x,
            test_y,
            batch_size=args.batch_size,
        )

        all_results[embedder.name] = results

        # Save model if requested
        if args.save_model:
            save_model(embedder, knn, args.save_model)

    # Plot comparison
    plot_results(all_results, args.output)

    print("\nSanity check completed successfully!")
    print("To use with your own data, import and call the functions directly:")
    print("  from train_knn import prepare_data, fit_model, eval")


if __name__ == "__main__":
    main()
