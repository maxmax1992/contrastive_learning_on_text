import argparse
import os
import json
from typing import cast, Dict, Tuple, List, Any
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform
import matplotlib.pyplot as plt

from dataloader import extract_dataset
from embeddings import (
    BaseEmbedder,
    BertBaselineEmbedder,
    ContrastiveEmbedder,
    OpenAIEmbedder,
    TFIDFEmbedder,
)


def load_cache_csv(cache_file: str | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load cache from CSV file, returns (index_df, query_df) or (None, None) if not found."""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cache from {cache_file}...")
        df = pd.read_csv(cache_file)
        index_df = cast(pd.DataFrame, df[df["set_type"] == "index"].copy())
        query_df = cast(pd.DataFrame, df[df["set_type"] == "query"].copy())
        return index_df, query_df
    return None, None


def save_cache_csv(
    cache_file: str | None,
    index_df: pd.DataFrame | None,
    query_df: pd.DataFrame | None
) -> None:
    """Save cache to CSV file."""
    if cache_file and index_df is not None and query_df is not None:
        print(f"Saving cache to {cache_file}...")
        combined = pd.concat([index_df, query_df], ignore_index=True)
        combined.to_csv(cache_file, index=False)


def get_embeddings_from_df(df, model_name):
    """Extract embeddings array from dataframe column."""
    col = f"emb_{model_name}"
    if col not in df.columns:
        return None
    embeddings = df[col].apply(lambda row: deserialize_embedding(model_name, row)).tolist()
    return np.vstack(embeddings)


def add_embeddings_to_df(df, model_name, embeddings):
    """Add embeddings as a new column to dataframe."""
    col = f"emb_{model_name}"
    df[col] = [serialize_embedding(model_name, emb) for emb in embeddings]


def serialize_embedding(model_name: str, embedding: np.ndarray) -> str:
    """Serialize embeddings; TF-IDF uses sparse representation to save space."""
    arr = np.asarray(embedding).ravel()
    if model_name == "TF_IDF":
        nonzero_indices = np.flatnonzero(arr)
        values = arr[nonzero_indices]
        payload = {
            "indices": nonzero_indices.tolist(),
            "values": values.tolist(),
            "size": arr.size,
        }
        return json.dumps(payload)
    return json.dumps(arr.tolist())


def deserialize_embedding(model_name: str, payload: str) -> np.ndarray:
    """Deserialize embeddings from cache."""
    data = json.loads(payload)
    if model_name == "TF_IDF" and isinstance(data, dict):
        size = data.get("size")
        if size is None:
            raise ValueError("TF-IDF cache entry is missing vector size.")
        vector = np.zeros(size, dtype=np.float32)
        indices = np.array(data.get("indices", []), dtype=np.int32)
        values = np.array(data.get("values", []), dtype=np.float32)
        vector[indices] = values
        return vector
    return np.asarray(data, dtype=np.float32)


def compute_metrics(y_true, y_pred) -> Tuple[float, List[float]]:
    """Compute overall accuracy and per-class accuracies."""
    acc = accuracy_score(y_true, y_pred)
    
    class_correct = {}
    class_total = {}

    for pred, true in zip(y_pred, y_true):
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



import optuna
from sklearn.metrics import accuracy_score

# Helper to suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_knn(index_emb, index_y, n_trials=20, cv=3):
    """Optimize KNN hyperparameters using Optuna."""
    # Split for CV inside objective
    # Note: For efficiency with large embeddings, we might want to use a fixed validation set
    # or just use cross_val_score. With 50k dim (TF-IDF), cross_val_score is slow.
    # But user asked for 'n-optimizations', so we follow the pattern.
    # To keep it reasonably fast, we'll use a simple hold-out validation inside the trial
    # or cross_val_score if feasible. Given 'cv=3' in prev code, let's stick to that logic.
    
    # Pre-split for faster optimization (hold-out)
    idx_train, idx_val, y_train, y_val = train_test_split(
        index_emb, index_y, test_size=0.2, random_state=42, stratify=index_y
    )
    
    def objective(trial):
        k = trial.suggest_int('n_neighbors', 3, 11)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        
        clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric='cosine')
        clf.fit(idx_train, y_train)
        preds = clf.predict(idx_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_lr(index_emb, index_y, n_trials=20, cv=3):
    """Optimize Logistic Regression hyperparameters using Optuna."""
    idx_train, idx_val, y_train, y_val = train_test_split(
        index_emb, index_y, test_size=0.2, random_state=42, stratify=index_y
    )
    
    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 1e4, log=True)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        
        clf = LogisticRegression(
            C=C, tol=tol, class_weight=class_weight, penalty='l2', 
            solver='lbfgs', max_iter=1000, random_state=42
        )
        clf.fit(idx_train, y_train)
        preds = clf.predict(idx_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_svm(index_emb, index_y, n_trials=20, cv=3):
    """Optimize SVM hyperparameters using Optuna."""
    idx_train, idx_val, y_train, y_val = train_test_split(
        index_emb, index_y, test_size=0.2, random_state=42, stratify=index_y
    )
    
    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 1e4, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001])
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        
        clf = SVC(
            C=C, gamma=gamma, kernel=kernel, 
            probability=True, class_weight='balanced', random_state=42
        )
        clf.fit(idx_train, y_train)
        preds = clf.predict(idx_val)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def optimize_ensemble(index_emb, index_y, knn, lr, svm, n_trials=20):
    """Optimize weights for the Ensemble model."""
    idx_train, idx_val, y_train, y_val = train_test_split(
        index_emb, index_y, test_size=0.2, random_state=42, stratify=index_y
    )
    
    # Pre-compute probabilities to speed up weight tuning
    # We must fit estimators on the training split first
    knn.fit(idx_train, y_train)
    lr.fit(idx_train, y_train)
    svm.fit(idx_train, y_train)
    
    probs_knn = knn.predict_proba(idx_val)
    probs_lr = lr.predict_proba(idx_val)
    probs_svm = svm.predict_proba(idx_val)
    
    # Map class indices to labels
    classes = knn.classes_
    
    def objective(trial):
        w_knn = trial.suggest_float('w_knn', 0.0, 1.0)
        w_lr = trial.suggest_float('w_lr', 0.0, 1.0)
        w_svm = trial.suggest_float('w_svm', 0.0, 1.0)
        
        # Normalize weights
        total = w_knn + w_lr + w_svm
        if total == 0:
            return 0.0
            
        final_probs = (w_knn * probs_knn + w_lr * probs_lr + w_svm * probs_svm) / total
        preds = classes[np.argmax(final_probs, axis=1)]
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best = study.best_params
    raw_weights = [best['w_knn'], best['w_lr'], best['w_svm']]
    total_w = sum(raw_weights)
    if total_w > 0:
        norm_weights = [w / total_w for w in raw_weights]
    else:
        norm_weights = [1.0/3, 1.0/3, 1.0/3] # Fallback if all zero
        
    return norm_weights

def train_and_evaluate_classifiers(
    index_emb: np.ndarray,
    index_y: list,
    query_emb: np.ndarray,
    query_y: list,
    k: int = 5,
    n_trials: int = 20,
    random_state: int = 42,
    run_extra_classifiers: bool = False
) -> Dict[str, Tuple[float, List[float]]]:
    """
    Train KNN, and optionally Logistic Regression, SVM, and Ensemble models using Optuna.
    """
    results = {}
    
    if not run_extra_classifiers:
        # Standard KNN (no optimization requested for non-OpenAI, or keep k=5 as baseline?)
        # User said: "including the K-NN as we might need to optimize that model hyperparams as well"
        # Since this function handles everything, I should probably optimize KNN always?
        # But for 'Baseline' models etc, usually we keep fixed args.
        # User explicitly asked for "N-optimizations... including the K-NN".
        # So I will optimize KNN for ALL embedders.
        
        print(f"  Optimizing KNN ({n_trials} trials)...")
        best_knn_params = optimize_knn(index_emb, index_y, n_trials=n_trials)
        print(f"    Best KNN params: {best_knn_params}")
        
        knn = KNeighborsClassifier(**best_knn_params, metric='cosine')
        knn.fit(index_emb, index_y)
        knn_preds = knn.predict(query_emb)
        results['KNN'] = compute_metrics(query_y, knn_preds)
        print(f"    KNN Accuracy: {results['KNN'][0]:.4f}")
        return results

    # 1. Optimize KNN
    print(f"  Optimizing KNN ({n_trials} trials)...")
    best_knn_params = optimize_knn(index_emb, index_y, n_trials=n_trials)
    print(f"    Best KNN params: {best_knn_params}")
    knn = KNeighborsClassifier(**best_knn_params, metric='cosine') # Fit later for ensemble optimization? 
    # Optuna code fits on split. We need final fit on full index.
    knn.fit(index_emb, index_y)
    knn_preds = knn.predict(query_emb)
    results['KNN'] = compute_metrics(query_y, knn_preds)
    print(f"    KNN Accuracy: {results['KNN'][0]:.4f}")

    # 2. Optimize LR
    print(f"  Optimizing Logistic Regression ({n_trials} trials)...")
    best_lr_params = optimize_lr(index_emb, index_y, n_trials=n_trials)
    print(f"    Best LR params: {best_lr_params}")
    lr = LogisticRegression(**best_lr_params, penalty='l2', solver='lbfgs', max_iter=1000, random_state=random_state)
    lr.fit(index_emb, index_y)
    lr_preds = lr.predict(query_emb)
    results['LogisticReg'] = compute_metrics(query_y, lr_preds)
    print(f"    LR Accuracy: {results['LogisticReg'][0]:.4f}")

    # 3. Optimize SVM
    print(f"  Optimizing SVM ({n_trials} trials)...")
    best_svm_params = optimize_svm(index_emb, index_y, n_trials=n_trials)
    print(f"    Best SVM params: {best_svm_params}")
    svm = SVC(
        **best_svm_params,
        probability=True, class_weight='balanced', random_state=random_state
    )
    svm.fit(index_emb, index_y)
    svm_preds = svm.predict(query_emb)
    results['SVM'] = compute_metrics(query_y, svm_preds)
    print(f"    SVM Accuracy: {results['SVM'][0]:.4f}")

    # 4. Optimize Ensemble
    print(f"  Optimizing Ensemble weights ({n_trials} trials)...")
    # Optimize weights using training split logic
    weights = optimize_ensemble(index_emb, index_y, knn, lr, svm, n_trials=n_trials)
    print(f"    Best Ensemble weights: KNN={weights[0]:.2f}, LR={weights[1]:.2f}, SVM={weights[2]:.2f}")
    
    ensemble = VotingClassifier(
        estimators=[
            ('knn', knn), 
            ('lr', lr), 
            ('svm', svm)
        ],
        voting='soft',
        weights=weights
    )
    ensemble.fit(index_emb, index_y)
    ens_preds = ensemble.predict(query_emb)
    results['Ensemble'] = compute_metrics(query_y, ens_preds)
    print(f"    Ensemble Accuracy: {results['Ensemble'][0]:.4f}")

    # Save models if this is the extensive OpenAI run
    if run_extra_classifiers:
        import joblib
        output_dir = "models"
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving models to {output_dir}/...")
        
        joblib.dump(knn, os.path.join(output_dir, "knn_openai.joblib"))
        joblib.dump(lr, os.path.join(output_dir, "lr_openai.joblib"))
        joblib.dump(svm, os.path.join(output_dir, "svm_openai.joblib"))
        joblib.dump(ensemble, os.path.join(output_dir, "ensemble_openai.joblib"))
        
        # Save weights metadata
        weights_data = {
            "w_knn": weights[0],
            "w_lr": weights[1],
            "w_svm": weights[2],
            "accuracy": results['Ensemble'][0]
        }
        with open(os.path.join(output_dir, "ensemble_weights.json"), "w") as f:
            json.dump(weights_data, f, indent=2)
            
    return results




def evaluate_embedder(
    embedder: BaseEmbedder,
    index_x: list[str],
    index_y: list,
    query_x: list[str],
    query_y: list,
    batch_size: int,
    k: int,
    n_trials: int = 20,
    index_df: pd.DataFrame | None = None,
    query_df: pd.DataFrame | None = None,
    run_extra_classifiers: bool = False,
) -> tuple[Dict[str, Tuple[float, List[float]]], bool]:
    """
    Evaluate an embedder using multiple classifiers.

    Returns:
        tuple of (results_dict, cache_was_updated)
    """
    name = embedder.name
    print(f"Evaluating {name}...")

    index_emb = None
    query_emb = None
    cache_updated = False

    # Try loading from cache
    if index_df is not None and query_df is not None:
        index_emb = get_embeddings_from_df(index_df, name)
        query_emb = get_embeddings_from_df(query_df, name)
        if index_emb is not None and query_emb is not None:
            print(f"Using cached embeddings for {name}...")
            print(f"Loaded embeddings: index={index_emb.shape}, query={query_emb.shape}")

    # Compute if not cached
    if index_emb is None or query_emb is None:
        print(f"Computing index embeddings for {name}...")
        index_emb = embedder.embed(index_x, batch_size=batch_size)

        print(f"Computing query embeddings for {name}...")
        query_emb = embedder.embed(query_x, batch_size=batch_size)

        # Update cache
        if index_df is not None and query_df is not None:
            add_embeddings_to_df(index_df, name, index_emb)
            add_embeddings_to_df(query_df, name, query_emb)
            cache_updated = True

    # Train and evaluate classifiers
    results = train_and_evaluate_classifiers(
        index_emb, index_y, 
        query_emb, query_y, 
        k=k,
        n_trials=n_trials,
        run_extra_classifiers=run_extra_classifiers
    )

    return results, cache_updated


def plot_results(results: dict[str, tuple[float, list[float]]], output_file: str = "evaluation_results.png"):
    """Plot per-class accuracy comparison between models."""
    print("Plotting results...")

    if not results:
        print("No results to plot.")
        return

    # Get number of classes from first result
    first_result = next(iter(results.values()))
    n_classes = len(first_result[1])
    indices = np.arange(n_classes)

    num_models = len(results)
    # Adjust bar width to fit more models if needed
    total_width = 0.85
    bar_width = total_width / num_models

    plt.figure(figsize=(14, 7)) # Increased width for more models

    for i, (name, (acc, class_accs)) in enumerate(results.items()):
        offset = i - (num_models - 1) / 2
        plt.bar(
            indices + offset * bar_width,
            class_accs,
            bar_width,
            label=f'{name} ({acc:.3f})'
        )

    plt.xlabel('Class ID')
    plt.ylabel('Accuracy')
    plt.title('Per-class Accuracy Comparison (Test Set Split)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legend outside to save space
    plt.xticks(indices)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding models using multiple classifiers")
    parser.add_argument("--trained_model_path", type=str, default=None, help="Path to trained contrastive model checkpoint")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_openai", action="store_true", help="Evaluate OpenAI embeddings")
    parser.add_argument("--cache", type=str, default=None, help="Path to cache CSV file for storing split data and embeddings")
    parser.add_argument("--output", type=str, default="evaluation_results.png", help="Output plot filename")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick dry run with minimal iterations for hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials for hyperparameter optimization")
    args = parser.parse_args()

    cache_file = args.cache
    cache_updated = False
    
    # Configure hyperparameter search settings
    if args.dry_run:
        print("Running in DRY RUN mode: n_trials=1")
        n_trials = 1
    else:
        n_trials = args.n_trials

    # Load or create cache dataframes
    index_df, query_df = load_cache_csv(cache_file)

    # Check if we have cached split data
    if index_df is not None and query_df is not None and "text" in index_df.columns:
        print("Using cached split data (skipping data loading)...")
        index_x = index_df["text"].tolist()
        query_x = query_df["text"].tolist()
        index_y = index_df["label"].tolist()
        query_y = query_df["label"].tolist()
    else:
        # Load Data
        print("Loading test data...")
        _, _, test_x, test_y = extract_dataset()
        # training has 10k examples
        # not used here
        
        # Split 80/20
        print("Splitting test data 80/20...")
        index_x, query_x, index_y, query_y = train_test_split(
            test_x, test_y, test_size=0.2, random_state=42, stratify=test_y
        )

        # Create cache dataframes
        if cache_file:
            index_df = pd.DataFrame({
                "set_type": ["index"] * len(index_x),
                "label": index_y,
                "text": index_x
            })
            query_df = pd.DataFrame({
                "set_type": ["query"] * len(query_x),
                "label": query_y,
                "text": query_x
            })
            cache_updated = True

    print(f"Index size: {len(index_x)}")
    print(f"Query size: {len(query_x)}")

    # Build list of embedders to evaluate
    embedders: list[BaseEmbedder] = []

    # 1. TF-IDF baseline
    embedders.append(TFIDFEmbedder())

    # 2. Baseline BERT (always evaluated)
    embedders.append(BertBaselineEmbedder())

    # 3. Contrastive trained model (if checkpoint provided)
    if args.trained_model_path:
        embedders.append(ContrastiveEmbedder(args.trained_model_path))

    # 4. OpenAI (if requested)
    if args.eval_openai:
        embedders.append(OpenAIEmbedder())

    # Evaluate all embedders
    all_results: dict[str, tuple[float, list[float]]] = {}

    for embedder in embedders:
        # User requested SVM/LogReg/Ensemble only for OpenAI
        run_extra = (embedder.name == "OpenAI")
        
        emb_results, updated = evaluate_embedder(
            embedder,
            index_x, index_y,
            query_x, query_y,
            batch_size=args.batch_size,
            k=args.k,
            n_trials=n_trials,
            index_df=index_df,
            query_df=query_df,
            run_extra_classifiers=run_extra
        )
        
        # Flatten results: {Embedder}_{Classifier}
        for clf_name, res in emb_results.items():
            all_results[f"{embedder.name}_{clf_name}"] = res
            
        cache_updated = cache_updated or updated

    # Save cache if updated
    if cache_updated and cache_file:
        save_cache_csv(cache_file, index_df, query_df)

    # Plot results
    plot_results(all_results, args.output)


if __name__ == "__main__":
    main()
