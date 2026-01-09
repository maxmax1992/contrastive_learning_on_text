import argparse
import json
import os
from typing import Dict, Tuple, List, Any
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb

from dataloader import extract_dataset
from embeddings import (
    BaseEmbedder,
    OpenAIEmbedder,
)


def load_cache(cache_file: str | None) -> tuple[list, list, list, list, dict] | None:
    """Load cache from NPZ file. Returns (index_x, index_y, query_x, query_y, embeddings) or None."""
    if not cache_file or not os.path.exists(cache_file):
        return None

    print(f"Loading cache from {cache_file}...")
    with np.load(cache_file, allow_pickle=True) as data:
        index_x = data["index_x"].tolist()
        index_y = data["index_y"].tolist()
        query_x = data["query_x"].tolist()
        query_y = data["query_y"].tolist()
        embeddings = {k: data[k] for k in data.files if k.endswith("_index") or k.endswith("_query")}

    return index_x, index_y, query_x, query_y, embeddings


def save_cache(cache_file: str | None, index_x: list, index_y: list, query_x: list, query_y: list, embeddings: dict) -> None:
    """Save cache to NPZ file."""
    if not cache_file:
        return

    print(f"Saving cache to {cache_file}...")
    np.savez_compressed(
        cache_file,
        index_x=np.array(index_x, dtype=object),
        index_y=np.array(index_y),
        query_x=np.array(query_x, dtype=object),
        query_y=np.array(query_y),
        **embeddings
    )


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
    """Optimize KNN hyperparameters using Optuna with k-fold CV."""
    
    def objective(trial):
        k = trial.suggest_int('n_neighbors', 3, 11)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        
        clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric='cosine')
        # Use k-fold cross-validation: score is mean accuracy across folds
        scores = cross_val_score(clf, index_emb, index_y, cv=cv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_lr(index_emb, index_y, n_trials=20, cv=3):
    """Optimize Logistic Regression hyperparameters using Optuna with k-fold CV."""
    
    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 1e4, log=True)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        
        clf = LogisticRegression(
            C=C, tol=tol, class_weight=class_weight, penalty='l2', 
            solver='lbfgs', max_iter=1000, random_state=42
        )
        # Use k-fold cross-validation
        scores = cross_val_score(clf, index_emb, index_y, cv=cv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_rf(index_emb, index_y, n_trials=20, cv=3):
    """Optimize RandomForest hyperparameters using Optuna with k-fold CV."""

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, n_jobs=-1, random_state=42
        )
        scores = cross_val_score(clf, index_emb, index_y, cv=cv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optimize_xgb(index_emb, index_y, n_trials=20, cv=3):
    """Optimize XGBoost hyperparameters using Optuna with k-fold CV."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

        clf = xgb.XGBClassifier(**params, n_jobs=-1, random_state=42, verbosity=0)
        scores = cross_val_score(clf, index_emb, index_y, cv=cv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optimize_et(index_emb, index_y, n_trials=20, cv=3):
    """Optimize ExtraTrees hyperparameters using Optuna with k-fold CV."""

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        clf = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, n_jobs=-1, random_state=42
        )
        scores = cross_val_score(clf, index_emb, index_y, cv=cv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def optimize_ensemble(index_emb, index_y, model_params: dict, n_trials=20, cv=3):
    """Optimize weights for the Ensemble model using k-fold CV."""
    model_names = list(model_params.keys())

    def objective(trial):
        weights_raw = {name: trial.suggest_float(f'w_{name}', 0.0, 1.0) for name in model_names}
        total = sum(weights_raw.values())
        if total == 0:
            return 0.0
        weights = [weights_raw[name] / total for name in model_names]

        estimators = []
        for name, params in model_params.items():
            if name == 'knn':
                estimators.append(('knn', KNeighborsClassifier(**params, metric='cosine')))
            elif name == 'lr':
                estimators.append(('lr', LogisticRegression(**params, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)))
            elif name == 'rf':
                estimators.append(('rf', RandomForestClassifier(**params, n_jobs=-1, random_state=42)))
            elif name == 'xgb':
                estimators.append(('xgb', xgb.XGBClassifier(**params, n_jobs=-1, random_state=42, verbosity=0)))
            elif name == 'et':
                estimators.append(('et', ExtraTreesClassifier(**params, n_jobs=-1, random_state=42)))

        ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        scores = cross_val_score(ensemble, index_emb, index_y, cv=cv, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    raw_weights = [best[f'w_{name}'] for name in model_names]
    total_w = sum(raw_weights)
    norm_weights = {name: w / total_w for name, w in zip(model_names, raw_weights)} if total_w > 0 else {name: 1/len(model_names) for name in model_names}

    return norm_weights, study.best_value

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
    """Train and evaluate multiple classifiers using Optuna for hyperparameter tuning."""
    results = {}
    models = {}
    model_params = {}

    # 1. KNN
    print(f"\n  🔍 Optimizing KNN ({n_trials} trials × 3-fold CV)...")
    best_knn_params, knn_cv_score = optimize_knn(index_emb, index_y, n_trials=n_trials)
    print(f"    Best params: {best_knn_params}, CV: {knn_cv_score:.4f}")
    knn = KNeighborsClassifier(**best_knn_params, metric='cosine')
    knn.fit(index_emb, index_y)
    results['KNN'] = compute_metrics(query_y, knn.predict(query_emb))
    print(f"    Test Accuracy: {results['KNN'][0]:.4f}")
    models['knn'] = knn
    model_params['knn'] = best_knn_params

    if not run_extra_classifiers:
        return results

    # 2. Logistic Regression
    print(f"\n  🔍 Optimizing LogisticReg ({n_trials} trials × 3-fold CV)...")
    best_lr_params, lr_cv_score = optimize_lr(index_emb, index_y, n_trials=n_trials)
    print(f"    Best params: {best_lr_params}, CV: {lr_cv_score:.4f}")
    lr = LogisticRegression(**best_lr_params, penalty='l2', solver='lbfgs', max_iter=1000, random_state=random_state)
    lr.fit(index_emb, index_y)
    results['LogisticReg'] = compute_metrics(query_y, lr.predict(query_emb))
    print(f"    Test Accuracy: {results['LogisticReg'][0]:.4f}")
    models['lr'] = lr
    model_params['lr'] = best_lr_params

    # 3. RandomForest
    print(f"\n  🔍 Optimizing RandomForest ({n_trials} trials × 3-fold CV)...")
    best_rf_params, rf_cv_score = optimize_rf(index_emb, index_y, n_trials=n_trials)
    print(f"    Best params: {best_rf_params}, CV: {rf_cv_score:.4f}")
    rf = RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=random_state)
    rf.fit(index_emb, index_y)
    results['RandomForest'] = compute_metrics(query_y, rf.predict(query_emb))
    print(f"    Test Accuracy: {results['RandomForest'][0]:.4f}")
    models['rf'] = rf
    model_params['rf'] = best_rf_params

    # 4. XGBoost
    print(f"\n  🔍 Optimizing XGBoost ({n_trials} trials × 3-fold CV)...")
    best_xgb_params, xgb_cv_score = optimize_xgb(index_emb, index_y, n_trials=n_trials)
    print(f"    Best params: {best_xgb_params}, CV: {xgb_cv_score:.4f}")
    xgb_clf = xgb.XGBClassifier(**best_xgb_params, n_jobs=-1, random_state=random_state, verbosity=0)
    xgb_clf.fit(index_emb, index_y)
    results['XGBoost'] = compute_metrics(query_y, xgb_clf.predict(query_emb))
    print(f"    Test Accuracy: {results['XGBoost'][0]:.4f}")
    models['xgb'] = xgb_clf
    model_params['xgb'] = best_xgb_params

    # 5. ExtraTrees
    print(f"\n  🔍 Optimizing ExtraTrees ({n_trials} trials × 3-fold CV)...")
    best_et_params, et_cv_score = optimize_et(index_emb, index_y, n_trials=n_trials)
    print(f"    Best params: {best_et_params}, CV: {et_cv_score:.4f}")
    et = ExtraTreesClassifier(**best_et_params, n_jobs=-1, random_state=random_state)
    et.fit(index_emb, index_y)
    results['ExtraTrees'] = compute_metrics(query_y, et.predict(query_emb))
    print(f"    Test Accuracy: {results['ExtraTrees'][0]:.4f}")
    models['et'] = et
    model_params['et'] = best_et_params

    # 6. Ensemble
    print(f"\n  🔍 Optimizing Ensemble weights ({n_trials} trials × 3-fold CV)...")
    weights, ensemble_cv_score = optimize_ensemble(index_emb, index_y, model_params, n_trials=n_trials)
    weights_str = ", ".join([f"{k}={v:.2f}" for k, v in weights.items()])
    print(f"    Best weights: {weights_str}, CV: {ensemble_cv_score:.4f}")

    ensemble = VotingClassifier(
        estimators=list(models.items()),
        voting='soft',
        weights=[weights[name] for name in models.keys()]
    )
    ensemble.fit(index_emb, index_y)
    results['Ensemble'] = compute_metrics(query_y, ensemble.predict(query_emb))
    print(f"    Test Accuracy: {results['Ensemble'][0]:.4f}")

    # Save models
    import joblib
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Saving models to {output_dir}/...")

    for name, model in models.items():
        joblib.dump(model, os.path.join(output_dir, f"{name}_openai.joblib"))
    joblib.dump(ensemble, os.path.join(output_dir, "ensemble_openai.joblib"))

    weights_data = {**weights, "accuracy": results['Ensemble'][0]}
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
    embeddings_cache: dict | None = None,
    run_extra_classifiers: bool = False,
) -> tuple[Dict[str, Tuple[float, List[float]]], bool]:
    """
    Evaluate an embedder using multiple classifiers.

    Returns:
        tuple of (results_dict, cache_was_updated)
    """
    name = embedder.name
    print(f"Evaluating {name}...")

    index_key = f"{name}_index"
    query_key = f"{name}_query"
    cache_updated = False

    # Try loading from cache
    index_emb = embeddings_cache.get(index_key) if embeddings_cache else None
    query_emb = embeddings_cache.get(query_key) if embeddings_cache else None

    if index_emb is not None and query_emb is not None:
        print(f"  Using cached embeddings: index={index_emb.shape}, query={query_emb.shape}")
    else:
        # Compute embeddings
        print(f"  Computing index embeddings ({len(index_x)} docs)...")
        index_emb = embedder.embed(index_x, batch_size=batch_size)

        print(f"  Computing query embeddings ({len(query_x)} docs)...")
        query_emb = embedder.embed(query_x, batch_size=batch_size)

        # Update cache
        if embeddings_cache is not None:
            embeddings_cache[index_key] = index_emb
            embeddings_cache[query_key] = query_emb
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


def plot_results(
    results: dict[str, tuple[float, list[float]]],
    output_file: str,
    train_info: dict[str, Any]
):
    """Plot per-class accuracy comparison between models with training setup info."""
    print("Plotting results...")

    if not results:
        print("No results to plot.")
        return

    # Get number of classes from first result
    first_result = next(iter(results.values()))
    n_classes = len(first_result[1])
    indices = np.arange(n_classes)

    num_models = len(results)
    total_width = 0.85
    bar_width = total_width / num_models

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (name, (acc, class_accs)) in enumerate(results.items()):
        offset = i - (num_models - 1) / 2
        ax.bar(
            indices + offset * bar_width,
            class_accs,
            bar_width,
            label=f'{name} ({acc:.3f})'
        )

    ax.set_xlabel('Class ID')
    ax.set_ylabel('Accuracy')
    ax.set_title('20 Newsgroups: Per-class Accuracy on Blind Test Set\n(Models selected via 3-Fold CV on optimization set)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(indices)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add training setup info as text box
    info_text = (
        f"Training Setup:\n"
        f"  Train/CV set: {train_info['n_optimization']:,} docs (3-fold CV, from test split)\n"
        f"  Blind test set: {train_info['n_blind_test']:,} docs (from train split, never seen during tuning)\n"
        f"  Optuna trials: {train_info['n_trials']} per classifier\n"
        f"  Embedder: {train_info['embedder']}\n"
        f"  Classes: {train_info['n_classes']}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, info_text, fontsize=9, verticalalignment='bottom',
             fontfamily='monospace', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenAI embeddings using multiple classifiers")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache", type=str, default=None, help="Path to cache CSV file for storing split data and embeddings")
    parser.add_argument("--output", type=str, default="evaluation_results.png", help="Output plot filename")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick dry run with minimal iterations for hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials for hyperparameter optimization")
    args = parser.parse_args()

    cache_file = args.cache
    cache_updated = False
    
    # Configure hyperparameter search settings
    print("\n" + "="*60)
    print("OpenAI Embeddings Evaluation Pipeline")
    print("="*60)
    
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    n_trials = 1 if args.dry_run else args.n_trials
    print(f"\n🔧 Mode: {'DRY RUN' if args.dry_run else 'FULL RUN'} (n_trials={n_trials}, seed={seed})")

    # Load cache
    cached = load_cache(cache_file)
    embeddings_cache = {}

    if cached:
        index_x, index_y, query_x, query_y, embeddings_cache = cached
    else:
        print("📁 Cache: Not found, loading fresh data...")
        train_x, train_y, test_x, test_y = extract_dataset()

        rng = np.random.default_rng(seed)
        if args.dry_run:
            # Dry run: 60 optimization (for 3-fold CV), 30 blind
            opt_idx = rng.choice(len(test_x), 60, replace=False)
            index_x = [test_x[i] for i in opt_idx]
            index_y = [test_y[i] for i in opt_idx]
            n_blind = 30
        else:
            index_x, index_y = test_x, test_y
            n_blind = 1000

        blind_idx = rng.choice(len(train_x), n_blind, replace=False)
        query_x = [train_x[i] for i in blind_idx]
        query_y = [train_y[i] for i in blind_idx]
        cache_updated = True

    # Print dataset summary
    n_classes = len(set(index_y))
    print(f"\n📊 Dataset Summary:")
    print(f"   • Train/CV set: {len(index_x)} docs (3-fold CV, from test split)")
    print(f"   • Blind test set: {len(query_x)} docs (from train split, never seen during tuning)")
    print(f"   • Classes: {n_classes}")
    print()

    # Build list of embedders to evaluate (OpenAI only)
    embedders: list[BaseEmbedder] = [OpenAIEmbedder()]

    # Evaluate all embedders
    all_results: dict[str, tuple[float, list[float]]] = {}

    for embedder in embedders:
        emb_results, updated = evaluate_embedder(
            embedder,
            index_x, index_y,
            query_x, query_y,
            batch_size=args.batch_size,
            k=args.k,
            n_trials=n_trials,
            embeddings_cache=embeddings_cache,
            run_extra_classifiers=True
        )

        for clf_name, res in emb_results.items():
            all_results[f"{embedder.name}_{clf_name}"] = res

        cache_updated = cache_updated or updated

    # Save cache if updated
    if cache_updated and cache_file:
        save_cache(cache_file, index_x, index_y, query_x, query_y, embeddings_cache)

    # Generate descriptive output filename
    mode_suffix = "dry_run" if args.dry_run else f"{n_trials}trials"
    output_file = f"openai_3fold_cv_blind_test_{mode_suffix}.png"

    # Gather training info for plot
    train_info = {
        'n_optimization': len(index_x),
        'n_blind_test': len(query_x),
        'n_trials': n_trials,
        'n_classes': n_classes,
        'embedder': 'OpenAI text-embedding-3-large (3072d)',
    }

    # Plot results
    plot_results(all_results, output_file, train_info)


if __name__ == "__main__":
    main()
