# Contrastive fine-tuning on embedding Transformer models

## Motivation
How to get embeddings that fit your use-case and/or demand?
Sometimes you'd like to use existing class information of the to-be embedded text, which could influence the classification decision on the downstream model (KNN or Logistic regression for example).

This implementation is a Contrastive Learning approach to the classification of the text, where I'm fine-tuning the embedding model. For the sake of the demo data used here is a "The 20 newsgroups text dataset" dataset which comprises of 20 different classses and about 10k+ training examples and about 250 test examples.

The comparison is done by:
- fine-tuning ModernBert model's last layer with contrastive learning (using triplet loss - same class items are clustered to be closer together in the embedding space, while different class embeddings are pushed away from each other)
- comparing it to simple Bert (non-finetuned) embedding -> KNN-Classification
- Another models will be added for comparison as well e.g. AzureOpenAI embeddings + SetFit (similar finetuning with CL)-
- Comparison to other models like AzureOpenAI Embeddings, SimFit approach, \#TODO


## Installation
Required: `python >= 3.12.9`, `uv`
run following:
> `uv sync`

## Experiments
Overall approach:  
Train the contrastive models on the built-in train split, evaluate with baseline (just embedding without Contrastive pre-training) eval split, 80% for populating the KNN index, the rest 20% of the test split on final evaluation.


## Training
Train the model with:  
> `uv run src/train.py`
This saves the checkpoints under the root such as `./contrastive_model_expoch_x.pt`

## Evaluation
For evaluation we're using 20 newsgroups built-in test split, the KNN index is populated with 80% of the test data, for the rest 20% of the data we're performing the testing, see the results below for the comparison between baseline BERT without CL finetuning (`Baseline`) and CL finetuned embedding projection (`Trained`).

![Evaluation Results](evaluation_results.png)

### Basic evaluation
Run evaluation with:
> `uv run src/evaluate_knn.py --trained_model_path contrastive_model_epoch_3.pt`

### Caching embeddings for model comparison
Use `--cache` to save embeddings and test split data to a CSV file. This allows you to:
- Skip expensive embedding computation on subsequent runs
- Add new models incrementally without recomputing existing embeddings
- Compare multiple models using the exact same test split

```bash
# First run - computes Baseline + Trained embeddings, saves to cache
uv run src/evaluate_knn.py --trained_model_path contrastive_model_epoch_3.pt --cache eval_cache.csv

# Add OpenAI embeddings - loads existing from cache, only computes OpenAI
uv run src/evaluate_knn.py --trained_model_path contrastive_model_epoch_3.pt --eval_openai --cache eval_cache.csv

# Future runs with same cache load all embeddings instantly
uv run src/evaluate_knn.py --trained_model_path contrastive_model_epoch_3.pt --eval_openai --cache eval_cache.csv
```

The cache CSV stores: `set_type`, `label`, `text`, and `emb_{ModelName}` columns for each evaluated model.

### Adding custom embedding models
The codebase uses a modular embedder architecture. To add a new embedding model:

1. Create `src/embeddings/{your_model}_embedder.py`
2. Implement the `BaseEmbedder` interface:

```python
from embeddings import BaseEmbedder
import numpy as np

class YourModelEmbedder(BaseEmbedder):
    @property
    def name(self) -> str:
        return "YourModel"  # Used as cache key (emb_YourModel)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        # Your embedding logic here
        # Return shape: (len(texts), embedding_dim)
        pass
```

3. Add to `src/embeddings/__init__.py`
4. Use in `evaluate_knn.py` by adding to the embedders list

**Note: epoch 3 checkpoint used in examples**

## Notes
While implementing this approach I wasn't aware that SetFit existed,, I've had Contrastive-Leanring experience during my intership at Nokia in 2020, but for image space. This is a nice exmaple how the Deep Learning can be applied to many disciplines and some concepts propagate across modalities.