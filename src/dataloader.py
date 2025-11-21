"""Utilities for fetching the 20 Newsgroups dataset via scikit-learn."""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import pickle
import re
from pathlib import Path
from typing import Iterable, Sequence, cast
from contextlib import suppress
import tqdm
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch
from bert_embedder import BertEmbedder

_CACHE_FILE = Path(__file__).resolve().parents[1] / ".dataset_cache"
_CACHE_FILE_TF_IDF = Path(__file__).resolve().parents[1] / ".dataset_cache_tf_idf"
_CACHE_FILE_BERT = Path(__file__).resolve().parents[1] / ".dataset_cache_bert"

import random
from typing import Callable

class SameClassContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], augmenter: Callable = None,
                n_pos_samples: int = 1, n_neg_samples: int = 1):
        self.dataset = texts
        self.labels = labels
        self.n_pos_samples = n_pos_samples
        self.n_neg_samples = n_neg_samples
        self.class_ids = list(set(labels))
        self.class_to_samples = defaultdict(list)
        for text, label in zip(texts, labels):
            self.class_to_samples[label].append(text)

    def get_pos_samples(self, pos_label: int) -> list[str]:
        samples = self.class_to_samples[pos_label]
        if len(samples) < self.n_pos_samples:
            return random.choices(samples, k=self.n_pos_samples)
        return random.sample(samples, self.n_pos_samples)
    
    def get_neg_samples(self, pos_label: int) -> list[str]:
        different_class_ids = [i for i in self.class_ids if i != pos_label]
        # Pick random classes for negatives
        random_neg_classes = random.choices(different_class_ids, k=self.n_neg_samples)
        neg_samples = []
        for neg_class in random_neg_classes:
            samples = self.class_to_samples[neg_class]
            if not samples: # Should not happen if labels are consistent
                continue
            neg_samples.append(random.choice(samples))
        return neg_samples

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor_text = self.dataset[idx]
        anchor_label = self.labels[idx]
        
        # Get positive and negative samples
        # We take the first one since we typically want 1 pos and 1 neg for simple triplet
        # If n_pos_samples > 1, the user might expect a list, but for standard contrastive 
        # (anchor, pos, neg) usually implies singletons. 
        # Given the user request "treat a positive example as a same class documents", 
        # I will return single strings if n=1, else list.
        
        pos_samples = self.get_pos_samples(anchor_label)
        neg_samples = self.get_neg_samples(anchor_label)
        
        positive_text = pos_samples[0] if self.n_pos_samples == 1 else pos_samples
        negative_text = neg_samples[0] if self.n_neg_samples == 1 else neg_samples
        
        return anchor_text, positive_text, negative_text, anchor_label

def load_20newsgroups(
    subset: str = "train",
    *,
    categories: Sequence[str] | None = None,
    remove: Iterable[str] | None = None,
    shuffle: bool = True,
    random_state: int = 42,
) -> Bunch:
    """Fetch a portion of the 20 Newsgroups dataset.

    Parameters
    ----------
    subset:
        One of ``"train"``, ``"test"`` or ``"all"`` to control the data split.
    categories:
        Optional collection of category names to limit the dataset to. When
        omitted, all 20 categories are returned.
    remove:
        Optional iterable with any combination of ``"headers"``, ``"footers"``,
        and ``"quotes"`` to strip those sections from the raw text.
    shuffle:
        Whether to shuffle the dataset after loading.
    random_state:
        Seed used when ``shuffle`` is True for reproducible ordering.

    Returns
    -------
    sklearn.utils.Bunch
        The standard scikit-learn container with ``data``, ``target``,
        ``target_names`` and metadata fields.
    """

    if remove is None:
        remove = ()

    dataset = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=tuple(remove),
        shuffle=shuffle,
        random_state=random_state,
    )

    return cast(Bunch, dataset)

def extract_tf_idf_vectors(dataset: list[str], force_refresh: bool = False):
    if not force_refresh:
        with suppress(OSError, pickle.UnpicklingError):
            cached = pickle.loads(_CACHE_FILE_TF_IDF.read_bytes())
            print("Loaded tf-idf vectors from cache")
            return cast(csr_matrix, cached)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(dataset)
    _CACHE_FILE_TF_IDF.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE_TF_IDF.write_bytes(pickle.dumps(vectors))
    print("Saved tf-idf vectors to cache")
    return vectors

def batch_embed_bert_text(embedder: BertEmbedder, text_batch: list[str]) -> np.ndarray:
    return embedder(text_batch)

def extract_bert_vectors(dataset: list[str], force_refresh: bool = False) -> np.ndarray:
    if not force_refresh:
        with suppress(OSError, pickle.UnpicklingError):
            cached = pickle.loads(_CACHE_FILE_BERT.read_bytes())
            print("Loaded bert vectors from cache")
            return cast(np.ndarray, cached)
    embedder = BertEmbedder()
    batch_size = 10
    all_vectors = np.empty((len(dataset), 768))
    for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Embedding text"):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        all_vectors[i:min(i+batch_size, len(dataset))] = batch_embed_bert_text(embedder, batch)
    _CACHE_FILE_BERT.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE_BERT.write_bytes(pickle.dumps(all_vectors))
    print("Saved bert vectors to cache")
    return all_vectors


def visualize_dataset_clusters(vectors: np.ndarray, labels: np.ndarray) -> None:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    vectors_pca = pca.fit_transform(vectors)
    import ipdb
    ipdb.set_trace()
    # map to palette the colors
    palette = plt.get_cmap('tab10')
    colors = [palette(i) for i in labels]
    plt.scatter(vectors_pca[:, 0], vectors_pca[:, 1], c=colors)
    plt.show()

def strip_punctuations(text: str) -> str:
    without_punctuations = re.sub(r'[^\w\s]', '', text)
    return without_punctuations

def test_strip_punctuations() -> None:
    text = "Hello, world! This is a test.\n\n"
    target = "Hello world This is a test\n\n"
    assert strip_punctuations(text) == target, "Stripping punctuations failed"

def format_documents(documents: Bunch, to_lower: bool = False) -> list[tuple[str, str]]:

    _SUBJECT_PATTERN = re.compile(r"Subject:\s*(?P<subject>.+?)\n")
    _CONTENT_PATTERN = re.compile(r"Lines:\s*\d+\s*\n(?P<content>[\s\S]+)", re.MULTILINE)
    _SUBJECT_PATTERN_LOWER = re.compile(r"subject:\s*(?P<subject>.+?)\n")
    _CONTENT_PATTERN_LOWER = re.compile(r"lines:\s*\d+\s*\n(?P<content>[\s\S]+)", re.MULTILINE)


    formatted: list[tuple[str, str]] = []
    labels: list[int] = []

    i = 0
    for document, label in zip(documents.data, documents.target):
        i += 1
        if to_lower:
            document = document.lower()
        subject_match = _SUBJECT_PATTERN_LOWER.search(document) if to_lower else _SUBJECT_PATTERN.search(document)
        content_match = _CONTENT_PATTERN_LOWER.search(document) if to_lower else _CONTENT_PATTERN.search(document)

        subject = subject_match.group("subject").strip() if subject_match else ""
        content = content_match.group("content").strip() if content_match else ""
        if subject != "" and content != "":
            formatted.append(('#' + subject.strip() + '\n' + strip_punctuations(content.strip()) ) )
            labels.append(label)
        else:
            invalid_fname = f"./invalid_documents/invalid_document_{i}.txt"
            with open(invalid_fname, "w") as f:
                f.write(document)
    print("Number of documents: ", len(formatted), "out of ", len(documents), " skipped % ", (len(documents) - len(formatted)) / len(documents) * 100)
    # formatted = [strip_punctuations(document) for document in formatted]
    return formatted, labels



def extract_dataset(
    force_refresh: bool = False,
) -> tuple[list[str], list[int], list[str], list[int]]:
    if not force_refresh:
        with suppress(OSError, pickle.UnpicklingError):
            cached = pickle.loads(_CACHE_FILE.read_bytes())
            print("Loaded cached dataset")
            return cast(tuple[list[str], list[int], list[str], list[int]], cached)

    dataset_train = load_20newsgroups(subset="train")
    dataset_test = load_20newsgroups(subset="test")
    train_x, train_y = format_documents(dataset_train)
    test_x, test_y = format_documents(dataset_test)

    data_tuple = (train_x, train_y, test_x, test_y)
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_bytes(pickle.dumps(data_tuple))
    print("Saved dataset to cache")
    return data_tuple

if __name__ == "__main__":
    test_strip_punctuations()

    train_x, train_y, test_x, test_y = extract_dataset()
    # vectors_train = extract_tf_idf_vectors(train_x)
    vectors_train = extract_bert_vectors(train_x)
    visualize_dataset_clusters(vectors_train, train_y)
    # print(dataset_train.data[0])