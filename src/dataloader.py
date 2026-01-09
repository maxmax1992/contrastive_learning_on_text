"""Utilities for fetching the 20 Newsgroups dataset via scikit-learn."""

from __future__ import annotations

import pickle
import re
from contextlib import suppress
from pathlib import Path
from typing import cast

from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch

_CACHE_FILE = Path(__file__).resolve().parents[1] / ".dataset_cache"



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

    dataset_train = fetch_20newsgroups(subset="train")
    dataset_test = fetch_20newsgroups(subset="test")
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
    print(f"Loaded {len(train_x)} train and {len(test_x)} test documents.")