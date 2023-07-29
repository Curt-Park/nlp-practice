"""Data handler for skip-gram model."""

from typing import Dict, List, Set, Tuple

import nltk
import numpy as np
from nltk.tokenize import word_tokenize

nltk.download("punkt")


def gather_word_freqs(
    split_text: List[str],
    sampling_ratio: float = 1e-3,
) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[int, str]]:
    """Subsample words and generate dictionaries."""
    vocab: Dict[str, int] = dict()
    idx_to_word: Dict[int, str] = dict()
    word_to_idx: Dict[str, int] = dict()
    total = 0.0

    # Get frequency for each word.
    for word in split_text:
        if word not in vocab:
            vocab[word] = 0
            idx_to_word[len(word_to_idx)] = word
            word_to_idx[word] = len(word_to_idx)
        vocab[word] += 1.0
        total += 1.0

    # Subsampling.
    # p(w_i) = 1 - (sqrt(sub_sampling / freq) + sub_sampling / freq)
    subsamples: List[str] = []
    for word in split_text:
        val = np.sqrt(sampling_ratio * total / vocab[word])
        prob = 1 - (val * (1 + val))
        if prob < np.random.sample():
            continue
        subsamples.append(word)

    return subsamples, vocab, word_to_idx, idx_to_word


def gather_training_data(
    split_text: List[str], word_to_idx: Dict[str, int], context_size: int
) -> List[Tuple[int, int]]:
    """Return training data that consists of [word, context]."""
    training_data: List[Tuple[int, int]] = []
    for i, word in enumerate(split_text):
        back_i, back_c = i - 1, 0
        forw_i, forw_c = i + 1, 0
        while back_i >= 0 and back_c < context_size:
            training_data.append((word_to_idx[word], word_to_idx[split_text[back_i]]))
            back_i -= 1
            back_c += 1
        while forw_i < len(split_text) and forw_c < context_size:
            training_data.append((word_to_idx[word], word_to_idx[split_text[forw_i]]))
            forw_i += 1
            forw_c += 1
    return training_data


def load_data(
    filename: str, context_size: int, sampling_ratio: float = 1e-3
) -> Tuple[
    List[str], Dict[str, int], Dict[str, int], Dict[int, str], List[Tuple[int, int]]
]:
    """Load data for training."""
    with open(filename, "rb") as file:
        processed_text = word_tokenize(file.read().decode("utf-8").strip())
        processed_text, vocab, word_to_idx, idx_to_word = gather_word_freqs(
            processed_text, sampling_ratio
        )
        training_data = gather_training_data(processed_text, word_to_idx, context_size)
    return processed_text, vocab, word_to_idx, idx_to_word, training_data


class SkipGramSampler:
    """Skip-gram sampler."""

    def __init__(
        self, filename: str, context_size: int, sampling_ratio: float = 1e-3
    ) -> None:
        """Init."""
        _, self.vocab, self.word_to_idx, self.idx_to_word, data = load_data(
            filename, context_size, sampling_ratio
        )
        data = np.array(data)
        self.words, self.contexts = data[:, 0], data[:, 1]
        self.num_words = len(self.idx_to_word)
        dist = (
            np.array([self.vocab[self.idx_to_word[i]] for i in range(len(self.vocab))])
            ** 0.75
        )
        self.dist = dist / np.sum(dist)

    def sample(
        self, batch_size: int = 32, negative_sample_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample data."""
        words, contexts, negative_samples = [], [], []
        for _ in range(batch_size):
            idx = np.random.randint(len(self.words), size=1)[0]
            word, context = self.words[idx], self.contexts[idx]
            negative_sample = self._get_negative_samples(
                negative_sample_size, set([word, context])
            )
            words.append(word)
            contexts.append(context)
            negative_samples.append(negative_sample)
        return np.array(words), np.array(contexts), np.array(negative_samples)

    def _get_negative_samples(
        self, negative_sample_size: int, positives: Set[int]
    ) -> List[int]:
        """Get negative samples."""
        samples = []
        for _ in range(negative_sample_size):
            w = np.random.choice(self.dist.shape[0], p=self.dist)
            while w in positives:
                w = np.random.choice(self.dist.shape[0], p=self.dist)
            samples.append(w)
        return samples


if __name__ == "__main__":
    sampler = SkipGramSampler("trainset.txt", context_size=2)
    words, contexts, negative_samples = sampler.sample()
    print("words:", words.shape)
    print("contexts:", contexts.shape)
    print("negative samples:", negative_samples.shape)
