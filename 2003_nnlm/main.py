"""Neural Network Language Model.

- Paper: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
- Summary:
    - A fundamental problem of language modeling by curse of dimensioinality.
    - Authors suggest:
        - A distributed representation for each word.
        - A probability function for word sequences.
- Reference: https://github.com/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM.py
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def make_batch(
    sentences: List[str],
    word_to_idx: Dict[str, int],
) -> Tuple[List[List[int]], List[int]]:
    """Make a batch for n-gram model training."""
    input_batch: List[List[int]] = []
    target_batch: List[int] = []

    for sentence in sentences:
        words = sentence.split()
        inputs = [word_to_idx[n] for n in words[:-1]]
        target = word_to_idx[words[-1]]

        input_batch.append(inputs)
        target_batch.append(target)

    return input_batch, target_batch


class NNLM(nn.Module):
    """NNLM."""

    def __init__(self, n_class: int, n_step: int, n_hidden: int, m: int) -> None:
        """Init."""
        super().__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward: y = b + Wx + U tanh(d + Hx)."""
        X = self.C(X)
        batch_size, n_step, m = X.size()
        X = X.view(batch_size, n_step * m)
        activation = torch.tanh(self.d + self.H(X))
        output = self.b + self.W(X) + self.U(activation)
        return output


if __name__ == "__main__":
    sentences = [
        "i like dog",
        "i love coffee",
        "i hate milk",
    ]
    words = list(set(" ".join(sentences).split()))
    word_to_idx = {w: i for i, w in enumerate(words)}
    idx_to_word = {i: w for i, w in enumerate(words)}

    model = NNLM(n_class=len(words), n_step=2, n_hidden=2, m=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch(sentences, word_to_idx)
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}\tLoss: {loss:.6f}")

        loss.backward()
        optimizer.step()

    # test
    predict = model(input_batch).data.max(1, keepdim=True)[1].squeeze()
    for sentence, target_idx in zip(sentences, predict):
        print(f"{' '.join(sentence.split()[:2])} => {idx_to_word[target_idx.item()]}")
