"""Skip-gram model."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip-gram model."""

    def __init__(self, vocab_size: int, emb_size: int = 200) -> None:
        """Init."""
        super().__init__()
        self.emb_size = emb_size
        self.u_embeddings = nn.Embedding(vocab_size, emb_size)
        self.v_embeddings = nn.Embedding(vocab_size, emb_size)
        self._init_emb()

    def _init_emb(self) -> None:
        """Initialize embeddings."""
        init_range = 0.5 / self.emb_size
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(0, 0)

    def forward(
        self, pos_u: torch.Tensor, pos_v: torch.Tensor, neg_v: torch.Tensor
    ) -> torch.Tensor:
        """Forward."""
        emb_u = self.u_embeddings(pos_u)  # (batch_size, emb_size)
        emb_v = self.v_embeddings(pos_v)  # (batch_size, emb_size)
        emb_neg = self.v_embeddings(neg_v)  # (batch_size, neg_sample_size, emb_size)

        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))

    def save_embeddings(
        self, idx_to_word: Dict[int, str], filename: str = "word_vectors.txt"
    ) -> None:
        """Save all embeddings to a file."""
        embedding = self.u_embeddings.weight.data.numpy()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{len(idx_to_word)} {self.emb_size}\n")
            for idx, word in idx_to_word.items():
                e = embedding[idx]
                e = " ".join(map(lambda x: str(x), e))
                f.write(f"{word} {e}\n")


if __name__ == "__main__":
    model = SkipGramModel(10, 100)
    idx_to_word = {i: str(i) for i in range(10)}
    model.save_embeddings(idx_to_word)
