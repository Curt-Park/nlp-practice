"""Skip-gram model training."""

import torch
from data import SkipGramSampler
from model import SkipGramModel

if __name__ == "__main__":
    sampler = SkipGramSampler("trainset.txt", context_size=2)
    model = SkipGramModel(len(sampler.vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 32

    for epoch in range(5):
        n_iter = len(sampler.words) // batch_size
        total_loss = 0.0
        for _ in range(n_iter):
            words, contexts, negative_samples = sampler.sample(batch_size)
            optimizer.zero_grad()
            loss = model(
                torch.LongTensor(words),
                torch.LongTensor(contexts),
                torch.LongTensor(negative_samples),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch}\tloss: {total_loss / n_iter: .6f}")

    model.save_embeddings(sampler.idx_to_word)
