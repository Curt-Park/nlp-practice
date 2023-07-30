"""Text CNN."""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCNN."""

    def __init__(
        self,
        n_vocab: int,
        n_class: int,
        n_channel: int,
        embedding_size: int,
        window_sizes: List[int],
        dropout_rate: float,
        embeddings: nn.Embedding = None,
    ) -> None:
        """Init."""
        super().__init__()
        self.embeddings = nn.Embedding(n_vocab, embedding_size)
        if embeddings is not None:
            self.embeddings.weight.data.copy_(embeddings.weight.data)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, n_channel, kernel_size=(w, embedding_size))
                for w in window_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(window_sizes) * n_channel, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        emb_x = self.embeddings(x)
        emb_x = emb_x.unsqueeze(1)
        # emb_x: (batch_size, 1, sequence_len, embedding_size)

        con_x = [conv(emb_x) for conv in self.convs]
        # con_x: [(batch_size, n_channel, (sequence_len - window_size + 1), 1)]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        # pool_x: [(batch_size, n_channel, 1)]

        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        # fc_x: (batch_size, n_channel x len(windows_sizes))

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        # logit: (batch_size, n_class)

        return logit


if __name__ == "__main__":
    # data
    sentences = [
        "i love you really much",
        "he loves me so much",
        "i think she likes baseball",
        "i hate you so much",
        "i am sorry for that",
        "i think this is awful",
    ]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    word_list = list(set(" ".join(sentences).split()))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    # preparation
    model = TextCNN(
        vocab_size,
        n_class=2,
        n_channel=100,
        embedding_size=300,
        window_sizes=[3, 4, 5],
        dropout_rate=0.5,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001)

    inputs = torch.LongTensor(
        np.array([[word_dict[n] for n in sen.split()] for sen in sentences])
    )
    targets = torch.LongTensor(np.array([out for out in labels]))

    # training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, targets)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch}\tLoss: {loss.item():.6f}")

        loss.backward()
        optimizer.step()
        # the author adds weight rescale if ||w||_2 > 3.

    # test
    test_input = "sorry i really hate you"
    test_batch = torch.LongTensor(
        np.array([[word_dict[n] for n in test_input.split()]])
    )
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_input, "is Negative")
    else:
        print(test_input, "is Positive")
