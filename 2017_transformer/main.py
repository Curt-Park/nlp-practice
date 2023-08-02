"""Transformer example."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def make_batch(
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    sentences: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make a single batch."""
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return (
        torch.LongTensor(input_batch),
        torch.LongTensor(output_batch),
        torch.LongTensor(target_batch),
    )


def get_pad_mask(matrix: torch.Tensor, pad_token: int) -> torch.Tensor:
    """Generate a mask for padding."""
    return matrix == pad_token


class Transformer(nn.Module):
    """Transformer architecture."""

    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        dropout_ratio: float = 0.1,
    ) -> None:
        """Init."""
        super().__init__()

        self.d_model = d_model
        self.positional_encoder = PositionalEncoding(
            d_model=d_model,
            dropout_ratio=dropout_ratio,
        )
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dropout=dropout_ratio,
        )
        self.out = nn.Linear(d_model, n_tokens)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward."""
        # src: (batch_size, src_sequence_length)
        # tgt: (batch_size, tgt_sequence_length)

        # mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1))

        # (batch_size, sequence_length, d_model)
        src = self.embedding(src) * np.sqrt(self.d_model)
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # (sequence_length, batch_size, d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # transformer: (sequence_length, batch_size, n_tokens)
        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        out = self.out(out)

        return out


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(
        self, d_model: int, dropout_ratio: float = 0.1, max_len: int = 5000
    ) -> None:
        """Init."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    sentences = [
        "SOS ich mochte ein bier EOS",
        "SOS i want a beer",
        "i want a beer EOS",
    ]

    src_vocab = {
        "PAD": 0,
        "ich": 1,
        "mochte": 2,
        "ein": 3,
        "bier": 4,
        "SOS": 5,
        "EOS": 6,
    }
    src_vocab_size = len(src_vocab)

    tgt_vocab = {"PAD": 0, "i": 1, "want": 2, "a": 3, "beer": 4, "SOS": 5, "EOS": 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    model = Transformer(
        n_tokens=src_vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(src_vocab, tgt_vocab, sentences)

    # train
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(enc_inputs, dec_inputs)
        outputs = outputs.permute(1, 0, 2).view(-1, tgt_vocab_size)
        loss = criterion(outputs, target_batch.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:2d}\tLoss: {loss.item():.6f}")

    # test
    dec_input = torch.LongTensor([[5]])
    for _ in range(10):
        predict = model(enc_inputs, dec_input).squeeze()
        next_item = predict.topk(1)[1].view(-1)[-1].item()
        next_item = torch.tensor([[next_item]])
        dec_input = torch.cat((dec_input, next_item), dim=1)
        if next_item.view(-1).item() == 6:  # EOS
            break
    print(sentences[0], "->", [number_dict[n] for n in dec_input.view(-1).tolist()])
