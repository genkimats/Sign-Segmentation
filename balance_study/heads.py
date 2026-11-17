"""
Per-stream BIO head used by the modality-imbalance methods (balance_study).

One head per stream reads that stream's branch output z_s (B, T, D) and predicts phrase B/I/O by itself.
It is training-only: the model's prediction never uses it. The heads exist because most imbalance methods
need a unimodal prediction, and the multi-stream model has none (the streams meet before the BiLSTM).

A 1-layer BiLSTM gives the head the same kind of temporal context the real model has (a k=9 conv, like the
auxiliary heads, sees only ~0.3 s, which would make every stream look weaker than it is). With
hidden = d_model the final Linear has the same shape as the model's classifier (2*d_model -> 3), which
CGGM's direction term needs.
"""
import torch.nn as nn


class StreamBIOHead(nn.Module):
    """(B, T, D) stream features -> (B, 3, T) logits."""

    def __init__(self, in_dim, hidden=256, n_layers=1, dropout=0.2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=n_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, num_classes)

    def forward(self, z):
        h, _ = self.lstm(z)
        return self.classifier(self.drop(h)).permute(0, 2, 1)
