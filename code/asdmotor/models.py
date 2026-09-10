"""Sequence models sized for a 334-sample corpus.

Capacity is deliberately small. With 334 samples over 32 subjects, a model with
more than a few tens of thousands of parameters memorises subjects rather than
learning motor structure, and cross-validated variance swamps any effect. Both
networks below sit in the 5k-60k parameter range and are compared against
non-deep baselines that are given exactly the same folds.

Masking is honoured everywhere: padded frames never contribute to a convolution
output that is pooled, nor to the GRU's summary state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the time axis of ``(B, T, C)`` using ``(B, T)`` bool mask."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class TinyTCN(nn.Module):
    """Small dilated temporal convolutional network with causal-free padding.

    Two or three residual blocks of (Conv1d -> BatchNorm -> ReLU -> Dropout),
    exponentially increasing dilation, then masked global average pooling.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        channels: int = 32,
        n_blocks: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        in_c = n_features
        for b in range(n_blocks):
            d = 2 ** b
            pad = d * (kernel_size - 1) // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_c, channels, kernel_size, padding=pad, dilation=d),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_c = channels
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                       # (B, C, T)
        m = mask.unsqueeze(1).to(h.dtype)
        for blk in self.blocks:
            h = h * m                               # keep padding from bleeding in
            h = blk(h)
        h = h.transpose(1, 2)                       # (B, T, C)
        return self.head(masked_mean(h, mask))


class TinyGRU(nn.Module):
    """Small GRU over packed sequences, summarised by a masked mean of outputs."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden: int = 48,
        n_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(self.drop(masked_mean(out, mask)))


MODEL_REGISTRY = {"tcn": TinyTCN, "gru": TinyGRU}


@dataclass(frozen=True, slots=True)
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    #: 64 rather than 32: with ~270 training sequences the GRU's per-timestep
    #: overhead dominates on CPU, and 64 is ~1.8x faster for the same schedule.
    batch_size: int = 64
    patience: int = 10
    seed: int = 0


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_model(
    model: nn.Module,
    x_tr: np.ndarray,
    m_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    m_va: np.ndarray,
    y_va: np.ndarray,
    cfg: TrainConfig,
    class_weight: np.ndarray | None = None,
) -> tuple[nn.Module, dict]:
    """Train with early stopping on validation loss; restore the best state."""
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    xt = torch.tensor(x_tr, dtype=torch.float32, device=dev)
    mt = torch.tensor(m_tr, dtype=torch.bool, device=dev)
    yt = torch.tensor(y_tr, dtype=torch.long, device=dev)
    xv = torch.tensor(x_va, dtype=torch.float32, device=dev)
    mv = torch.tensor(m_va, dtype=torch.bool, device=dev)
    yv = torch.tensor(y_va, dtype=torch.long, device=dev)

    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n = len(xt)
    best, best_state, bad, best_epoch = float("inf"), None, 0, 0
    g = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            if len(idx) < 2:                     # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(xt[idx], mt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(xv, mv), yv))
        if vloss < best - 1e-5:
            best, bad, best_epoch = vloss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # Hand back a CPU model: predict_proba (and anything else downstream)
    # builds its input tensors on the default device and never moves the
    # model, so training may run on CUDA but inference stays device-agnostic.
    model = model.to("cpu")
    return model, {"best_val_loss": best, "best_epoch": best_epoch, "epochs_run": epoch + 1}


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.tensor(x, dtype=torch.float32), torch.tensor(m, dtype=torch.bool))
    return torch.softmax(logits, dim=1).numpy()
