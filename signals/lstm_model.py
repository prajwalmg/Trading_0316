"""
================================================================
  signals/lstm_model.py
  PyTorch LSTM — sklearn-compatible wrapper for stacked ensemble.

  Architecture:
    Input  : (batch, seq_len, n_features) — last seq_len bars
    LSTM   : 2 layers, 128 hidden units, 0.3 dropout
    Output : 3-class softmax (sell / flat / buy)

  Labels  : encoded as [0, 1, 2] → decoded to [-1, 0, 1]

  Sklearn API:
    fit(X, y)          — X shape (n_samples, n_features), flat 2-D
    predict_proba(X)   — returns (n_samples, n_classes)
    predict(X)         — returns encoded class array [0,1,2]

  Sequencing:
    Internally reshape each sample into (seq_len, n_features)
    by taking the preceding seq_len rows from X.
    Samples without a full history window are dropped.
================================================================
"""
import logging
import numpy as np

logger = logging.getLogger("trading_firm.lstm")

SEQ_LEN    = 32      # number of bars per sequence
HIDDEN     = 128     # LSTM hidden units
N_LAYERS   = 2       # stacked LSTM depth
DROPOUT    = 0.3
EPOCHS     = 30
BATCH_SIZE = 128
LR         = 1e-3


def _to_sequences(X: np.ndarray, seq_len: int):
    """
    Convert flat 2-D feature matrix to 3-D sequences.

    Returns
    -------
    X_seq : (n_valid, seq_len, n_features)
    valid_idx : indices into original X that have a full window
    """
    n = len(X)
    valid_idx = np.arange(seq_len - 1, n)
    X_seq = np.stack([X[i - seq_len + 1: i + 1] for i in valid_idx])
    return X_seq, valid_idx


class LSTMClassifier:
    """
    Sklearn-compatible LSTM classifier built on PyTorch.
    Falls back gracefully to random outputs if PyTorch is not installed.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_classes:  int = 3,
        seq_len:    int = SEQ_LEN,
        hidden:     int = HIDDEN,
        n_layers:   int = N_LAYERS,
        dropout:    float = DROPOUT,
        epochs:     int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr:         float = LR,
        random_state: int = 42,
    ):
        self.n_features  = n_features
        self.n_classes   = n_classes
        self.seq_len     = seq_len
        self.hidden      = hidden
        self.n_layers    = n_layers
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_state = random_state

        self._model     = None
        self._device    = None
        self._has_torch = False
        self.classes_   = np.array([0, 1, 2])   # encoded classes

    # ── PyTorch model definition ──────────────────────────────

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn

            class _Net(nn.Module):
                def __init__(self, n_feat, n_cls, hidden, n_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=n_feat,
                        hidden_size=hidden,
                        num_layers=n_layers,
                        batch_first=True,
                        dropout=dropout if n_layers > 1 else 0.0,
                    )
                    self.drop = nn.Dropout(dropout)
                    self.fc   = nn.Linear(hidden, n_cls)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    out    = self.drop(out[:, -1, :])   # last time-step
                    return self.fc(out)

            return _Net(
                self.n_features, self.n_classes,
                self.hidden, self.n_layers, self.dropout
            )
        except ImportError:
            return None

    # ── fit ───────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
            self._has_torch = True
        except ImportError:
            logger.warning("PyTorch not installed — LSTM will output neutral proba")
            return self

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = X.shape[1]

        X_seq, valid_idx = _to_sequences(X, self.seq_len)
        y_seq = y[valid_idx]

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=False,   # preserve time order
        )

        self._model = self._build_model().to(self._device)
        optimizer   = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion   = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"LSTM epoch {epoch+1}/{self.epochs} "
                             f"loss={epoch_loss/len(loader):.4f}")

        return self

    # ── predict_proba ─────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        out = np.full((n, self.n_classes), 1.0 / self.n_classes)

        if not self._has_torch or self._model is None:
            return out

        try:
            import torch
            import torch.nn.functional as F

            X_seq, valid_idx = _to_sequences(X, self.seq_len)
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)

            self._model.eval()
            with torch.no_grad():
                logits = self._model(X_t)
                probs  = F.softmax(logits, dim=-1).cpu().numpy()

            out[valid_idx] = probs
        except Exception as e:
            logger.warning(f"LSTM predict_proba failed: {e}")

        return out

    # ── predict ───────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
