import logging
import numpy as np

logger = logging.getLogger("trading_firm.lstm")

SEQ_LEN    = 32
HIDDEN     = 128
N_LAYERS   = 2
DROPOUT    = 0.3
EPOCHS     = 30
BATCH_SIZE = 128
LR         = 1e-3


def _to_sequences(X: np.ndarray, seq_len: int):
    n = len(X)
    valid_idx = np.arange(seq_len - 1, n)
    X_seq = np.stack([X[i - seq_len + 1: i + 1] for i in valid_idx])
    return X_seq, valid_idx


# Module-level class so pickle can serialize it
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
            out    = self.drop(out[:, -1, :])
            return self.fc(out)

    _TORCH_AVAILABLE = True

except Exception:
    _TORCH_AVAILABLE = False
    _Net = None


class LSTMClassifier:
    def __init__(
        self,
        n_features:   int   = 1,
        n_classes:    int   = 3,
        seq_len:      int   = SEQ_LEN,
        hidden:       int   = HIDDEN,
        n_layers:     int   = N_LAYERS,
        dropout:      float = DROPOUT,
        epochs:       int   = EPOCHS,
        batch_size:   int   = BATCH_SIZE,
        lr:           float = LR,
        random_state: int   = 42,
    ):
        self.n_features   = n_features
        self.n_classes    = n_classes
        self.seq_len      = seq_len
        self.hidden       = hidden
        self.n_layers     = n_layers
        self.dropout      = dropout
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state
        self._model       = None
        self._device      = None
        self._has_torch   = False
        self.classes_     = np.array([0, 1, 2])

    def _build_model(self):
        if not _TORCH_AVAILABLE or _Net is None:
            return None
        return _Net(
            self.n_features, self.n_classes,
            self.hidden, self.n_layers, self.dropout
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — LSTM neutral proba")
            return self
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
            self._has_torch = True
        except Exception as e:
            logger.warning(f"PyTorch import failed: {e}")
            return self

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self._device    = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_features = X.shape[1]

        X_seq, valid_idx = _to_sequences(X, self.seq_len)
        y_seq = y[valid_idx]

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=False,
        )

        self._model = self._build_model().to(self._device)
        optimizer   = torch.optim.Adam(
            self._model.parameters(), lr=self.lr)
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
                logger.debug(
                    f"LSTM epoch {epoch+1}/{self.epochs} "
                    f"loss={epoch_loss/len(loader):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n   = len(X)
        out = np.full((n, self.n_classes), 1.0 / self.n_classes)

        if not self._has_torch or self._model is None:
            return out

        # Guard: not enough rows to form even one sequence
        if n < self.seq_len:
            return out

        try:
            import torch
            import torch.nn.functional as F

            try:
                X_seq, valid_idx = _to_sequences(X, self.seq_len)
            except (ValueError, IndexError):
                return out

            if len(X_seq) == 0:
                return out

            X_t = torch.tensor(
                X_seq, dtype=torch.float32).to(self._device)

            self._model.eval()
            with torch.no_grad():
                logits = self._model(X_t)
                probs  = np.array(
                    F.softmax(logits, dim=-1)
                    .cpu().detach().tolist(),
                    dtype=np.float32,
                )
            out[valid_idx] = probs
        except Exception as e:
            logger.warning(f"LSTM predict_proba failed: {e}")

        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
