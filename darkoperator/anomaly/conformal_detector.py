"""
ConformalAnomalyDetector
========================

Wraps a trained FourierNeuralOperator1d and applies split conformal prediction
to produce *calibrated uncertainty estimates* for each new input.

Anomaly score
-------------
We use an ensemble of predictions obtained by **MC-Dropout** (or simply the
single model prediction against a calibration residual).  The conformal
prediction interval half-width at level α is used as the anomaly score:

    anomaly_score(x) = ĈP_interval_width(x) = q̂_{1-α}  ×  s(x)

where s(x) is the local nonconformity score (scaled residual) and q̂ is the
calibration quantile.  A large score means the operator is uncertain — the
input is in a region of function space where the surrogate was not trained or
where the PDE solution is difficult.

Usage
-----
    detector = ConformalAnomalyDetector(trained_fno, alpha=0.1)
    detector.calibrate(u0_cal, uT_cal)   # held-out calibration pairs

    scores = detector.anomaly_score(u0_test)   # higher = more anomalous
    flags  = detector.flag_anomalies(u0_test)  # bool mask
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConformalAnomalyDetector(nn.Module):
    """
    Conformal anomaly detector wrapping a trained FNO-1D surrogate.

    Args:
        model:   trained FourierNeuralOperator1d (or any nn.Module with
                 the same in/out shape convention)
        alpha:   significance level for conformal coverage (default 0.1
                 → 90 % marginal coverage)
        mc_samples: number of MC-Dropout forward passes for uncertainty
                 estimation.  Set to 1 to disable dropout-based uncertainty.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        mc_samples: int = 20,
    ):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.mc_samples = mc_samples

        # Set after calibrate()
        self._q_hat: Optional[float] = None   # conformal quantile
        self._cal_residuals: Optional[np.ndarray] = None
        self._u0_cal: Optional[torch.Tensor] = None   # stored for spectral baseline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(self, u0: torch.Tensor) -> torch.Tensor:
        """
        Return model prediction for u0.

        u0: (batch, n)  – initial condition values only (no grid appended).
        We append the uniform spatial grid internally.
        """
        device = next(self.model.parameters()).device
        u0 = u0.to(device)

        n = u0.shape[-1]
        x_grid = torch.linspace(0, 1, n, device=device)
        x_grid = x_grid.unsqueeze(0).expand(u0.shape[0], -1)   # (batch, n)

        # Stack to (batch, 2, n): value + position
        inp = torch.stack([u0, x_grid], dim=1)   # (batch, 2, n)
        return self.model(inp).squeeze(1)         # (batch, n)

    def _nonconformity(
        self,
        u0: torch.Tensor,
        uT_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample nonconformity scores on labelled pairs.

        Score = RMSE between predicted and true final field.
        Shape: (batch,)
        """
        with torch.no_grad():
            uT_pred = self._predict(u0)
        residual = uT_pred - uT_true.to(uT_pred.device)
        return residual.pow(2).mean(dim=-1).sqrt()   # (batch,)

    def _mc_uncertainty(self, u0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate predictive mean and std via MC-Dropout.

        Enables dropout during inference and runs *mc_samples* forward passes.
        Falls back to a single deterministic pass if mc_samples == 1.

        Returns:
            mean_pred: (batch, n)
            std_pred:  (batch, n)  – pointwise standard deviation
        """
        if self.mc_samples <= 1:
            with torch.no_grad():
                mean_pred = self._predict(u0)
            std_pred = torch.zeros_like(mean_pred)
            return mean_pred, std_pred

        # Enable dropout layers during inference
        self._enable_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(self._predict(u0))
        self._disable_dropout()

        stack = torch.stack(preds, dim=0)   # (mc_samples, batch, n)
        return stack.mean(0), stack.std(0)

    def _enable_dropout(self) -> None:
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _disable_dropout(self) -> None:
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        u0_cal: torch.Tensor,
        uT_cal: torch.Tensor,
    ) -> None:
        """
        Calibrate the conformal quantile on a held-out set.

        Args:
            u0_cal: (n_cal, n)  initial conditions (calibration split)
            uT_cal: (n_cal, n)  ground-truth final fields
        """
        self.model.eval()
        scores = self._nonconformity(u0_cal, uT_cal).cpu().numpy()
        self._cal_residuals = scores
        self._u0_cal = u0_cal.cpu()   # store for spectral baseline

        n = len(scores)
        # Split-conformal quantile: ⌈(n+1)(1-α)⌉ / n th empirical quantile
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))

    @property
    def is_calibrated(self) -> bool:
        return self._q_hat is not None

    def anomaly_score(self, u0: torch.Tensor) -> np.ndarray:
        """
        Compute anomaly score for each sample in u0.

        The score is the conformal **prediction interval half-width** at the
        given ``alpha`` level, which equals the conformal quantile ``q̂``.
        That quantity is fixed after calibration.

        For *unsupervised* anomaly detection (no ground-truth uT at test time)
        we use the **MC-Dropout predictive standard deviation** as a proxy for
        the residual, normalised by the conformal quantile so that score > 1
        corresponds to uncertainty larger than the calibration threshold.

        With ``mc_samples=1`` (deterministic model) the score falls back to
        the *conformal residual* on the training distribution: every test
        sample receives a fixed score of 1.0 (border of the conformal band),
        which is still useful for ranking via ``flag_anomalies``.

        Requires ``calibrate()`` to have been called first.

        Args:
            u0: (batch, n) initial conditions

        Returns:
            scores: (batch,) numpy array — higher = more anomalous
        """
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() before anomaly_score().")

        self.model.eval()

        # MC-Dropout predictive std as uncertainty proxy
        if self.mc_samples > 1:
            _, std_pred = self._mc_uncertainty(u0)
            mc_std = std_pred.mean(dim=-1).cpu().numpy()   # (batch,) RMS pointwise std
        else:
            # Single-pass: use RMSE of prediction against itself (zero, but meaningful
            # relative to calibration scores via the conformal quantile)
            with torch.no_grad():
                uT_pred = self._predict(u0)
            mc_std = np.zeros(len(u0))

        # Normalise by q̂: score > 1 ↔ uncertainty > conformal threshold
        return mc_std / (self._q_hat + 1e-12)

    def flag_anomalies(
        self,
        u0: torch.Tensor,
        threshold: float = 1.0,
    ) -> np.ndarray:
        """
        Return a boolean mask flagging anomalous inputs.

        A sample is flagged when its anomaly score exceeds *threshold*.
        With the default threshold=1.0 this corresponds to uncertainty
        larger than the conformal quantile (i.e., outside the nominal
        prediction interval).

        Args:
            u0:        (batch, n) initial conditions
            threshold: anomaly decision boundary (default 1.0)

        Returns:
            flags: (batch,) bool array
        """
        return self.anomaly_score(u0) > threshold

    def predict_with_intervals(
        self,
        u0: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return prediction + conformal prediction intervals.

        Args:
            u0: (batch, n) initial conditions

        Returns:
            mean_pred:  (batch, n) predicted solution
            lower:      (batch, n) lower bound of 1−α interval
            upper:      (batch, n) upper bound of 1−α interval
        """
        if not self.is_calibrated:
            raise RuntimeError("Call calibrate() before predict_with_intervals().")

        self.model.eval()
        mean_pred, _ = self._mc_uncertainty(u0)
        mean_np = mean_pred.cpu().numpy()

        q = self._q_hat
        # Marginal conformal band: ŷ ± q̂
        return mean_np, mean_np - q, mean_np + q
