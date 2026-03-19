#!/usr/bin/env python3
"""
DarkOperator Studio — Demo
==========================

Train a 1-D Fourier Neural Operator on the heat equation, then use
ConformalAnomalyDetector to identify "dark" regions: inputs where
the surrogate residual exceeds the conformal threshold.

Usage
-----
    python examples/demo_pde_anomaly.py

No GPU required; runs in ~30 seconds on a laptop CPU.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from darkoperator.models.fno1d import FourierNeuralOperator1d
from darkoperator.anomaly.conformal_detector import ConformalAnomalyDetector
from darkoperator.physics.pde_data import generate_heat_data, generate_burgers_data


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GRID   = 64
N_TRAIN  = 800
N_CAL    = 200
N_TEST   = 100
EPOCHS   = 40
LR       = 1e-3
BATCH    = 32


# ---------------------------------------------------------------------------
# Data — train/cal/test on heat equation (smooth, low-freq ICs)
# ---------------------------------------------------------------------------
print("Generating heat-equation data …")
u0, uT = generate_heat_data(
    n_samples=N_TRAIN + N_CAL + N_TEST,
    n_grid=N_GRID,
    nu=0.01,
    T=1.0,
    n_modes_ic=10,
    seed=0,
)

u0_train, uT_train = u0[:N_TRAIN],                       uT[:N_TRAIN]
u0_cal,   uT_cal   = u0[N_TRAIN : N_TRAIN + N_CAL],      uT[N_TRAIN : N_TRAIN + N_CAL]
u0_test,  uT_test  = u0[N_TRAIN + N_CAL :],              uT[N_TRAIN + N_CAL :]

# ---------------------------------------------------------------------------
# OOD data — Burgers' equation (different PDE, harder solutions)
# ---------------------------------------------------------------------------
print("Generating Burgers' equation OOD data …")
u0_ood, uT_ood = generate_burgers_data(
    n_samples=N_TEST,
    n_grid=N_GRID,
    nu=0.005,
    T=1.0,
    seed=99,
)


# ---------------------------------------------------------------------------
# Build & train FNO-1D
# ---------------------------------------------------------------------------

def make_input(u0: torch.Tensor) -> torch.Tensor:
    """Append uniform spatial grid: (batch, n) → (batch, 2, n)."""
    n = u0.shape[-1]
    x = torch.linspace(0, 1, n).unsqueeze(0).expand(u0.shape[0], -1)
    return torch.stack([u0, x], dim=1)


print(f"\nTraining FNO-1D on heat equation  (device={DEVICE}) …")
model = FourierNeuralOperator1d(
    modes=12, width=32, n_layers=4, in_channels=2, out_channels=1, dropout=0.1
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

dataset = TensorDataset(make_input(u0_train), uT_train.unsqueeze(1))
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    scheduler.step()
    if epoch % 10 == 0 or epoch == 1:
        avg = total_loss / len(dataset)
        print(f"  Epoch {epoch:3d}/{EPOCHS}  |  train MSE = {avg:.6f}")

# Evaluate on heat-eq test set
model.eval()
with torch.no_grad():
    pred_test = model(make_input(u0_test).to(DEVICE)).squeeze(1).cpu()
test_mse = nn.functional.mse_loss(pred_test, uT_test).item()
print(f"\nTest MSE (heat eq.): {test_mse:.6f}")

# Evaluate on Burgers OOD set
with torch.no_grad():
    pred_ood = model(make_input(u0_ood).to(DEVICE)).squeeze(1).cpu()
ood_mse = nn.functional.mse_loss(pred_ood, uT_ood).item()
print(f"Test MSE (Burgers): {ood_mse:.6f}  ← should be higher than heat-eq MSE")


# ---------------------------------------------------------------------------
# Conformal Anomaly Detector
# ---------------------------------------------------------------------------
print("\nCalibrating ConformalAnomalyDetector on heat-eq calibration set …")
detector = ConformalAnomalyDetector(model, alpha=0.1, mc_samples=30)
detector.calibrate(u0_cal, uT_cal)
print(f"  Conformal quantile q̂ = {detector._q_hat:.4f}  "
      f"(RMSE threshold for 90% coverage)")

# ---------------------------------------------------------------------------
# Anomaly detection via residuals (labelled pairs required)
# ---------------------------------------------------------------------------
print("\n--- Anomaly Detection (labelled) ---")

# NOTE: in production inference is unsupervised; here we use labels to
# demonstrate that the detector's *threshold* meaningfully separates IDs.
scores_id  = detector._nonconformity(u0_test, uT_test).cpu().numpy()
scores_ood = detector._nonconformity(u0_ood,  uT_ood).cpu().numpy()

flagged_id  = (scores_id  > detector._q_hat).sum()
flagged_ood = (scores_ood > detector._q_hat).sum()

print(f"In-distribution heat-eq   ({len(u0_test)} samples):")
print(f"  mean RMSE = {scores_id.mean():.4f}  |  flagged (> q̂) = {flagged_id}/{len(scores_id)}")

print(f"Out-of-distribution Burgers ({len(u0_ood)} samples):")
print(f"  mean RMSE = {scores_ood.mean():.4f}  |  flagged (> q̂) = {flagged_ood}/{len(scores_ood)}")

assert scores_ood.mean() > scores_id.mean(), (
    f"OOD Burgers residuals should exceed heat-eq residuals. "
    f"Got: {scores_ood.mean():.4f} vs {scores_id.mean():.4f}"
)
print("\n✓ Burgers (OOD) has higher residuals than heat-eq (in-distribution).")


# ---------------------------------------------------------------------------
# Conformal prediction intervals
# ---------------------------------------------------------------------------
print("\n--- Conformal Prediction Intervals ---")
mean_pred, lower, upper = detector.predict_with_intervals(u0_test[:5])
widths = (upper - lower).mean(axis=1)
print(f"  Interval width (heat-eq, 5 samples): {widths.mean():.4f} ≈ 2 × q̂")

coverage = np.mean(
    (uT_test[:5].numpy() >= lower) & (uT_test[:5].numpy() <= upper)
)
print(f"  Pointwise coverage: {coverage:.1%}  (target ≥ {1-detector.alpha:.0%})")


# ---------------------------------------------------------------------------
# MC-Dropout anomaly scores (unsupervised — no labels needed)
# ---------------------------------------------------------------------------
print("\n--- Unsupervised Anomaly Scores (MC-Dropout) ---")
mc_scores_id  = detector.anomaly_score(u0_test)
mc_scores_ood = detector.anomaly_score(u0_ood)

print(f"Heat-eq mean anomaly score:  {mc_scores_id.mean():.4f}")
print(f"Burgers mean anomaly score:  {mc_scores_ood.mean():.4f}")

# Report without asserting strict ordering since MC-Dropout signal is soft
print("\n✓ Demo complete.")
print(f"\nSummary:")
print(f"  FNO-1D test MSE (heat eq.):  {test_mse:.2e}")
print(f"  FNO-1D test MSE (Burgers):   {ood_mse:.2e}  (OOD — {ood_mse/test_mse:.0f}× worse)")
print(f"  Conformal q̂:                 {detector._q_hat:.4f}")
print(f"  Heat-eq flagged:             {flagged_id}/{len(scores_id)} ({flagged_id/len(scores_id):.0%})")
print(f"  Burgers flagged:             {flagged_ood}/{len(scores_ood)} ({flagged_ood/len(scores_ood):.0%})")
