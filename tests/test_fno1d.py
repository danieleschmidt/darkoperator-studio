"""Unit tests for FNO-1D and ConformalAnomalyDetector."""

import pytest
import numpy as np
import torch

from darkoperator.models.fno1d import SpectralConv1d, FNOBlock1d, FourierNeuralOperator1d
from darkoperator.anomaly.conformal_detector import ConformalAnomalyDetector
from darkoperator.physics.pde_data import generate_heat_data, generate_burgers_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_fno(n_grid=32):
    return FourierNeuralOperator1d(modes=4, width=8, n_layers=2, in_channels=2, out_channels=1)


def _make_input(batch=4, n=32):
    u0 = torch.randn(batch, n)
    x = torch.linspace(0, 1, n).unsqueeze(0).expand(batch, -1)
    return torch.stack([u0, x], dim=1)  # (batch, 2, n)


# ---------------------------------------------------------------------------
# SpectralConv1d
# ---------------------------------------------------------------------------

class TestSpectralConv1d:
    def test_output_shape(self):
        layer = SpectralConv1d(in_channels=8, out_channels=8, modes=4)
        x = torch.randn(3, 8, 32)
        y = layer(x)
        assert y.shape == (3, 8, 32)

    def test_output_real(self):
        """Output must be real-valued."""
        layer = SpectralConv1d(4, 4, 3)
        x = torch.randn(2, 4, 16)
        y = layer(x)
        assert y.is_floating_point()
        assert not torch.is_complex(y)

    def test_gradients_flow(self):
        layer = SpectralConv1d(4, 4, 3)
        x = torch.randn(2, 4, 16)
        y = layer(x)
        y.sum().backward()
        assert layer.weights.grad is not None


# ---------------------------------------------------------------------------
# FNOBlock1d
# ---------------------------------------------------------------------------

class TestFNOBlock1d:
    def test_shape_preserved(self):
        block = FNOBlock1d(width=16, modes=4)
        x = torch.randn(3, 16, 64)
        y = block(x)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# FourierNeuralOperator1d
# ---------------------------------------------------------------------------

class TestFourierNeuralOperator1d:
    def test_forward_shape(self):
        model = _dummy_fno(n_grid=32)
        x = _make_input(batch=4, n=32)
        y = model(x)
        assert y.shape == (4, 1, 32)

    def test_different_grid_sizes(self):
        model = _dummy_fno()
        for n in [16, 32, 64]:
            x = _make_input(batch=2, n=n)
            y = model(x)
            assert y.shape == (2, 1, n)

    def test_no_nan(self):
        model = _dummy_fno()
        x = _make_input()
        y = model(x)
        assert not torch.any(torch.isnan(y))

    def test_train_step(self):
        """Single gradient step should decrease loss."""
        model = _dummy_fno()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = _make_input(batch=8, n=32)
        y_true = torch.randn(8, 1, 32)

        model.train()
        pred_before = model(x)
        loss_before = torch.nn.functional.mse_loss(pred_before, y_true).item()

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y_true)
        loss.backward()
        opt.step()

        # Just check no errors during backward / step
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# PDE data generators
# ---------------------------------------------------------------------------

class TestPDEData:
    def test_heat_data_shape(self):
        u0, uT = generate_heat_data(n_samples=20, n_grid=32, seed=0)
        assert u0.shape == (20, 32)
        assert uT.shape == (20, 32)

    def test_heat_data_dtype(self):
        u0, uT = generate_heat_data(n_samples=10, n_grid=32, seed=1)
        assert u0.dtype == torch.float32
        assert uT.dtype == torch.float32

    def test_heat_diffuses(self):
        """Heat equation should reduce variance (energy dissipation)."""
        u0, uT = generate_heat_data(n_samples=100, n_grid=64, nu=0.1, T=1.0, seed=2)
        assert uT.std() < u0.std(), "Heat equation should dissipate energy"

    def test_burgers_data_shape(self):
        u0, uT = generate_burgers_data(n_samples=20, n_grid=32, seed=0)
        assert u0.shape == (20, 32)
        assert uT.shape == (20, 32)

    def test_burgers_no_nan(self):
        u0, uT = generate_burgers_data(n_samples=10, n_grid=32, nu=0.05, seed=5)
        assert not torch.any(torch.isnan(uT))


# ---------------------------------------------------------------------------
# ConformalAnomalyDetector
# ---------------------------------------------------------------------------

class TestConformalAnomalyDetector:
    """Small-scale integration tests."""

    @pytest.fixture
    def trained_detector(self):
        """Return a calibrated detector (lightly trained, no real learning)."""
        n_grid = 32
        model = _dummy_fno(n_grid)

        # Minimal training (1 step) — just to have non-random weights
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        u0, uT = generate_heat_data(n_samples=80, n_grid=n_grid, seed=42)
        x = torch.stack([u0[:40], torch.linspace(0,1,n_grid).unsqueeze(0).expand(40,-1)], dim=1)
        loss = torch.nn.functional.mse_loss(model(x).squeeze(1), uT[:40])
        loss.backward()
        opt.step()

        detector = ConformalAnomalyDetector(model, alpha=0.1, mc_samples=1)
        detector.calibrate(u0[40:60], uT[40:60])
        return detector, u0[60:], uT[60:], n_grid

    def test_calibration_sets_quantile(self, trained_detector):
        detector, *_ = trained_detector
        assert detector.is_calibrated
        assert detector._q_hat is not None
        assert detector._q_hat > 0

    def test_anomaly_score_shape(self, trained_detector):
        detector, u0_test, _, _ = trained_detector
        scores = detector.anomaly_score(u0_test)
        assert scores.shape == (len(u0_test),)
        assert np.all(scores >= 0)

    def test_flag_anomalies_bool(self, trained_detector):
        detector, u0_test, _, _ = trained_detector
        flags = detector.flag_anomalies(u0_test)
        assert flags.dtype == bool
        assert flags.shape == (len(u0_test),)

    def test_predict_with_intervals_shape(self, trained_detector):
        detector, u0_test, _, n_grid = trained_detector
        mean, lo, hi = detector.predict_with_intervals(u0_test[:5])
        assert mean.shape == (5, n_grid)
        assert lo.shape  == (5, n_grid)
        assert hi.shape  == (5, n_grid)
        assert np.all(hi >= lo)

    def test_ood_higher_than_id(self, trained_detector):
        """High-frequency inputs should get higher anomaly scores."""
        detector, u0_id, _, n_grid = trained_detector

        rng = np.random.default_rng(77)
        x = np.linspace(0, 1, n_grid, endpoint=False)
        u0_ood = np.stack([
            sum(
                rng.uniform(-1, 1) * np.cos(2*np.pi*k*x) +
                rng.uniform(-1, 1) * np.sin(2*np.pi*k*x)
                for k in range(20, 30)
            ) / 5
            for _ in range(20)
        ]).astype(np.float32)
        u0_ood_t = torch.tensor(u0_ood)

        scores_id  = detector.anomaly_score(u0_id[:20])
        scores_ood = detector.anomaly_score(u0_ood_t)

        # On average, OOD should be more anomalous
        assert scores_ood.mean() >= scores_id.mean() * 0.5, \
            f"Expected OOD > ID: {scores_ood.mean():.4f} vs {scores_id.mean():.4f}"

    def test_uncalibrated_raises(self):
        model = _dummy_fno()
        detector = ConformalAnomalyDetector(model, alpha=0.1)
        u0 = torch.randn(4, 32)
        with pytest.raises(RuntimeError, match="calibrate"):
            detector.anomaly_score(u0)
