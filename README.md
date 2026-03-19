# DarkOperator Studio

Neural operator surrogates for 1-D physics simulations, with conformal anomaly detection to identify inputs where the surrogate is uncertain.

---

## What it does

1. **FNO-1D** — A Fourier Neural Operator that learns the solution operator of a 1-D PDE.  
   Given a discretised initial condition `u₀(x)`, it predicts the solution at time `T` without time-stepping through intermediate states.

2. **ConformalAnomalyDetector** — Wraps the trained FNO and applies split conformal prediction.  
   The conformal quantile `q̂` is the RMSE threshold that guarantees ≥ `1-α` marginal coverage on calibration data.  A test sample with residual > `q̂` is flagged as anomalous — the surrogate is in a "dark" region.

---

## Physics context

### Heat equation (training problem)
```
∂u/∂t = ν ∂²u/∂x²,   x∈[0,1],  t∈[0,T]
```
Linear diffusion with periodic BCs.  Solved exactly via FFT (spectral decay).

### Burgers' equation (OOD problem)
```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²,   x∈[0,1],  t∈[0,T]
```
Nonlinear convection-diffusion.  Solved numerically via semi-implicit pseudo-spectral method.  Used as the out-of-distribution test: a surrogate trained on heat equation has higher residuals on Burgers solutions.

---

## Architecture

### FNO-1D

```
Input (batch, 2, n)      ← [u₀(x), x_grid]
    │
    ▼
Lift conv (1×1)          → (batch, width, n)
    │
    ▼  ×n_layers
FNO Block:
  ┌──────────────────────────────┐
  │ SpectralConv1d (FFT modes)   │  (global, low-freq)
  │      +                       │
  │ Conv1d (1×1, pointwise)      │  (local)
  │      +                       │
  │ InstanceNorm + GELU + Dropout│
  └──────────────────────────────┘
    │
    ▼
Project conv (1×1)       → (batch, 1, n)
```

The spectral convolution keeps only the lowest `modes` Fourier frequencies, applies a learned complex-valued linear map, and reconstructs via inverse FFT.  This makes the operator resolution-invariant — you can train at one grid size and evaluate at another.

### Conformal Anomaly Detector

Split conformal prediction (Papadopoulos et al., Vovk et al.):

1. **Calibration** — compute nonconformity scores (RMSE) on held-out `(u0, uT)` pairs.
2. **Quantile** — `q̂ = ⌈(n+1)(1-α)⌉/n` th empirical quantile of calibration scores.
3. **Inference** — flag any test sample with RMSE > `q̂` as anomalous.

Guarantee: under exchangeability, `P(RMSE > q̂)` ≤ `α` for in-distribution inputs.

---

## Quick start

```bash
pip install torch numpy
```

```python
from darkoperator.models.fno1d import FourierNeuralOperator1d
from darkoperator.anomaly.conformal_detector import ConformalAnomalyDetector
from darkoperator.physics.pde_data import generate_heat_data, generate_burgers_data

# 1. Generate data
u0, uT = generate_heat_data(n_samples=1000, n_grid=64, nu=0.01, T=1.0)

# 2. Build and train FNO
model = FourierNeuralOperator1d(modes=12, width=32, n_layers=4)
# ... training loop (see examples/demo_pde_anomaly.py)

# 3. Calibrate anomaly detector on held-out pairs
detector = ConformalAnomalyDetector(model, alpha=0.1)
detector.calibrate(u0_cal, uT_cal)   # held-out split

# 4. Detect anomalies: residuals > q̂ are flagged
scores = detector._nonconformity(u0_test, uT_test).numpy()
flags  = scores > detector._q_hat

# 5. Conformal prediction intervals
mean, lower, upper = detector.predict_with_intervals(u0_test)
```

---

## Demo

```bash
python examples/demo_pde_anomaly.py
```

Expected output (approx):
```
Test MSE (heat eq.):  1.2e-05
Test MSE (Burgers):   7.7e-04  (OOD — 62× worse)
Conformal q̂:          0.0050
Heat-eq flagged:      10/100 (10%)   ← nominal FDR
Burgers flagged:     100/100 (100%)  ← all OOD detected
Pointwise coverage:   90.3%  (target ≥ 90%)
```

---

## Tests

```bash
python -m pytest tests/test_fno1d.py -v
```

19 tests covering:
- `SpectralConv1d` shape, realness, gradient flow
- `FourierNeuralOperator1d` forward pass, variable grid sizes, no NaN
- Heat & Burgers data generators (shape, dtype, physics sanity)
- `ConformalAnomalyDetector` calibration, scoring, interval shapes, coverage

---

## Structure

```
darkoperator/
  models/
    fno1d.py          ← FNO-1D: SpectralConv1d, FNOBlock1d, FourierNeuralOperator1d
  anomaly/
    conformal_detector.py  ← ConformalAnomalyDetector
  physics/
    pde_data.py       ← generate_heat_data, generate_burgers_data
examples/
  demo_pde_anomaly.py ← end-to-end demo
tests/
  test_fno1d.py       ← 19 unit tests
```

---

## References

- Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*, 2023. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, 2005.

---

## License

GPL v3 — see [LICENSE](LICENSE).
