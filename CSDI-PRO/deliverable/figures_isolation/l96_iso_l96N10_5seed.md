
### Isolation table — VPT10 mean ± sd  [Pr(>0.5)]

| Cell (imputer → forecaster) | S2 | S3 | S4 | S5 |
|:---|:-:|:-:|:-:|:-:|
| Linear → Panda-72M | 2.00 ± 1.85  [80%] | 1.80 ± 1.73  [60%] | 0.77 ± 1.02  [40%] | 0.03 ± 0.07  [0%] |
| Kalman → Panda-72M | 2.13 ± 1.53  [80%] | 1.50 ± 1.84  [40%] | 0.32 ± 0.56  [20%] | 0.18 ± 0.37  [20%] |
| CSDI (ours) → Panda-72M | 2.20 ± 1.60  [80%] | 1.86 ± 1.70  [60%] | 1.88 ± 1.05  [80%] | 0.54 ± 0.95  [20%] |
| Linear → DeepEDM (delay-manifold) | 0.94 ± 0.55  [60%] | 0.12 ± 0.07  [0%] | 0.20 ± 0.33  [20%] | 0.03 ± 0.07  [0%] |
| Kalman → DeepEDM (delay-manifold) | 0.66 ± 0.51  [60%] | 0.30 ± 0.35  [20%] | 0.32 ± 0.52  [20%] | 0.00 ± 0.00  [0%] |
| CSDI (ours) → DeepEDM (delay-manifold) | 1.08 ± 0.64  [100%] | 0.55 ± 0.53  [20%] | 0.87 ± 0.41  [80%] | 0.87 ± 1.12  [40%] |


### Paired-bootstrap headline contrasts (Δ = mean(csdi) - mean(linear))

| Forecaster | Scenario | Δ VPT@1.0 | 95% paired CI | sign |
|:---|:---|:-:|:-:|:-:|
| Panda-72M | S2 | +0.20 | [-0.20, +0.84] | ≈ |
| Panda-72M | S3 | +0.07 | [-0.17, +0.29] | ≈ |
| Panda-72M | S4 | +1.11 | [+0.08, +2.22] | ↑ |
| Panda-72M | S5 | +0.50 | [+0.00, +1.41] | ≈ |
| DeepEDM (delay-manifold) | S2 | +0.13 | [-0.20, +0.40] | ≈ |
| DeepEDM (delay-manifold) | S3 | +0.44 | [+0.08, +0.99] | ↑ |
| DeepEDM (delay-manifold) | S4 | +0.67 | [+0.30, +1.04] | ↑ |
| DeepEDM (delay-manifold) | S5 | +0.84 | [+0.00, +1.85] | ≈ |
