
### Isolation table — VPT10 mean ± sd  [Pr(>0.5)]

| Cell (imputer → forecaster) | S2 | S3 | S4 | S5 |
|:---|:-:|:-:|:-:|:-:|
| Linear → Panda-72M | 0.58 ± 0.43  [60%] | 0.43 ± 0.21  [40%] | 0.09 ± 0.13  [0%] | 0.03 ± 0.05  [0%] |
| Kalman → Panda-72M | 0.68 ± 0.36  [60%] | 0.41 ± 0.21  [40%] | 0.17 ± 0.13  [0%] | 0.16 ± 0.16  [0%] |
| CSDI (ours) → Panda-72M | 1.40 ± 0.45  [100%] | 0.83 ± 0.44  [60%] | 0.16 ± 0.15  [0%] | 0.16 ± 0.16  [0%] |
| Linear → DeepEDM (delay-manifold) | 0.47 ± 0.48  [40%] | 0.39 ± 0.22  [40%] | 0.05 ± 0.08  [0%] | 0.00 ± 0.01  [0%] |
| Kalman → DeepEDM (delay-manifold) | 0.35 ± 0.20  [40%] | 0.20 ± 0.10  [0%] | 0.21 ± 0.24  [20%] | 0.12 ± 0.16  [0%] |
| CSDI (ours) → DeepEDM (delay-manifold) | 1.21 ± 0.98  [60%] | 0.72 ± 0.60  [60%] | 0.12 ± 0.14  [0%] | 0.15 ± 0.15  [0%] |


### Paired-bootstrap headline contrasts (Δ = mean(csdi) - mean(linear))

| Forecaster | Scenario | Δ VPT@1.0 | 95% paired CI | sign |
|:---|:---|:-:|:-:|:-:|
| Panda-72M | S2 | +0.82 | [+0.32, +1.37] | ↑ |
| Panda-72M | S3 | +0.40 | [+0.13, +0.67] | ↑ |
| Panda-72M | S4 | +0.07 | [-0.02, +0.19] | ≈ |
| Panda-72M | S5 | +0.13 | [-0.01, +0.29] | ≈ |
| DeepEDM (delay-manifold) | S2 | +0.75 | [+0.30, +1.20] | ↑ |
| DeepEDM (delay-manifold) | S3 | +0.33 | [-0.18, +1.05] | ≈ |
| DeepEDM (delay-manifold) | S4 | +0.07 | [-0.01, +0.14] | ≈ |
| DeepEDM (delay-manifold) | S5 | +0.15 | [+0.02, +0.28] | ↑ |
