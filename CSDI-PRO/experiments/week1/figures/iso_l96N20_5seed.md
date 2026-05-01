
### Isolation table — VPT10 mean ± sd  [Pr(>0.5)]

| Cell (imputer → forecaster) | S2 | S3 | S4 | S5 |
|:---|:-:|:-:|:-:|:-:|
| Linear → Panda-72M | 2.32 ± 2.42  [80%] | 1.46 ± 1.85  [80%] | 0.52 ± 0.48  [60%] | 0.00 ± 0.00  [0%] |
| Kalman → Panda-72M | 2.69 ± 3.12  [80%] | 1.31 ± 1.94  [60%] | 0.35 ± 0.31  [40%] | 0.00 ± 0.00  [0%] |
| CSDI (ours) → Panda-72M | 2.44 ± 1.91  [100%] | 2.47 ± 1.83  [100%] | 3.60 ± 3.73  [100%] | 0.40 ± 0.81  [20%] |
| Linear → DeepEDM (delay-manifold) | 0.40 ± 0.25  [60%] | 0.24 ± 0.23  [20%] | 0.08 ± 0.13  [0%] | 0.00 ± 0.00  [0%] |
| Kalman → DeepEDM (delay-manifold) | 0.24 ± 0.13  [0%] | 0.25 ± 0.25  [40%] | 0.17 ± 0.15  [0%] | 0.00 ± 0.00  [0%] |
| CSDI (ours) → DeepEDM (delay-manifold) | 0.44 ± 0.22  [40%] | 0.71 ± 0.24  [80%] | 0.72 ± 0.41  [60%] | 0.18 ± 0.23  [20%] |


### Paired-bootstrap headline contrasts (Δ = mean(csdi) - mean(linear))

| Forecaster | Scenario | Δ VPT@1.0 | 95% paired CI | sign |
|:---|:---|:-:|:-:|:-:|
| Panda-72M | S2 | +0.12 | [-0.52, +0.82] | ≈ |
| Panda-72M | S3 | +1.01 | [+0.12, +2.50] | ↑ |
| Panda-72M | S4 | +3.07 | [+0.57, +6.45] | ↑ |
| Panda-72M | S5 | +0.40 | [+0.00, +1.21] | ≈ |
| DeepEDM (delay-manifold) | S2 | +0.03 | [-0.25, +0.42] | ≈ |
| DeepEDM (delay-manifold) | S3 | +0.47 | [+0.05, +0.84] | ↑ |
| DeepEDM (delay-manifold) | S4 | +0.64 | [+0.20, +1.09] | ↑ |
| DeepEDM (delay-manifold) | S5 | +0.18 | [+0.00, +0.39] | ≈ |
