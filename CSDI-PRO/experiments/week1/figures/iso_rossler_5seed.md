
### Isolation table — VPT10 mean ± sd  [Pr(>0.5)]

| Cell (imputer → forecaster) | S2 | S3 | S4 | S5 |
|:---|:-:|:-:|:-:|:-:|
| Linear → Panda-72M | 0.58 ± 0.19  [80%] | 0.50 ± 0.17  [60%] | 0.39 ± 0.25  [40%] | 0.06 ± 0.08  [0%] |
| Kalman → Panda-72M | 0.55 ± 0.17  [80%] | 0.46 ± 0.19  [60%] | 0.29 ± 0.25  [20%] | 0.27 ± 0.26  [20%] |
| CSDI (ours) → Panda-72M | 0.65 ± 0.25  [80%] | 0.59 ± 0.22  [80%] | 0.58 ± 0.23  [80%] | 0.31 ± 0.23  [20%] |
| Linear → DeepEDM (delay-manifold) | 0.34 ± 0.20  [20%] | 0.13 ± 0.05  [0%] | 0.08 ± 0.04  [0%] | 0.02 ± 0.03  [0%] |
| Kalman → DeepEDM (delay-manifold) | 0.25 ± 0.19  [20%] | 0.19 ± 0.11  [0%] | 0.16 ± 0.15  [0%] | 0.08 ± 0.09  [0%] |
| CSDI (ours) → DeepEDM (delay-manifold) | 0.34 ± 0.17  [40%] | 0.35 ± 0.18  [20%] | 0.33 ± 0.18  [20%] | 0.35 ± 0.22  [40%] |


### Paired-bootstrap headline contrasts (Δ = mean(csdi) - mean(linear))

| Forecaster | Scenario | Δ VPT@1.0 | 95% paired CI | sign |
|:---|:---|:-:|:-:|:-:|
| Panda-72M | S2 | +0.07 | [+0.00, +0.16] | ↑ |
| Panda-72M | S3 | +0.10 | [+0.00, +0.19] | ↑ |
| Panda-72M | S4 | +0.19 | [-0.02, +0.56] | ≈ |
| Panda-72M | S5 | +0.25 | [-0.02, +0.52] | ≈ |
| DeepEDM (delay-manifold) | S2 | -0.00 | [-0.29, +0.26] | ≈ |
| DeepEDM (delay-manifold) | S3 | +0.22 | [+0.11, +0.40] | ↑ |
| DeepEDM (delay-manifold) | S4 | +0.26 | [+0.10, +0.45] | ↑ |
| DeepEDM (delay-manifold) | S5 | +0.34 | [+0.13, +0.54] | ↑ |
