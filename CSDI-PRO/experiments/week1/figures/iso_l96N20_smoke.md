
### Isolation table — VPT10 mean ± sd  [Pr(>0.5)]

| Cell (imputer → forecaster) | S2 | S3 | S4 | S5 |
|:---|:-:|:-:|:-:|:-:|
| Linear → Panda-72M | 0.50 ± 0.00  [100%] | 0.59 ± 0.00  [100%] | 0.00 ± 0.00  [0%] | 0.00 ± 0.00  [0%] |
| CSDI (ours) → Panda-72M | 0.59 ± 0.00  [100%] | 0.59 ± 0.00  [100%] | 0.34 ± 0.00  [0%] | 0.00 ± 0.00  [0%] |
| Linear → DeepEDM (delay-manifold) | 0.84 ± 0.00  [100%] | 0.17 ± 0.00  [0%] | 0.00 ± 0.00  [0%] | 0.00 ± 0.00  [0%] |
| CSDI (ours) → DeepEDM (delay-manifold) | 0.92 ± 0.00  [100%] | 0.25 ± 0.00  [0%] | 0.50 ± 0.00  [100%] | 0.34 ± 0.00  [0%] |


### Paired-bootstrap headline contrasts (Δ = mean(csdi) - mean(linear))

| Forecaster | Scenario | Δ VPT@1.0 | 95% paired CI | sign |
|:---|:---|:-:|:-:|:-:|
| Panda-72M | S2 | +0.08 | [+0.08, +0.08] | ↑ |
| Panda-72M | S3 | +0.00 | [+0.00, +0.00] | ≈ |
| Panda-72M | S4 | +0.34 | [+0.34, +0.34] | ↑ |
| Panda-72M | S5 | +0.00 | [+0.00, +0.00] | ≈ |
| DeepEDM (delay-manifold) | S2 | +0.08 | [+0.08, +0.08] | ↑ |
| DeepEDM (delay-manifold) | S3 | +0.08 | [+0.08, +0.08] | ↑ |
| DeepEDM (delay-manifold) | S4 | +0.50 | [+0.50, +0.50] | ↑ |
| DeepEDM (delay-manifold) | S5 | +0.34 | [+0.34, +0.34] | ↑ |
