# Panda Representation-Space OOD Diagnostic (L63)

Distance is matched-to-clean paired L2 unless noted.

## SP65

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 0.5114 | 0.0637 | 8.025 | yes |
| embed | 103.1416 | 16.8645 | 6.116 | yes |
| encoder | 63.8614 | 9.5393 | 6.695 | yes |
| pooled | 9.2160 | 1.0419 | 8.846 | yes |

## SP82

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 1.6108 | 1.5750 | 1.023 | yes |
| embed | 215.7204 | 202.9649 | 1.063 | yes |
| encoder | 120.0615 | 138.1111 | 0.869 | no |
| pooled | 14.2598 | 13.6619 | 1.044 | yes |

## Verdict

Verdict: Panda representation distances support the tokenizer/OOD mechanism in a majority of tested scenario-stage cells.
