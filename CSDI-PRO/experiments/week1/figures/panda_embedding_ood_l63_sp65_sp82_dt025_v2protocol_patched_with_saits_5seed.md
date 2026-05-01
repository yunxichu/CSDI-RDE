# Panda Representation-Space OOD Diagnostic (L63)

Distance is matched-to-clean paired L2 unless noted.

## SP65

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 0.5114 | 0.0343 | 14.926 | yes |
| embed | 103.1416 | 8.5910 | 12.006 | yes |
| encoder | 63.8614 | 5.4302 | 11.760 | yes |
| pooled | 9.2160 | 0.5054 | 18.235 | yes |

## SP82

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 1.6108 | 0.7061 | 2.281 | yes |
| embed | 215.7204 | 93.5402 | 2.306 | yes |
| encoder | 120.0615 | 75.8477 | 1.583 | yes |
| pooled | 14.2598 | 6.7631 | 2.108 | yes |

## Verdict

Verdict: Panda representation distances support the tokenizer/OOD mechanism in a majority of tested scenario-stage cells.
