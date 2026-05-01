# Panda Representation-Space OOD Diagnostic (L63)

Distance is matched-to-clean paired L2 unless noted.

## SP65

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 0.5114 | 0.0305 | 16.772 | yes |
| embed | 103.1416 | 8.0304 | 12.844 | yes |
| encoder | 63.8614 | 4.5542 | 14.023 | yes |
| pooled | 9.2160 | 0.4219 | 21.845 | yes |

## SP82

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 1.6108 | 0.6645 | 2.424 | yes |
| embed | 215.7204 | 88.8453 | 2.428 | yes |
| encoder | 120.0615 | 73.5127 | 1.633 | yes |
| pooled | 14.2598 | 7.3012 | 1.953 | yes |

## Verdict

Verdict: Panda representation distances support the tokenizer/OOD mechanism in a majority of tested scenario-stage cells.
