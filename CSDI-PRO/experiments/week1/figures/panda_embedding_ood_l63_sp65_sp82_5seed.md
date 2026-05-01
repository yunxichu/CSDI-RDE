# Panda Representation-Space OOD Diagnostic (L63)

Distance is matched-to-clean paired L2 unless noted.

## SP65

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 0.0950 | 0.8378 | 0.113 | no |
| embed | 21.2215 | 151.6025 | 0.140 | no |
| encoder | 16.1163 | 112.5252 | 0.143 | no |
| pooled | 2.1031 | 15.4653 | 0.136 | no |

## SP82

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 0.3162 | 1.2654 | 0.250 | no |
| embed | 53.7487 | 182.4484 | 0.295 | no |
| encoder | 40.5232 | 118.9359 | 0.341 | no |
| pooled | 5.4595 | 13.7137 | 0.398 | no |

## Verdict

Verdict: Panda representation distances do not support the simple claim that linear-fill is farther from clean than CSDI-fill. The mechanism should be written as a non-obvious puzzle unless a more targeted internal signal is found.
