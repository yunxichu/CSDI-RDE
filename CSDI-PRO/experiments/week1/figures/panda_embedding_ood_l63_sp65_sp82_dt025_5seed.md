# Panda Representation-Space OOD Diagnostic (L63)

Distance is matched-to-clean paired L2 unless noted.

## SP65

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 0.4907 | 0.8787 | 0.558 | no |
| embed | 100.5452 | 181.8848 | 0.553 | no |
| encoder | 62.2283 | 137.9158 | 0.451 | no |
| pooled | 7.4180 | 20.6826 | 0.359 | no |

## SP82

| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |
|:--|--:|--:|--:|:--:|
| patch | 1.2819 | 1.2966 | 0.989 | no |
| embed | 190.0109 | 205.1406 | 0.926 | no |
| encoder | 112.3649 | 143.3224 | 0.784 | no |
| pooled | 12.7734 | 17.1636 | 0.744 | no |

## Verdict

Verdict: Panda representation distances do not support the simple claim that linear-fill is farther from clean than CSDI-fill. The mechanism should be written as a non-obvious puzzle unless a more targeted internal signal is found.
