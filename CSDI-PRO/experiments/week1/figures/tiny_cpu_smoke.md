# Corruption Grid Summary (vpt10)

| config | mask | s | sigma | keep | obs/patch | max gap L | cell | n | mean | median | sd | Pr>0 | Pr>0.5 |
|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| SP00 | iid_time | 0.00 | 0.00 | 1.000 | 16.00 | 0.00 | deepedm_linear | 1 | 0.09 | 0.09 | 0.00 | 1.00 | 0.00 |

Design read: obs/patch and max gap L are the first sanity checks. If adjacent stages barely change either quantity, the stage ladder is too fine; if they jump by several Lyapunov tenths at once, it is too coarse.
