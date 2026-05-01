# Corruption Grid Summary (vpt10)

| config | mask | s | sigma | keep | obs/patch | max gap L | cell | n | mean | median | sd | Pr>0 | Pr>0.5 |
|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| H0 | iid_time | 0.00 | 0.00 | 1.000 | 16.00 | 0.00 | metadata_only | 0 | - | - | - | - | - |
| H1 | iid_time | 0.20 | 0.05 | 0.758 | 12.12 | 0.07 | metadata_only | 0 | - | - | - | - | - |
| H2 | iid_time | 0.40 | 0.10 | 0.584 | 9.34 | 0.18 | metadata_only | 0 | - | - | - | - | - |
| H3 | iid_time | 0.65 | 0.20 | 0.354 | 5.66 | 0.20 | metadata_only | 0 | - | - | - | - | - |
| H4 | iid_time | 0.82 | 0.35 | 0.168 | 2.69 | 0.50 | metadata_only | 0 | - | - | - | - | - |
| H5 | iid_time | 0.90 | 0.60 | 0.098 | 1.56 | 0.77 | metadata_only | 0 | - | - | - | - | - |
| H6 | iid_time | 0.95 | 1.00 | 0.059 | 0.94 | 1.31 | metadata_only | 0 | - | - | - | - | - |

Design read: obs/patch and max gap L are the first sanity checks. If adjacent stages barely change either quantity, the stage ladder is too fine; if they jump by several Lyapunov tenths at once, it is too coarse.
