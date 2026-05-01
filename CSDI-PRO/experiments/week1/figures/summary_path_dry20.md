# Corruption Grid Summary (vpt10)

| config | mask | s | sigma | keep | obs/patch | max gap L | cell | n | mean | median | sd | Pr>0 | Pr>0.5 |
|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| H0 | iid_time | 0.00 | 0.00 | 1.000 | 16.00 | 0.00 | metadata_only | 0 | - | - | - | - | - |
| H1 | iid_time | 0.20 | 0.05 | 0.797 | 12.76 | 0.08 | metadata_only | 0 | - | - | - | - | - |
| H2 | iid_time | 0.40 | 0.10 | 0.590 | 9.43 | 0.14 | metadata_only | 0 | - | - | - | - | - |
| H3 | iid_time | 0.65 | 0.20 | 0.345 | 5.52 | 0.31 | metadata_only | 0 | - | - | - | - | - |
| H4 | iid_time | 0.82 | 0.35 | 0.179 | 2.86 | 0.51 | metadata_only | 0 | - | - | - | - | - |
| H5 | iid_time | 0.90 | 0.60 | 0.105 | 1.68 | 0.87 | metadata_only | 0 | - | - | - | - | - |
| H6 | iid_time | 0.95 | 1.00 | 0.048 | 0.76 | 1.65 | metadata_only | 0 | - | - | - | - | - |

Design read: obs/patch and max gap L are the first sanity checks. If adjacent stages barely change either quantity, the stage ladder is too fine; if they jump by several Lyapunov tenths at once, it is too coarse.
