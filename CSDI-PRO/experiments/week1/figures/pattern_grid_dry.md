# Corruption Grid Summary (vpt10)

| config | mask | s | sigma | keep | obs/patch | max gap L | cell | n | mean | median | sd | Pr>0 | Pr>0.5 |
|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| blk08_s60 | block_time | 0.60 | 0.20 | 0.400 | 6.41 | 0.66 | metadata_only | 0 | - | - | - | - | - |
| blk16_s60 | block_time | 0.60 | 0.20 | 0.389 | 6.22 | 2.79 | metadata_only | 0 | - | - | - | - | - |
| blk32_s60 | block_time | 0.60 | 0.20 | 0.395 | 6.31 | 2.49 | metadata_only | 0 | - | - | - | - | - |
| iid_c_s60 | iid_channel | 0.60 | 0.20 | 0.387 | 6.19 | 0.14 | metadata_only | 0 | - | - | - | - | - |
| iid_t_s60 | iid_time | 0.60 | 0.20 | 0.398 | 6.38 | 0.23 | metadata_only | 0 | - | - | - | - | - |
| jit_s80 | periodic_subsample | 0.80 | 0.20 | 0.201 | 3.22 | 0.14 | metadata_only | 0 | - | - | - | - | - |
| mnar_s60 | mnar_curvature | 0.60 | 0.20 | 0.471 | 7.53 | 0.39 | metadata_only | 0 | - | - | - | - | - |
| per_s80 | periodic_subsample | 0.80 | 0.20 | 0.199 | 3.19 | 0.09 | metadata_only | 0 | - | - | - | - | - |

Design read: obs/patch and max gap L are the first sanity checks. If adjacent stages barely change either quantity, the stage ladder is too fine; if they jump by several Lyapunov tenths at once, it is too coarse.
