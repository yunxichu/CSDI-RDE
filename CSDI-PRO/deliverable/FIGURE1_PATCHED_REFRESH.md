# Figure 1 Patched Refresh — 2026-05-01

Purpose: refresh the L63 Figure-1 headline grid after the CSDI
`sigma_override` patch. This supersedes the numerical values in
`FIGURE1_LOCKED.md` for any CSDI-dependent cell, but it does not change the
paper story.

Files:
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_h0.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_h5.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_h0.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_h5.json`
- `deliverable/figures_main/figure1_l63_v2_10seed_patched.png`
- `deliverable/figures_main/figure1_l63_v2_10seed_patched.md`

## Main Result

The locked story survives and becomes stronger. Pure-sparsity CSDI cells now
respect exact observed anchors (`sigma_override=0`), and CSDI rescue increases
substantially in the transition/floor band. Dense-noise cells remain neutral or
hurtful for CSDI, supporting the "gap-imputation, not denoising" framing.

## Key Sparsity Cells

Format: mean VPT@1.0 / `Pr(VPT>1.0)`, old -> patched.

| Scenario | Panda linear | Panda CSDI | Patched paired CSDI-linear |
|---|---:|---:|---:|
| SP55 | 1.64/70% -> 1.69/70% | 2.71/100% -> 2.86/100% | +1.17 [+0.60,+1.76] |
| SP65 | 1.22/70% -> 1.22/70% | 2.89/100% -> 2.86/100% | +1.64 [+1.40,+1.87] |
| SP75 | 0.51/20% -> 0.52/20% | 1.16/60% -> 2.29/100% | +1.77 [+1.39,+2.17] |
| SP82 | 0.33/0% -> 0.34/0% | 0.76/20% -> 1.34/60% | +1.00 [+0.54,+1.51] |
| SP88 | 0.24/0% -> 0.24/0% | 0.16/0% -> 0.66/20% | +0.41 [+0.08,+0.73] |
| SP93 | 0.04/0% -> 0.04/0% | 0.12/0% -> 0.37/0% | +0.33 [+0.18,+0.48] |
| SP97 | 0.15/0% -> 0.19/0% | 0.10/0% -> 0.25/0% | +0.06 [-0.21,+0.32] |

DeepEDM also benefits more strongly after patching:

| Scenario | Patched DeepEDM CSDI-linear |
|---|---:|
| SP55 | +0.88 [+0.36,+1.37] |
| SP65 | +0.91 [+0.54,+1.29] |
| SP75 | +0.87 [+0.34,+1.42] |
| SP82 | +1.01 [+0.66,+1.39] |

## Pure Noise

Pure noise remains a non-rescue axis for Panda:

| Scenario | Panda linear old->patched | Panda CSDI old->patched | Patched CSDI-linear |
|---|---:|---:|---:|
| NO010 | 2.16 -> 2.03 | 2.16 -> 2.03 | +0.00 [+0.00,+0.00] |
| NO020 | 1.53 -> 1.53 | 1.40 -> 1.40 | -0.13 [-0.31,+0.00] |
| NO050 | 0.89 -> 0.89 | 0.72 -> 0.72 | -0.17 [-0.33,-0.02] |
| NO080 | 0.57 -> 0.57 | 0.35 -> 0.28 | -0.29 [-0.57,-0.07] |
| NO120 | 0.42 -> 0.42 | 0.13 -> 0.12 | -0.29 [-0.47,-0.12] |

## Decision

No sign flip and no weakening of the locked story. Update the paper headline
numbers to the patched Figure-1 values after the remaining L96/Rössler refresh
lands.
