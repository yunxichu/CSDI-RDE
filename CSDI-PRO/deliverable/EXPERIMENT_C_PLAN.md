# Experiment C — alt-imputer reviewer defense (revised 2026-04-30)

> **Revision:** the original plan ran SAITS / BRITS *per-instance from
> scratch* against pretrained CSDI. That comparison is unfair and was
> killed mid-run. C is now split into C0 (cheap appendix sanity) and C1
> (real reviewer-defense, requires SAITS/Glocal-IB pretraining on the
> same chaos corpus that CSDI used).

## Why the original C is downgraded

CSDI was pretrained on a 512K-window L63 chaos corpus (and equivalent for
L96). SAITS and BRITS as run today were trained from scratch on a single
512-step test trajectory with ~36 % observed values. There is no global
prior available to them in that setting, so the comparison answers a
*different* question than "is structured imputation per se the lever".

Independent sanity check: SAITS / BRITS imputations in our partial run do
respect the observed positions (anchored, no clobbering) — so the failure
is genuine high-missingness fitting from one short trajectory, not a
driver bug.

## Current state

- `panda_altimputer_control.py` exists and runs the full 4-cell matrix.
  Locked at v2 protocol (`LORENZ63_ATTRACTOR_STD = 8.51`,
  `lorenz96_attractor_std(N=20, F=8) = 3.6387`, grid-index seeds).
- L63 SP65 phase (5 seeds × 4 cells) ran to completion before the kill;
  L63 SP82 only got two records before kill; L96 SP82 not started.
- Glocal-IB **not** integrated into the script.
- Run was stopped at PID 3772278 to free GPU 0.

## C0 — appendix sanity (already done, save as such)

Status: **complete from log salvage**.

L63 SP65 5 seeds, v2 protocol, Panda forecaster:

| Cell | mean | std | median | Pr>0.5 | Pr>1.0 |
|---|---:|---:|---:|---:|---:|
| linear | 1.29 | 0.52 | 1.29 | 80 % | 80 % |
| **SAITS (per-instance)** | **0.15** | 0.15 | 0.11 | 0 % | 0 % |
| **BRITS (per-instance)** | **0.16** | 0.16 | 0.09 | 0 % | 0 % |
| **CSDI (pretrained)** | **2.90** | 0.00 | 2.90 | 100 % | 100 % |

Paired Δ vs linear, 95 % bootstrap:

| Cell | Δ mean | 95 % CI | sign |
|---|---:|---|---:|
| SAITS  | −1.14 | [−1.67, −0.53] | ↓ |
| BRITS  | −1.13 | [−1.66, −0.51] | ↓ |
| CSDI   | +1.61 | [+1.16, +2.09] | ↑ |

CSDI's `2.90` is the VPT ceiling (`pred_len · dt · λ_L63 = 128 · 0.025 · 0.906`);
sanity-checked separately by inspecting forecast amplitude vs truth and
RMSE-vs-horizon (CSDI maintains amplitude, RMSE crosses no threshold within
128 steps; linear damps amplitude and crosses at step ~49).

Files:
- `experiments/week1/results/panda_altimputer_l63sp65_partial_5seed.json`

**How this is used in the paper:** appendix-only, framed as "per-instance
training of generic structured imputers under high missingness collapses
to noise; the global prior matters". Not a positive demonstration of
CSDI uniqueness; explicitly disclosed as biased against SAITS/BRITS.

## C1 — proper reviewer defense (not yet started)

Goal: run SAITS / Glocal-IB (and optionally BRITS) **pretrained on the
same chaos corpus that CSDI used**, then evaluate at the CSDI-decisive
cells under the v2 protocol. Only C1 can support a clean claim of the
form "is corruption-aware imputation in general the lever, or is CSDI
specifically required".

### C1 prerequisites

1. **Locate / re-create the CSDI training corpus** for L63 (the 512K
   independent-IC windows used to train `dyn_csdi_full_v6_center_ep20.pt`).
   Same for L96 (`dyn_csdi_l96_full_c192_vales_best.pt`). Both should be
   in repo or in `experiments/week2_modules/data/`.

2. **PyPOTS SAITS pretraining loop**
   - Train SAITS on a held-out copy of the CSDI corpus with matched
     missingness (iid_time at sparsity range matching v2 grid).
   - Same number of effective parameter updates as CSDI's training.
   - Save checkpoint, document hyperparameters.

3. **Glocal-IB integration**
   - Reference implementation needed; PyPOTS does not have it yet (last
     check 2026-04-30).
   - If port-effort > half a day, defer to follow-up work and cite as
     adjacent prior art only.

### C1 evaluation matrix (after pretraining)

3 settings × 4-5 cells × 5 seeds:

- L63 SP82 (entrance / floor band)
- L96 N=20 SP82 (cross-system, only L96 cell with strict positive CSDI CI)
- (optional) L63 SP65 to compare against C0 sanity

Cells: linear / Kalman / SAITS-pretrained / [Glocal-IB-pretrained] / CSDI.

### C1 success criteria

- Each pretrained alt-imputer either fully matches CSDI within paired CI
  → narrow main claim to "structured imputation is the lever; CSDI is
  one strong instance".
- Or fails to match → strengthen main claim that CSDI's dynamics-aware
  diffusion residuals carry value above generic structured imputation.

Either is publishable.

### C1 estimated cost

- SAITS pretraining on 512K L63 windows: ~6-12 h on 1 GPU
- (Optional) BRITS pretraining: similar
- Glocal-IB integration + pretraining: 1-2 days if reference impl exists
- Evaluation: ~30 min
- Total: 1-2 days of GPU time, likely 0.5-1 day of engineering

## Decision: do not run C1 yet

C1 is the right reviewer-defense experiment but is not on the critical
path for paper §1-§4 writing. Recommended order:

1. **Now**: write paper §2 / §5 / §6 against the locked story. C0 sanity
   is sufficient for an appendix sentence.
2. **Before submission**: do C1 with SAITS-pretrained, optionally
   Glocal-IB. Update §4 / §6 with results.
3. **If reviewer asks**: have C1 ready and reference its result.

Glocal-IB stays in §2 related work as adjacent imputation literature
that motivates "high-missingness imputation needs global structure"; the
paper's claim is downstream of this — that even global-structure
imputers do not automatically improve forecastability without dynamics
priors. Whether Glocal-IB rescues Panda or not is empirically testable
in C1 and is not pre-committed in the paper.

## Pure noise axis (deliberately out of scope for C)

CSDI is a sparse-gap-imputation lever, not a dense-noise denoiser
(Figure 1 noise panel). C1 should not claim alt-imputer comparison on
σ > 0; that is a separate denoising-comparison experiment we do not
prioritize for this paper.
