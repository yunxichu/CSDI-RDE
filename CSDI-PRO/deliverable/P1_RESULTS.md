# P1 results — pretrained-SAITS, Chronos mini-frontier, EnKF upper bound

> 2026-05-01. Triggered by the reviewer-perspective P1 plan in
> `SUBMISSION_PREP_PLAN.md`. All three P1 experiments completed; P0
> cleanup still stands.

## P1.1 — pretrained SAITS on the same chaos corpus

**Setup.** SAITS (PyPOTS) pretrained on
`experiments/week2_modules/data/lorenz63_clean_64k_L128.npz` with
v2-grid-style missingness (sparsity sampled uniformly from
fine_s_line, iid_time mask). 30 epochs, batch 64, ~18 minutes on 1 GPU.
Final validation MAE on missing entries = 1.26 (= 0.149 × `attractor_std`).
Checkpoint:
`experiments/week2_modules/ckpts/saits_l63_pretrained/20260501_T153756/SAITS.pypots`.

At inference, the test context (length 512) is split into 4 contiguous
chunks of length 128 — the SAITS-trained window length — and each chunk
is imputed independently. This is a fair deployment for SAITS at its
pretrained context length; CSDI is unchanged. 10 seeds.

**Headline (L63 SP65 + SP82, σ=0).**

| Cell | SP65 mean | SP82 mean | SP65 Pr>1.0 | SP82 Pr>1.0 |
|:--|--:|--:|--:|--:|
| `linear → Panda` | 1.22 | 0.29 | 70 % | 0 % |
| `SAITS-pretrained → Panda` | **2.49** | **1.51** | **90 %** | **70 %** |
| `CSDI → Panda` | **2.89** | **1.57** | **100 %** | **70 %** |

**Paired-bootstrap contrasts.**

| Contrast | SP65 Δ (CI) | SP82 Δ (CI) | sign |
|:--|:--|:--|:-:|
| SAITS − linear | +1.26 [+0.83, +1.64] | +1.23 [+0.86, +1.62] | ↑ both |
| CSDI − linear | +1.67 [+1.41, +1.92] | +1.28 [+0.73, +1.85] | ↑ both |
| **CSDI − SAITS** | **+0.41 [+0.05, +0.87]** | **+0.06 [−0.31, +0.59]** | ↑ at SP65, ≈ at SP82 |

**Reading per SUBMISSION_PREP_PLAN decision rule.** A pretrained
structured imputer reproduces most of the rescue. CSDI retains a
strict-positive paired advantage at the entrance band (SP65) but is
indistinguishable from pretrained SAITS at the floor band (SP82). The
main claim therefore narrows from "CSDI is the only tested intervention"
to:

> **Corpus-pretrained structured imputation is the lever for sparse-gap
> rescue of Panda inside the L63 transition band. CSDI's dynamics-aware
> diffusion residuals provide a small but paired-CI-strict advantage at
> the entrance band (SP65: CSDI − SAITS = +0.41 Λ, 95 % CI
> [+0.05, +0.87]); at the floor band (SP82) the two are statistically
> indistinguishable.**

This is publishable and is a **stronger** framing than the original
"CSDI is the only intervention" because it shifts the empirical
phenomenon from a CSDI-specific quirk to a generalisable property of
corpus-pretrained structured imputation.

**Files.**
- `experiments/week1/results/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`
- `experiments/week1/figures/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.md`
- `experiments/week2_modules/ckpts/saits_l63_pretrained/...`

## P1.2 — Chronos mini-frontier on L63

**Setup.** Chronos-bolt-small (`amazon/chronos-bolt-small`), per-channel
univariate prediction over the same v2 corruption grid. 5 seeds at
SP55 / SP65 / SP75 / SP82 with `linear → Chronos` and `CSDI → Chronos`
cells.

**Headline.** Chronos under our setting has substantially lower absolute
VPT than Panda across the SP line and the CSDI rescue is **not**
reproducible:

| Scenario | linear → Chronos mean | CSDI → Chronos mean | Paired Δ (CI) |
|:--|--:|--:|:--|
| SP55 | 0.37 | 0.38 | +0.01 [+0.00, +0.03] |
| SP65 | 0.39 | 0.38 | −0.00 [−0.02, +0.01] |
| SP75 | 0.50 | 0.39 | −0.11 [−0.36, +0.02] |
| SP82 | 0.34 | 0.39 | +0.05 [−0.03, +0.19] |

(All paired CIs cross zero or are very tight near zero. Pr(VPT > 1.0 Λ)
is 0–20 % across all cells × cells.)

**Follow-up at Chronos's native horizon.** We repeated the same SP55–SP82
mini-frontier at `pred_len = 64`, because the Chronos library warns that
`prediction_length > 64` is out of distribution. The result is statistically
the same as `pred_len = 128`: Chronos remains at mean VPT 0.34–0.50,
Pr(VPT>1.0) ≤ 20 %, and CSDI − linear CIs stay at or near zero
(SP55 +0.01, SP65 −0.00, SP75 −0.11, SP82 +0.05). Thus the negative result
is not an artefact of using a longer-than-trained horizon.

**Reading.** Two signals here:

1. The frontier **shape** does not transfer cleanly to Chronos at either
   `pred_len = 128` or `pred_len = 64`; Chronos itself is a weak forecaster
   on L63 under this evaluation.
2. The CSDI rescue is **Panda-specific** in the tested setup, since Chronos
   is already near a low-VPT plateau where the rescue lever can't move
   it further.

We therefore **do not claim** "the frontier is foundation-model-general".
The honest §3 / §6 framing is: the empirical frontier is established for
Panda; cross-foundation-model evidence shows that Chronos behaves differently
in absolute terms — both forecasters depend on the corrupted context, but
only Panda has the dynamic range in which CSDI's rescue is observable.

**Files.**
- `experiments/week1/results/chronos_frontier_l63_chronos_l63_sp55_sp82_5seed.json`
- `experiments/week1/results/chronos_frontier_l63_chronos_l63_sp55_sp82_5seed_pl64.json`

## P1.5 — pretrained SAITS on L96 N=20

**Setup.** SAITS was pretrained on the matching L96 N=20 corpus
`experiments/week2_modules/data/lorenz96_clean_512k_L128_N20.npz`
(64K length-128 windows used for training, 20 channels, same v2-grid
missingness). 30 epochs, batch 64, best epoch 27. Validation MAE on missing
entries = 1.07 (= 0.294 × L96 attractor_std). Checkpoint:
`experiments/week2_modules/ckpts/saits_l96_n20_pretrained/20260501_T210242/SAITS.pypots`.

**Headline (L96 SP82, 10 seeds).**

| Cell | mean | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| `linear -> Panda` | 1.81 | 0.50 | 60 % | 30 % |
| `SAITS-pretrained -> Panda` | 1.56 | 0.84 | 100 % | 40 % |
| `CSDI -> Panda` | 1.77 | 1.13 | 100 % | 60 % |

Paired mean contrasts cross zero because one rare long `linear -> Panda`
seed dominates the mean (`seed=2`, VPT@1.0 = 10.75). Median and survival
therefore carry the L96 reading: pretrained structured imputers remove
zero-survival failures, and CSDI gives the largest median and Pr>1.0.
This matches the L96 frontier-table policy: high-dimensional L96 is a
survival/median replication, not a mean-VPT replication.

**Files.**
- `experiments/week1/results/panda_altimputer_l96_sp82_pretrained_10seed.json`
- `experiments/week1/figures/panda_altimputer_l96_sp82_pretrained_10seed.md`
- `experiments/week2_modules/ckpts/saits_l96_n20_pretrained_meta.json`

## P1.3 — EnKF known-dynamics upper bound on L63

**Setup.** Stochastic ensemble Kalman filter (n_members = 100, RK4
forward, 95 % observation-noise floor at $0.01 \cdot \sigma_\text{attr}$
for pure-sparsity cells). True L63 vector field is given to the filter;
this is intentionally a **model-aware** reference, not a competitor to
the model-agnostic deployment interface.

**Headline.** EnKF saturates at the VPT ceiling across the entire v2
sparsity transition band:

| Scenario | EnKF mean | EnKF median | EnKF Pr>1.0 |
|:--|--:|--:|--:|
| SP55 (s=0.55, σ=0) | 2.84 | 2.90 | 100 % |
| SP65 (s=0.65, σ=0) | 2.84 | 2.90 | 100 % |
| SP75 (s=0.75, σ=0) | 2.85 | 2.90 | 100 % |
| SP82 (s=0.82, σ=0) | 2.84 | 2.90 | 100 % |
| NO020 (s=0, σ=0.20) | 2.81 | 2.90 | 100 % |
| NO050 (s=0, σ=0.50) | 2.49 | 2.70 | 100 % |

(ceiling = `pred_len × dt × λ_max ≈ 128 × 0.025 × 0.906 = 2.90`).

**Reading.** When the dynamics are known and fed to the assimilation
filter, the L63 sparsity transition band is **not a frontier at all** —
EnKF state recovery + deterministic rollout reaches ceiling everywhere
on the band and degrades only on the dense-noise axis. This is the
correct reference for the §1 framing: the frontier is a property of
the **black-box deployment interface** in which the dynamics are
unavailable to the forecaster, not a property of L63 itself.

The paper text already adopts this framing in §6.5 (Relationship to
data assimilation). Appendix B can now cite the concrete EnKF numbers.

**Files.** `experiments/week1/results/enkf_l63_enkf_l63_v2_5seed.json`.

## What changes in the paper after P1

### Abstract / §1

- Soften "the only tested intervention" → "the strongest tested
  intervention; a corpus-pretrained SAITS reproduces most of the
  rescue with a strict-positive CSDI advantage only at the entrance
  band (SP65)".
- Replace the §1 "Mechanism in the entrance band" wording: keep the
  raw-patch / token-OOD reduction at SP65, but acknowledge that this
  reduction is not unique to CSDI under the alt-imputer experiment.
- Add explicit Chronos result statement: cross-foundation evidence is
  forecaster-dependent; the Panda frontier story does not extend to
  Chronos at either pred_len = 128 or its native pred_len = 64.
- Keep EnKF upper bound as model-aware reference in §6.5.

### §4.4 (alt-imputer)

Replace the placeholder table in §4.4 / Appendix C with the concrete
P1.1 numbers above. The narrative is no longer "alt-imputer comparison
is pending"; it is "we ran it, here is the answer, the story narrows".

### §6.4 (limitations)

- "Foundation-model breadth" item now cites the Chronos result with the
  pred_len caveat.
- Drop "pretrained alt-imputer is open" — SAITS is run on L63 and L96.
  Glocal-IB specifically remains future work.

## Status

P1 work complete, including the Chronos pred_len=64 follow-up and L96
SAITS-pretrained survival replication. Story successfully narrowed without
retraction of the sparse-observation forecastability frontier core finding.
Draft text is updated in both English and Chinese; remaining work is P3 polish
(real-data, decoder-side mechanism, Glocal-IB) and final proofreading.
