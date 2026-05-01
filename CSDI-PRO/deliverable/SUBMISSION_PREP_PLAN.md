# Submission Preparation Plan — 2026-05-01

> Built from a NeurIPS / ICLR 2026 reviewer-perspective read of the locked
> story draft. Verdict on the as-of-2026-05-01 draft: **weak reject** (3
> blocking issues + several moderates).
>
> This document supersedes `PAPER_STATUS_2026-05-01.md` § "Immediate Next
> Step" — those items (figure index, experiment table, supplementary
> compression) are **completed**. Going forward, the submission gate is
> the four P0 cleanups + three P1 experiments listed below.

## Reviewer-perspective verdict (one paragraph)

The empirical evidence is solid and the controls are well thought out, but
three reviewer-killer issues remain: (i) Appendix C (pretrained alt-imputer)
is not run, so the §4.4 dismissal of SAITS / BRITS is correctly self-flagged
as biased and an alert reviewer will reject the alt-imputer story outright;
(ii) the headline numbers carry "draft / must be refreshed" caveats in the
frontmatter and abstract, signalling an unfinished submission; (iii) Panda is
the only foundation forecaster fully evaluated, so the mechanism claim looks
like it could be Panda-tokenizer-specific. Each of these three is sufficient
on its own to drop a borderline accept to a weak reject. None of the three
is a research-direction problem; they are submission-prep problems.

## Locked story (unchanged)

> Sparse observations create a sharp forecastability frontier on the
> sparsity axis for pretrained chaotic forecasters. Inside the transition
> band, CSDI-style corruption-aware *gap* imputation moves Panda back
> across that frontier; in the entrance band the rescue coincides with
> reduced raw-patch and Panda-token OOD, while at the floor band
> distance-to-clean alone no longer fully accounts for the residual
> survival gain. CSDI is a sparse-gap imputation lever, not a generic
> dense-noise denoiser. Delay-manifold forecasting (DeepEDM in Takens
> coordinates) is a complementary dynamics-aware route, with explicit
> scope boundaries on Mackey-Glass and Chua.

The four reviewer-defeating qualifiers are unchanged:

1. **Sparse-observation** frontier — not all corruption.
2. **Inside the transition band** — not universal improvement.
3. **Sparse-gap imputation lever** — not dense-noise denoising.
4. **Structured residual matters** — iid noise of matched magnitude is not a
   substitute for Pr(VPT > 1.0 Λ).

## P0 — must do before submission (cleanup, no GPU)

### P0.1 Strip internal-facing notes from `paper_draft_en.md` and `paper_draft_zh.md`

Remove every phrase that signals "this is a working draft" or
"this is review-defense scaffolding":

- The frontmatter `> Locked under STORY_LOCK_2026-04-28.md ...` block — entire
  reference to lock files, dual-model collab, "reviewer-defeating qualifiers
  must remain", etc.
- The abstract caveat "Current headline numerical evidence ... (to be
  refreshed under the patched CSDI sigma_override protocol before
  submission)".
- The "deferred" framing in Appendix C; replace with a forward-looking
  description of what the alt-imputer experiment measures (we will have
  actually run it, see P1.1).
- Any "old draft archive" pointer in the body of the paper. Archive
  references are appropriate in commit history and supplementary, not in
  the main text.

The Chinese mirror gets the same edits.

### P0.2 Freeze the patched headline numbers

Single source of truth: the patched JSONs listed in Appendix B
(`*_patched_*.json`). Every cited number in the abstract / §3 / §4 / §6 /
Appendix B / Appendix D must match those JSONs.

Critical change relative to the existing draft:

- The abstract currently cites "L96 N=20 SP82: 0.91 → 3.31 mean,
  CI [+0.10, +6.59], Pr(VPT > 1.0) 20% → 60%". This is the **pre-patch
  mean-headline** explicitly disowned by `PAPER_STATUS_2026-05-01.md`
  §4.D. Replace with the patched-n=10 statement: median 0.50 → 1.05,
  Pr(VPT > 0.5) 60% → 100%, DeepEDM paired Δ +0.43 Λ
  CI [+0.29, +0.57]. Acknowledge L96 mean is high-variance and
  intentionally not the headline.
- §3.2, §4.1, §4.3 cross-check against the patched table.
- Appendix D figure paths use the `_patched_` files only.

### P0.3 Simplify the three-regime taxonomy

The current §1 / §4.3 split into "entrance-band CSDI / generic-regularization
/ floor-band CSDI" + a fourth pure-noise regime reads like an *ad hoc*
counter-example shield to a skeptical reviewer. Replace with empirical
descriptions of each (system, scenario) cell:

- L63 SP65 (σ=0): CSDI Δ = +1.65 Λ [+1.39, +1.91]; iid jitter Δ ≈ 0.
- L63 SP82 (σ=0): CSDI Δ = +1.09 Λ [+0.65, +1.61]; iid jitter Δ crosses 0.
- L96 SP65 (σ=0): mean is recovered by jitter as well as CSDI; tail
  Pr(VPT > 1.0) is 60% (CSDI) vs 40% (jitter) vs 20% (linear).
- L96 SP82 (σ=0, n=10): mean is high-variance; we report median 0.50→1.05
  and Pr(VPT > 0.5) 60→100%; DeepEDM paired Δ is the cleaner cross-system
  signal.
- Pure-noise (s=0, σ>0): CSDI is neutral or slightly hurtful; this is the
  intentional scope boundary.

Stop calling these "regimes". Stop using "high-dimensional generic
regularization regime" as a name; just describe what L96 SP65 actually shows
and what tail metric discriminates CSDI there.

### P0.4 Demote per-instance SAITS / BRITS to appendix sanity

Currently §4.4 reports the per-instance result *and* explicitly admits the
comparison is "biased against SAITS / BRITS". A reviewer will read that
self-disclosure and discount the entire alt-imputer table. Move the
per-instance numbers to an appendix paragraph framed as a sanity check on
"high-missingness imputation requires a global prior", and have §4.4 cite
the **pretrained** SAITS result (which P1.1 produces) as the actual
reviewer-defense.

### P0.5 Add Wilson CI to headline survival numbers

`Pr(VPT > 1.0) = 100%` at n = 10 has Wilson 95% interval [69.2%, 100%]; at
n = 5 it is [56.6%, 100%]. Without a CI, "100%" reads as stronger than it
is. Same for the L96 / Rössler 5-seed survival numbers. Do this in the
abstract, §3, §4.3, and Appendix D / B tables.

### P0.6 Soften §3 title and theory-leaning language

The "Sharp Forecastability Frontiers" title and parts of §3 still carry
faintly theoretical language. Either:

- rename §3 "Empirical Forecastability Frontiers" and remove "phase-transition-like" / "transition band" wherever they read theoretical;
- or fold a short narrowed-Theorem-2 statement back into the body, and earn
  the theoretical language.

Default is the first option; second option is only if Appendix A's narrowed
bound is rewritten as a half-page main-text proposition.

### P0.7 Reposition DeepEDM

The current §1 contributions list DeepEDM as a major pillar, but §4.1
admits `CSDI → Panda` is the strongest absolute cell in several places.
Two options:

- demote DeepEDM to "complementary route" with a single, defensible claim
  (e.g. "the DeepEDM CSDI − linear paired Δ is the only strict-positive
  paired CI on L96 N=20 across SP55–SP82") and pull it out of the
  contribution headlines;
- or strengthen DeepEDM's role with a comparison case it uniquely wins
  (e.g. Rössler floor band).

Default is the first option.

## P1 — experiments before submission (must do)

### P1.1 Pretrained SAITS on L63 SP65 + SP82

Train a SAITS model on the same chaos corpus that CSDI was trained on
(`experiments/week2_modules/data/`). At least the L63 corpus, with a
matched missingness distribution.

Evaluate at L63 SP65 and SP82 with cells `linear → Panda`,
`SAITS-pretrained → Panda`, `CSDI → Panda`. 10 seeds.

Decision rule:

- if SAITS recovers comparable rescue (paired Δ overlaps CSDI within 95% CI),
  the main claim narrows to "structured / corpus-pretrained imputation is
  the lever; CSDI is one strong instance"; the paper survives;
- if SAITS underperforms CSDI (paired Δ strictly below CSDI Δ),
  the dynamics-aware diffusion prior is doing measurable work;
- if SAITS fully matches or exceeds CSDI on L63 (unlikely given our small
  L63 corpus but possible), §4.4 needs a careful "structured imputation as
  a class is the lever" rewrite without retracting the frontier or
  intervention claim.

All three outcomes are publishable. The unacceptable outcome is **not running
this experiment**.

Output: `panda_altimputer_l63_sp65_sp82_pretrained_10seed.json` plus a
`deliverable/EXPERIMENT_C1_RESULTS.md` summary.

### P1.2 Chronos mini-frontier on L63

Run Chronos under the same v2 corruption grid as Figure 1, but only on the
sparsity transition band: SP55, SP65, SP75, SP82. Cells:
`linear → Chronos`, `CSDI → Chronos`. 5 seeds (10 if cheap).

Decision rule:

- if Chronos shows a sharp frontier in the same SP cells, §3.2 cross-system
  claim is upgraded from "Panda only" to "two foundation forecasters";
- if Chronos shows a frontier in different cells, §3.2 should report it as
  forecaster-dependent;
- if Chronos shows no frontier, §3.2 should explicitly say so and limit the
  intervention claim to the Panda-Chronos union of failure cells, with
  Chronos as a forecaster that is already robust to this corruption pattern.

Whichever the result, this single experiment removes the "Panda-only"
attack surface.

Output: `pt_l63_grid_v2_chronos_5seed.json`, plus a small Figure 1b panel
in `deliverable/figures_main/`.

### P1.3 EnKF / LETKF as known-dynamics upper bound

Implement a basic EnKF for L63 (and optionally LETKF for L96 N=20). Use
the same masks and noise as the v2 grid. Report VPT per cell.

Frame as a model-aware upper bound, **not** a competitor. The paper's text
becomes: "When the dynamics are known, EnKF achieves VPT ≈ X across the
transition band. Our setting is the black-box deployment interface, where
that information is not available; CSDI gives the best model-agnostic
forecastability we measured." This converts an apparent omission into a
deliberate framing decision.

Output: `enkf_l63_v2_5seed.json`, table in §3 or Appendix B.

## P2 — strongly recommended but not strictly blocking

- Reposition DeepEDM cleanly (covered in P0.7).
- Wilson CI on every survival probability (covered in P0.5).
- Title change for §3 (covered in P0.6).
- Per-instance SAITS / BRITS demotion (covered in P0.4).

## P3 — optional polish, post-submission OK

- Real-data case study (ECG, EEG, climate reanalysis).
- Panda decoder-side instrumentation for the floor-band mechanism.
- Glocal-IB pretrained as third alt-imputer.
- KSE / dysts breadth on §3.

## Execution order

1. **P0.1 → P0.2 → P0.3 → P0.4 → P0.5 → P0.6 → P0.7** in `paper_draft_en.md`.
2. Mirror to `paper_draft_zh.md` (P0.4 of plan = "Chinese mirror").
3. **P1.1** (pretrained SAITS): designed in `EXPERIMENT_C_PLAN.md`,
   resources budget ~6–12 h pretraining + ~30 min eval on 1 GPU.
4. **P1.2** (Chronos mini-frontier): need to install / locate Chronos; if
   already in `baselines/`, ~30 min eval. Otherwise the install is the
   bottleneck.
5. **P1.3** (EnKF upper bound): pure-CPU, ~1–2 h to write the runner +
   30 min eval.
6. Update P0 documents with P1 results.
7. Final cross-check pass (no internal notes, all numbers consistent, no
   "deferred" anywhere in the body).

## Stop conditions / reverts

- If P1.1 shows pretrained SAITS fully matches CSDI on **both** L63 SP65
  and SP82, retract the "CSDI is the only tested intervention" abstract
  sentence; rewrite as "corpus-pretrained structured imputation is the
  lever". Add a paragraph in §4.4 explaining the equivalence.
- If P1.2 shows Chronos has no frontier on the same cells, retract any
  "pretrained chaotic forecasters" plural framing in §1; restrict the
  empirical claim to Panda explicitly.
- If P1.3 EnKF gets VPT ≈ ceiling at every cell, the framing of "black-box
  deployment matters" stands; if EnKF is itself fragile, include that
  observation neutrally and note it as future work.

## Out of scope for this submission

- Reframing the story away from sparse-observation frontier. The story is
  locked.
- Adding new systems (KSE, dysts) to the headline.
- Replacing Panda as the headline forecaster.
- Re-running the pre-pivot 4-module pipeline experiments.

These are deliberate scope choices, not gaps.
