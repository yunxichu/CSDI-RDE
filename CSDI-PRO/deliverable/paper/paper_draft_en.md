# Forecastability Frontiers Under Corrupted Chaotic Contexts

**Authors.** (TBD)  **Venue.** NeurIPS / ICLR 2026 target.  **Status.** Locked-story draft, 2026-04-30.

> Locked under `STORY_LOCK_2026-04-28.md` (mechanism reframed 2026-04-30).
> Story-locked drop-in sections live at
> `deliverable/paper/story_locked_sections_{1_3_4,2_5_6}_en.md`.
> The previous narrative (4-module pipeline + universal tokenizer-OOD
> mechanism) is archived at `paper_draft_en_archive_2026-04-30.md`.
>
> Three reviewer-defeating qualifiers must remain in any paraphrase of the
> abstract or §1 opening: (i) **sparse-observation** frontier (not all
> corruption); (ii) **inside the transition band** (rescue is regime-conditional);
> (iii) **structured CSDI residuals are not fully interchangeable with iid
> noise of matched magnitude**, especially in tail survival.
>
> All hard numbers come from JSONs in `experiments/{week1,week2_modules}/results/`.
> Figure references point to `deliverable/figures_*/` and `experiments/week1/figures/`.
> CSDI-dependent headline numbers are draft values from the v2 protocol and
> must be refreshed before submission under the patched `sigma_override`
> imputation protocol documented in `deliverable/CSDI_SANITY_FINDINGS.md`.

---

## Abstract

Pretrained chaotic forecasters fail across sharp sparse-observation forecastability
frontiers. Inside the sparsity transition band, corruption-aware imputation is the
only tested intervention that reliably moves Panda back across the frontier. The
mechanism is strongest in the entrance band: CSDI produces large reductions in
both raw-patch and Panda-token distance to the clean context (linear/CSDI
distance ratios of roughly 12 to 34 across local stdev, lag-1 autocorrelation,
mid-frequency power, and Panda's patch / embedder / encoder / pooled stages on
L63 SP65). Near the floor, Panda-token distances still favor CSDI but one raw
temporal metric becomes mixed, so distance-to-clean is informative but not a
complete account of tail survival. CSDI is
therefore a sparse-gap imputation lever, not a generic dense-noise denoiser, and
structured CSDI residuals are not fully interchangeable with iid noise of matched
magnitude, especially in tail survival probability rather than mean VPT.
Delay-manifold forecasting (DeepEDM in Takens coordinates) supplies a
complementary dynamics-aware route through the same frontier, with explicit scope
boundaries on non-smooth systems such as Chua and scalar delay-differential
systems such as Mackey-Glass.

Patched headline numerical evidence under the Figure-1 v2 protocol:
on **L63 SP65**, CSDI-filled Panda raises mean VPT from 1.22 to 2.86 Lyapunov
times (paired-bootstrap CI [+1.40, +1.87]), with `Pr(VPT > 1.0)` rising from
70 % to 100 %. On **L63 SP82**, it raises mean VPT from 0.34 to 1.34
(CI [+0.54, +1.51]) and `Pr(VPT > 1.0)` from 0 % to 60 %. In L96 N=20,
Panda's mean is dominated by rare lucky linear seeds, but CSDI improves median
and survival at SP82 over 10 seeds (median 0.50 to 1.05, `Pr(VPT>0.5)` 60 %
to 100 %), and DeepEDM has a cleaner paired gain at SP82 (+0.43, CI
[+0.29, +0.57]). On the
pure-noise axis at every tested sigma, CSDI is neutral or slightly hurtful for
Panda — confirming the gap-imputation framing.

---

## 1 Introduction

Pretrained time-series forecasters are usually evaluated under a generous
condition: a dense, clean context window is handed to the model, and the model
is asked to continue the trajectory. Chaotic sensing systems rarely look like
that. Weather stations drop readings, physical sensors saturate, laboratory
measurements jitter, and deployed forecasting systems usually fill missing
values before the forecasting model ever sees the context. In this paper we
study what happens when pretrained chaotic forecasters are placed behind that
realistic sparse-observation interface, with dense noise mapped as a separate
stress axis rather than folded into the headline claim.

Our main empirical finding is that sparse observations create **sharp
forecastability frontiers**. As sparsity increases, models such as Panda-72M,
Chronos, and context-parroting baselines do not simply lose valid prediction
time smoothly. Instead, their survival probability
$\Pr(\mathrm{VPT} > \theta)$ drops abruptly in a transition band. This matters
because mean VPT is highly seed-sensitive near the frontier: one long forecast
can hide four failed ones. We therefore use survival probabilities, paired
bootstrap contrasts, and Lyapunov-normalized VPT as the primary lens.

The first explanation to test is that standard linear interpolation creates
filled contexts that are out of distribution for a tokenized chaotic forecaster,
and that a learned imputer such as CSDI rescues the model by moving the context
back toward the clean trajectory. Protocol-aligned diagnostics support this
mechanism in the entrance band of the L63 frontier: at SP65, CSDI is closer to
clean than linear interpolation in local raw-patch statistics and in Panda's
patch, embedder, encoder, and pooled-latent representations. At SP82, the
Panda-space distances still favor CSDI and most raw metrics do as well, but one
raw temporal statistic becomes mixed; the mechanism is therefore strong but not
reducible to a single scalar fidelity metric.

The paper is therefore not a story about a new mask trick, nor a claim that
ambient foundation models are intrinsically doomed. It is a regime-aware failure
law: under sparse observations, preprocessing choices can move a pretrained
forecaster between survival and collapse. In the entrance band, the mechanism
looks like raw/token OOD mitigation. Near the frontier floor, distance-to-clean
metrics still matter, but tail survival remains more seed-sensitive and
cannot be reduced to one raw-patch statistic.

The regime qualification is essential. We observe three relevant regimes inside
the sparse-observation frontier:

1. **Entrance-band CSDI regime.** On L63 SP65 under the Figure-1 protocol,
   `CSDI -> Panda` reaches mean VPT 2.86 versus 1.22 for `linear -> Panda`,
   with paired-bootstrap gain +1.64 Lyapunov times and CI [+1.40, +1.87].
   Raw-patch and Panda-token distances both move toward clean.
2. **High-dimensional high-variance regime.** On L96 N=20 SP65, mean VPT is
   dominated by rare long-survival Panda seeds and generic residual controls
   are not cleanly separated. CSDI still gives the best median and strongest
   tail survival (`Pr(VPT>1.0)=80 %` vs 40 % for linear / iid / shuffled).
3. **Floor-band CSDI regime.** Deeper in the transition band, generic noise is
   not enough to match CSDI. On L63 SP82, CSDI improves Panda by +1.09
   Lyapunov times with 95 % paired CI [+0.65, +1.61] and raises
   `Pr(VPT>1.0)` from 0 % to 70 %. Shuffled CSDI residuals help but remain
   smaller (+0.34), while iid jitter does not cross zero. On L96 N=20 SP82,
   Panda mean remains high-variance, but median and survival improve under
   CSDI and DeepEDM has a strict-positive CSDI gain.

A fourth regime we deliberately surface is **pure-noise non-rescue**: at $s=0$
with $\sigma>0$, CSDI does not improve Panda and at higher $\sigma$ slightly
hurts. CSDI is therefore a sparse-gap imputation lever, not a generic dense-noise
denoiser.

Delay-manifold forecasting provides a complementary route through the same
frontier. DeepEDM in Takens coordinates is not the only survivor, and in several
absolute-VPT tables `CSDI -> Panda` is stronger than `CSDI -> DeepEDM`. But the
imputer-by-forecaster isolation matrix shows that delay-coordinate forecasting
also benefits from corruption-aware reconstruction and provides a
dynamics-structured companion. This framing lets us report negative boundary
cases honestly: Mackey-Glass and Chua are not hidden failures, but scope
conditions for the present smooth-attractor / delay-manifold assumptions.

### Contributions

**Failure law.** We map sparse-observation forecastability frontiers and the
separate dense-noise stress line for pretrained chaotic forecasters using
Lyapunov-normalized VPT and survival probability, rather than mean-only
degradation curves.

**Intervention law.** We isolate imputers from forecasters through a
$\{$linear, Kalman, CSDI$\} \times \{$Panda, DeepEDM$\}$ matrix and show that
corruption-aware reconstruction reliably improves survival inside the transition
band.

**Regime-aware mechanism.** We show that entrance-band rescue coincides with
reduced raw-patch and Panda-token OOD, while floor-band survival is not
explained by distance-to-clean alone.

**Regime taxonomy.** Jitter and shuffled-residual controls separate
entrance-band CSDI rescue, high-dimensional generic regularization, floor-band
CSDI survival, and pure-noise non-rescue, preventing an over-broad claim that
CSDI is universally beneficial.

**Dynamics-structured companion.** We retain delay-manifold forecasting as a
second, dynamics-aware route across the frontier and state its scope conditions
explicitly.

---

## 2 Related Work

**Pretrained chaotic / time-series forecasters.** Chronos [Ansari24],
TimesFM [Das23], Lag-Llama [Rasul23], TimeGPT [Garza23], and the chaos-specific
Panda-72M [Wang25] pretrain decoder Transformers on large time-series corpora
and are evaluated primarily on dense, clean context windows. Our work asks what
happens at the realistic interface where the context is sparse and noisy and is
filled before forecasting. We pick Panda-72M as the headline because it is
already specialized for chaotic dynamics and therefore the strongest available
candidate for "the forecaster does not need a corruption-aware front-end".

**Time-series imputation under missingness.** BRITS [Cao18] frames imputation
as a bidirectional recurrent process; SAITS [Du22] uses diagonally-masked
self-attention; CSDI [Tashiro21] introduces score-based diffusion imputation
conditioned on observed points; recent work such as Glocal-IB argues that
high-missingness imputation must preserve a global latent structure and not
just minimize pointwise reconstruction error. We extend this conversation
downstream: in our chaotic-forecasting setting, even imputations that are
*closer* to the clean trajectory in raw or foundation-model token space do not
always forecast better, and structured residuals are not interchangeable with
iid noise of matched magnitude. That is, the relevant objective is not
imputation fidelity but whether the reconstructed context lies on the
forecastable side of a sharp forecastability frontier.

**Delay-coordinate forecasting.** Classical Takens-style delay-embedding plus
local linear or kernel prediction goes back to [Farmer-Sidorowich87, Casdagli89].
Echo-state networks [Jaeger01, Pathak18], reservoir computing, and
operator-theoretic approaches [Brunton16, Lu21] also provide forecasters whose
state is constructed from delay or projected coordinates. DeepEDM /
LETS-Forecast [Majeedi25] recasts delay-coordinate prediction as
softmax-attention-as-learned-kernel on Takens tokens; we use it as a
complementary dynamics-structured route that does not depend on a foundation
model's tokenizer.

**Phase transitions and survival probabilities in forecasting.** Although the
words "phase transition" appear informally in chaos literature, our operational
claim — that survival probability $\Pr(\mathrm{VPT}>\theta\Lambda)$ collapses
non-smoothly across a narrow band of sparsity — does not assume a thermodynamic
critical exponent. We follow numerical-weather-forecasting practice [Bauer15]
in treating tail-survival as the operational quantity, and use the language
*sharp forecastability frontier* rather than *phase transition* to keep the
empirical claim distinct from a physics analogy.

**Data assimilation as classical sparse-observation baseline.** EnKF / LETKF [Hunt07]
is the standard baseline for partially observed chaotic state estimation but is typically deployed
jointly with the model dynamics, not as a preprocessing step feeding a
black-box forecaster. We do not include it as a primary baseline because the
comparison would conflate state-estimation quality with the forecastability
question we isolate; we discuss the relationship in §6.

---

## 3 Sharp Forecastability Frontiers

### 3.1 Observation Model and Order Parameters

Let $x_t$ be a clean trajectory sampled from a chaotic attractor. We observe
$y_t = x_t + \eta_t$ with $\eta_t \sim \mathcal{N}(0, \sigma^2 I)$, and then
remove a fraction $s$ of time steps according to a missingness mask. The
forecaster receives only the corrupted context. For models that cannot consume
missing values natively, the context is filled before forecasting.

We measure forecastability using valid prediction time (VPT) in Lyapunov units.
For seed-stable comparison near the frontier, the primary order parameters are

- $\Pr(\mathrm{VPT} > 0.5\,\Lambda)$,
- $\Pr(\mathrm{VPT} > 1.0\,\Lambda)$,
- paired-bootstrap differences in VPT between matched imputers or forecasters.

Mean VPT remains useful, but it is not the only headline metric: in transition
bands it is dominated by rare long-survival seeds. We report both, and use
Wilson 95 % intervals on survival probability and bootstrap 95 % intervals on
mean differences.

### 3.2 Cross-System Frontier Evidence

The L63 v2 fine grid (Figure 1) shows the cleanest sparsity transition: between
SP65 ($s=0.65$) and SP82 ($s=0.82$, both $\sigma=0$), `linear -> Panda` mean VPT
falls from 1.22 to 0.33 and $\Pr(\mathrm{VPT}>1.0)$ drops from 70 % to 0 %,
while `CSDI -> Panda` retains 2.86 / 100 % at SP65 and degrades much more
smoothly to 1.34 / 60 % at SP82.

Cross-system replication on L96 N=20 (10 seeds, same v2 protocol) confirms the
same direction at high dimensionality, but also shows why we report median and
survival instead of mean alone: Panda has rare long linear-filled forecasts that
dominate the mean. At L96 SP82, CSDI improves Panda's median VPT
(0.50 to 1.05) and `Pr(VPT>0.5)` (60 % to 100 %), while mean VPT is not a
stable summary. `CSDI -> DeepEDM` gives the cleaner paired evidence:
strict-positive CSDI gains across SP55–SP82, with +0.43 and CI [+0.29,+0.57]
at SP82, supporting the delay-manifold companion claim. Rössler shows lower
absolute VPT because of its small Lyapunov exponent and finite prediction
window, but still exhibits the same transition-band sensitivity. Mackey-Glass
and Chua are kept as boundary cases (§6.3).

This is why we use the term "sharp forecastability frontier" rather than a
purely theoretical "phase transition" without qualification. The operational
frontier is the region in $(s, \sigma)$ where survival probabilities change
rapidly under small changes in corruption or preprocessing — and only on the
sparsity axis; the noise axis degrades monotonely and CSDI does not rescue
there (§3.4).

### 3.3 Pure-Sparsity Transition Band

The v2 corruption grid separates sparsity from observation noise. Pure-sparsity
settings ($\sigma=0$) are especially important because they remove the easy
explanation that the learned imputer simply denoises. We use SP65 and SP82 as
two representative points: SP65 sits at the entrance to the frontier, while
SP82 is deeper in the transition band and is the cleanest L63 CSDI regime. The
L96 N=20 v2 replication confirms that the band transfers to high-dimensional
chaos, but it must be read through survival and median metrics because its mean
VPT is highly seed-sensitive.

### 3.4 What the Frontier Is Not

The frontier is not a claim that corruption-aware reconstruction always helps
under every corruption. The pure-noise line is the counterexample we keep in
the main text: when the context is dense but noisy, CSDI does not rescue Panda
and can slightly hurt. This clarifies that CSDI is acting as a gap-imputation
lever, not as a generic dense-noise denoiser.

The frontier is also not a claim that CSDI is the only way to improve mean VPT.
L96 N=20 SP65 shows a generic-regularization regime where iid jitter and
shuffled residuals recover much of the mean gain. The important distinction is
tail survival: CSDI is still strongest on median VPT and $\Pr(\mathrm{VPT}>1.0)$.

Thus the intervention claim is deliberately conditional:

> Inside the transition band, corruption-aware reconstruction is the only
> tested intervention that reliably moves models back across the
> forecastability frontier; structured CSDI residuals are not fully
> interchangeable with iid noise of matched magnitude, especially in tail
> survival probability rather than mean VPT.

---

## 4 Mechanism and Intervention Isolation

### 4.1 Imputer-by-Forecaster Matrix

The main isolation experiment crosses three imputers with two forecasters:

- imputers: linear interpolation, AR-Kalman, CSDI;
- forecasters: Panda-72M in ambient coordinates, DeepEDM in Takens / delay
  coordinates.

This matrix answers a reviewer-critical question: is Panda failing because
ambient foundation models are intrinsically bad at chaos, or because the
corrupted context presented to Panda is a bad forecasting object?

The answer is the second one, but with a twist. CSDI often rescues Panda. Under
the original 5-seed S0–S6 isolation matrix, on L96 N=20 S4, `CSDI -> Panda`
raises mean VPT from 0.52 to 3.60 and $\Pr(\mathrm{VPT}>0.5)$ from 60 % to
100 %, with paired-bootstrap gain +3.07 Lyapunov times and CI [+0.57, +6.45].
On L96 N=10 S4, the gain is +1.11 with CI [+0.08, +2.22]. On L63 S2, the gain
is +0.82 with CI [+0.32, +1.37]. Rössler is lower in absolute VPT but keeps
positive CSDI directions, especially for DeepEDM.

The same matrix also keeps the role of DeepEDM honest. Delay-manifold
forecasting is complementary, not uniquely dominant. CSDI improves DeepEDM in
several transition-band cells, but `CSDI -> Panda` is often the best absolute
cell. This prevents the paper from becoming a fragile "delay coordinates are
the only survivor" claim. In the L96 N=20 v2 cross-system replication, the
DeepEDM CSDI-vs-linear paired CI is strict-positive across SP55–SP82, which is
*cleaner* than the Panda CSDI-vs-linear CI at the same cells; this supports
keeping delay-manifold forecasting as a real companion rather than appendix
material.

### 4.2 Regime-Aware Mechanism: OOD in the Entrance Band, Mixed at the Floor

We compare clean, linear-fill, and CSDI-fill contexts under the same protocol
as Figure 1. At L63 SP65, CSDI is closer to clean than linear interpolation in
all three raw-patch v2 metrics:

| Metric | Linear/CSDI W₁-to-clean ratio |
|:--|--:|
| local stdev | 21.02 |
| lag-1 autocorrelation | 15.02 |
| mid-frequency power | 33.71 |

The same pattern appears inside Panda. At SP65, the linear/CSDI paired-distance
ratios to clean are 16.77 at the patch stage, 12.84 after the DynamicsEmbedder,
14.02 after the encoder, and 21.85 in the pooled latent. This supports a
straightforward entrance-band mechanism: corruption-aware imputation reduces
raw / token OOD and restores forecastability.

At L63 SP82, Panda-space distances still favor CSDI, but less dramatically
(linear/CSDI ratios 1.63–2.43). Raw metrics are partly mixed: local stdev and
mid-frequency power favor CSDI (3.54× and 5.19×), while lag-1 autocorrelation
favors linear (0.62×). CSDI still improves survival, but the floor-band
mechanism is no longer captured by a single scalar fidelity metric. This is
the mechanism boundary we keep in the main text — and the reason the paper's
mechanism claim is regime-aware rather than a single tokenizer-OOD line.

### 4.3 Jitter Controls and the Three Regimes

To test whether CSDI is merely stochastic regularization, we run a four-cell
Panda-only control on six (system, scenario) settings:

- `linear`,
- `linear + iid jitter` matched to the per-channel CSDI residual scale,
- `linear + shuffled CSDI residual` applied at missing entries,
- `CSDI`.

All controls use the same missing masks and the same forecast model.

The result separates three regimes inside the sparse-observation frontier.

**Entrance-band CSDI regime** (L63 SP65 under v2 protocol). Mean VPT is 1.22
for linear and 2.87 for CSDI, with paired gain +1.65 and CI [+1.41, +1.87].
Iid jitter (Δ +0.17, CI [−0.01, +0.36]) and shuffled residuals (Δ −0.16,
CI [−0.34, −0.02]) do not reproduce the gain.

**High-dimensional high-variance regime** (L96 N=20 SP65). Iid jitter,
shuffled residuals, and CSDI all move in the positive direction, but none is
cleanly separated in mean because Panda has rare long-survival seeds. CSDI
remains best in median and tail survival: $\Pr(\mathrm{VPT}>1.0)$ is 80 % for
CSDI and 40 % for linear / iid / shuffled.

**Floor-band CSDI regime** (L63 SP82, L96 N=20 SP82, Rössler SP65 / SP82). At
SP82, generic noise does not match CSDI. On L63 SP82, iid jitter does not cross
zero, shuffled residuals help modestly (+0.34), and CSDI is the strongest
intervention: +1.09 with CI [+0.65, +1.61] and `Pr(VPT>1.0)` 70 %. On L96
N=20 SP82, Panda's mean is dominated by a lucky linear seed, but CSDI improves
median and survival; DeepEDM gives the clean paired result (+0.43, CI
[+0.29,+0.57]). Rössler
shows the same positive CSDI direction across SP65 / SP82, but its small
Lyapunov exponent makes $\Pr(\mathrm{VPT}>1.0)$ too strict; $\Pr(\mathrm{VPT}>0.5)$
is the more appropriate tail metric there.

These controls are the reason our abstract must say "inside the transition
band". CSDI is not universally better than linear, and generic jitter explains
part of one L96 frontier cell. But across the deeper transition band, CSDI is
the only tested intervention that reliably moves models back across the
forecastability frontier; **structured CSDI residuals are not fully
interchangeable with iid noise of matched magnitude, especially in tail
survival probability rather than mean VPT**. The mean / tail asymmetry is
itself a finding: in the generic-regularization regime any plausible variability
recovers most of the mean gain, but only the structured residual recovers the
fraction of seeds that survive past one Lyapunov decorrelation time.

### 4.4 Alt-Imputer Reviewer Defense

A natural reviewer question is whether structured imputation per se is the
lever, or whether CSDI's specific dynamics-aware diffusion residuals are
required. As an appendix sanity check we ran SAITS and BRITS in a per-instance
fitting regime (single 512-step trajectory, no pretraining corpus) at L63 SP65;
both collapsed to mean VPT 0.15 / 0.16, well below `linear -> Panda` (1.29) and
far below `CSDI -> Panda` (2.90, ceiling). This is **biased against
SAITS / BRITS** because they are normally pretrained on a large corpus, which
matches CSDI's training pipeline. The appendix-only conclusion is therefore
narrow: per-instance training of generic structured imputers under high
missingness is not a viable substitute. A pretrained alt-imputer comparison
(SAITS / Glocal-IB on the same chaos corpus that CSDI used) is the natural
follow-up; we describe its design and decision rule in §6.4 and Appendix C.

### 4.5 Interpretation

The mechanism we can support is:

1. Sparse sensing creates a forecastability frontier on the sparsity
   axis.
2. In the entrance band, CSDI moves contexts closer to the clean trajectory in
   raw-patch statistics and Panda-token geometry, and this is where the largest
   forecastability rescue occurs.
3. Near the frontier floor, Panda-token distances still favor CSDI but raw
   patch statistics become partially mixed; CSDI can retain a survival
   advantage even when no single raw fidelity metric is sufficient.
4. The residual structure matters: iid noise of matched magnitude and shuffled
   residuals do not reproduce the CSDI-unique SP82 gains.

We therefore avoid the overclaim that we have fully characterized Panda's
internal failure channel. What is settled is the empirical law: the sparsity
frontier is real, CSDI crosses it in the transition band, and the mechanism is
regime-dependent rather than a universal one-line tokenizer story.

### 4.6 Scope Conditions

The delay-manifold companion assumes a smooth attractor and a useful
finite-dimensional Takens representation. Mackey-Glass and Chua violate this
comfort zone in different ways: Mackey-Glass is a scalar delay-differential
system whose effective state is infinite-dimensional under the observation
window, and Chua has piecewise-linear / non-smooth circuit dynamics. We
report these systems as scope boundaries (§6.3). They are not used to inflate
the main claim, and they prevent the method section from sounding like a
universal chaos solver.

---

## 5 Method

We deliberately keep this section short. The main contributions of the paper
are the failure law, the intervention law, and the regime-aware mechanism, not
a new modular pipeline.

### 5.1 Corruption-aware imputation (M1)

We use a CSDI-style score-based diffusion imputer [Tashiro21] to fill the
corrupted sparse context. A clean attractor scale $\sigma_\mathrm{attr}$ (per-axis
mean over a long reference trajectory) is used to normalize inputs and the
diffusion noise schedule; mismatched normalization leads to under- or
over-noised samples and was the origin of an earlier protocol inconsistency
that we report transparently in the reproducibility appendix. Inference uses
$\sigma_\mathrm{override}$ matched to the actual scenario noise level
($\sigma \cdot \sigma_\mathrm{attr}$ for $s>0$, exactly $0$ for pure-sparsity
cells); under-the-correct-protocol L63 imputations achieve max observed-anchor
error on the order of $7 \cdot 10^{-6}$, confirming CSDI does not clobber
observed timestamps.

Only one imputation is required per context; we use the diffusion median across
a small sample budget for the deterministic VPT panels and the full sample
distribution for tail survival.

### 5.2 Delay-manifold forecaster (DeepEDM) as companion

Our companion forecaster predicts the next state from a fixed-length delay
vector $X_t = [x_t, x_{t-\tau_1}, \ldots, x_{t-\tau_L}]$ using a softmax-attention
learned-kernel head [Majeedi25] trained on delay / next-state pairs derived
from the imputed context. Lags $\{\tau_i\}$ are selected by a mutual-information
/ Lyapunov objective (Appendix A) to balance injectivity and stretch rate.
DeepEDM is included primarily because it does not depend on a foundation-model
tokenizer; the §4 isolation matrix shows that this route can also be improved
by corruption-aware imputation.

### 5.3 Forecasters under test

The forecaster under test in the main figures is Panda-72M [Wang25]. The
isolation matrix in §4 spans $\{$linear, AR-Kalman, CSDI$\}$ imputers times
$\{$Panda, DeepEDM$\}$ forecasters. The alt-imputer reviewer-defense table in
§4.4 adds SAITS and BRITS at the strongest CSDI-decisive cells; neither is
pretrained on the chaos corpus, so the comparison is biased against them — we
keep this caveat in §6.4.

### 5.4 Operational metrics

We report three metrics throughout the main paper:

- mean valid-prediction time (VPT) in Lyapunov-time units;
- survival probability $\Pr(\mathrm{VPT} > 0.5\,\Lambda)$ and $\Pr(\mathrm{VPT}
  > 1.0\,\Lambda)$, the latter being the operational tail metric;
- paired-bootstrap mean differences between matched cells of the isolation
  matrix.

CIs are 95 % bootstrap on the mean and Wilson 95 % on survival probabilities.

---

## 6 Discussion and Limitations

### 6.1 What the paper claims, restated

Inside the sparse-observation transition band, corruption-aware imputation
reliably moves Panda back across the forecastability frontier. In the entrance
band we can attribute this to a large reduction in raw-patch and Panda-token
distance to the clean context; near the frontier floor Panda-token distances
still favor CSDI, but raw temporal metrics become partially mixed and survival
is more seed-sensitive. CSDI is therefore a sparse-gap imputation lever, not a
generic dense-noise
denoiser. Delay-manifold forecasting is a complementary dynamics-aware route
through the same frontier.

### 6.2 What the paper does not claim

We do not claim that pretrained chaotic forecasters are intrinsically broken;
the rescue results show they are highly recoverable when the filled context is
corruption-aware. We do not claim CSDI is the only imputer that works — only
the only *tested* intervention that reliably rescues across the band; the
alt-imputer experiment in §4.4 sets the current boundary, and a pretrained
SAITS / Glocal-IB comparison is deferred to Appendix C. We do not claim that
the mechanism is fully characterized; distance-to-clean explains the entrance
band but not the floor band. We do not claim universality across all foundation
forecasters; Panda-72M is the headline and Chronos / context-parroting are
present only as side baselines.

### 6.3 Scope conditions

The delay-manifold companion assumes a smooth attractor and a useful
finite-dimensional Takens representation. Mackey-Glass and Chua are reported
as appendix scope boundaries. Mackey-Glass is a scalar delay-differential
system whose effective state is infinite-dimensional under the observation
window; the available CSDI training corpus and delay configuration do not span
the relevant history dimension. Chua is a piecewise-linear, non-smooth circuit;
the smooth-attractor assumptions implicit in M1 / DeepEDM are violated. These
are honest boundaries, not hidden failures.

### 6.4 Limitations

- **Corpus asymmetry.** Our CSDI is pretrained on each system separately
  ($\sim$500K independent-IC windows per system). SAITS and BRITS in our
  alt-imputer experiment are trained per-instance from a single 512-step
  trajectory. A pretrained SAITS / BRITS / Glocal-IB comparison on the same
  chaos corpus is the natural follow-up.
- **Single-forecaster headline.** Panda is the only foundation forecaster
  fully evaluated under the v2 corruption grid. Replicating the frontier
  curve with TimesFM / Chronos / Lag-Llama would extend the external validity.
- **Pure-noise axis.** The paper's intervention claim is restricted to the
  sparse-observation axis. CSDI is neutral or slightly hurtful on the
  dense-noise axis; a denoising-aware variant is an open follow-up.
- **System breadth.** L63, L96 N=10/20, Rössler, and Kuramoto cover the
  positive replication; Mackey-Glass and Chua are scope boundaries. KSE /
  dysts breadth and real-data case studies (EEG, climate reanalysis) remain
  future work.
- **Foundation-model interpretability.** Why CSDI's residual produces a
  forecastable context near the floor band, where Panda-token distances favor
  CSDI but raw temporal metrics are not uniformly aligned, is still open. A
  natural hypothesis is that the relevant geometric quantity lives in Panda's
  deeper latent dynamics (decoder side rather than encoder side), which we
  have not instrumented.

### 6.5 Relationship to data assimilation

Sequential data assimilation (EnKF / LETKF) is a richer and more
information-efficient approach to sparse-observation chaos when the dynamics are
known and tracked online. Our setting is different: the forecaster is a black
box (Panda) and the corruption is preprocessed offline. We therefore compare
against preprocessing-style baselines (linear / Kalman / CSDI) that match the
deployment interface, and we read the corresponding DA literature as motivating
the existence of the forecastability frontier rather than as a direct
competitor.

---

## 7 Conclusion

Pretrained chaotic forecasters fail across sharp sparse-observation
forecastability frontiers. Inside the transition band, the rescue is regime-aware,
not universal: in the entrance band, corruption-aware imputation moves Panda
contexts toward clean raw patches and tokens and forecastability returns; near
the frontier floor, the same distances are still useful but no single raw
fidelity metric explains all tail behavior, and structured CSDI residuals are
not interchangeable with iid noise of matched magnitude. CSDI is
the gap-imputation lever — not a denoiser — and DeepEDM in Takens coordinates
is the dynamics-aware companion route, with Mackey-Glass and Chua reported as
explicit scope boundaries.

The cleanest one-line takeaway is therefore not "foundation models are
intrinsically OOD" and not "structured imputers always rescue", but rather:
**sparse-observation preprocessing places a pretrained chaotic forecaster on
one side of a sharp forecastability frontier, and corruption-aware imputation
is the lever that crosses it inside the transition band — but the geometry
that controls this crossing is not pointwise reconstruction fidelity**.

Code, CSDI checkpoints, and the locked Figure-1 / isolation / jitter / embedding
data are released. A pretrained SAITS / Glocal-IB reviewer-defense comparison
(Appendix C) is the natural follow-up before camera-ready.

---

> **Appendix material below is inherited from the previous draft and has not yet
> been re-aligned to the locked story.** Specifically: Appendix A's theorem
> proofs reference the older "tokenizer-OOD as universal mechanism" framing and
> need to be either narrowed to the entrance-band claim or moved to a separate
> theory note. Appendix D's figure index needs to be re-keyed against
> `deliverable/figures_main/` and `deliverable/figures_jitter/`. Appendices E–G
> describe the older 4-module pipeline at a level of detail no longer matched
> by §5 of the main text.
>
> Appendices retained verbatim from the archive for now; review pass is the
> next pre-submission task.

## Appendix A: Proof sketches

*(See `paper_draft_en_archive_2026-04-30.md` §"Appendix A: Proof sketches" for
the inherited proof material. Theorem 2's "tokenizer-OOD" bound is now a
mechanism only inside the entrance band, per §4.2; the main-text claim does
not depend on the proof.)*

## Appendix B: Reproducibility

Locked v2 protocol:
- L63: `LORENZ63_ATTRACTOR_STD = 8.51`, `dt = 0.025`, `n_ctx = 512`,
  corruption seed `1000 * seed + 5000 + grid_index` from
  `experiments/week1/configs/corruption_grid_v2.json`.
- L96 N=20: `lorenz96_attractor_std(N=20, F=8) = 3.6387`, `dt = 0.05`,
  `n_ctx = 512`, same seed scheme.
- CSDI inference: `set_csdi_attractor_std()` matches the system, and
  `sigma_override = sigma * attractor_std` is passed to `impute(..., kind="csdi")`
  (`sigma_override = 0` for pure-sparsity SP cells).

Result JSONs:
- Figure 1 (L63 v2 fine_s_line + fine_sigma_line, 10 seeds, 4 cells):
  `experiments/week1/results/pt_l63_grid_v2_l63_fine_{s,sigma}_v2_10seed_{h0,h5}.json`
  — aggregated by `experiments/week1/aggregate_figure1_v2.py` to
  `deliverable/figures_main/figure1_l63_v2_10seed.{png,md}`.
- Cross-system isolation (5 seeds, 6 cells, S0–S6):
  `experiments/week1/results/pt_{l63,l96_iso_l96N{10,20},rossler_iso_rossler}_5seed.json`
  — aggregated to `deliverable/figures_isolation/`.
- L96 N=20 v2 cross-system (10 seeds, 4 cells, SP55–SP82 + NO010–NO050):
  `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_5seed.json`
  and `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json`
  — see `deliverable/L96_V2_B_PATCHED_N10.md`.
- Jitter controls (v2 protocol, 5 / 10 seeds):
  `experiments/week1/results/panda_jitter_control_l{63,96N20,rossler}_*5seed*.json`
  — aggregated to `deliverable/figures_jitter/jitter_milestone_summary.md`.
- Panda embedding diagnostic (v2 protocol):
  `experiments/week1/results/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_5seed.json`.
- Raw-patch diagnostic (v2 protocol):
  `experiments/week1/results/l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json`.
- Alt-imputer C0 sanity:
  `experiments/week1/results/panda_altimputer_l63sp65_partial_5seed.json`.

## Appendix C: Pretrained Alt-Imputer Plan (deferred)

See `deliverable/EXPERIMENT_C_PLAN.md`. The reviewer-defense run trains
SAITS and (if implementable) Glocal-IB on the same chaos corpus that CSDI
used, then evaluates at L63 SP82 + L96 N=20 SP82. The decision rule:

- alt-imputer recovers comparable rescue → narrow main claim to "structured
  / global-aware imputation is the lever; CSDI is one strong instance";
- alt-imputer fails → strengthen claim that dynamics-aware reconstruction is
  required.

Both outcomes preserve the §3 frontier and §6 scope conditions.

## Appendix D: Figure Index

*(To be re-keyed against the new figure layout; see archive for the
inherited index.)*

Main figures:
- **Figure 1** — L63 v2 10-seed fine grid:
  `deliverable/figures_main/figure1_l63_v2_10seed.png`.
- **Figure 2** — Cross-system isolation (4-system 5-seed heatmaps):
  `deliverable/figures_isolation/*_heatmap.png`.
- **Figure 3** — Jitter milestone (3-system tail vs mean):
  `deliverable/figures_jitter/jitter_milestone_SP82.png`.

Mechanism panels (§4.2):
- L63 raw-patch diagnostics (curvature, lag-1, mid-freq):
  `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP{65,82}.png`.
- Panda representation distance bars:
  `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_5seed_bars.png`.

## Appendix E: τ-search detailed evidence

*(Inherited from archive; supports DeepEDM lag selection in §5.2.)*

## Appendix F: τ-coupling complete empirical analysis

*(Inherited from archive; not load-bearing for the locked-story main claim.)*

## Appendix G: Delay Manifold Perspective

*(Inherited from archive; provides the geometric framing that motivated the
delay-manifold companion in §5.2.)*
