# Story-Locked Draft Sections: §1, §3, §4

> Drop-in draft for the next paper rewrite. This supersedes the older
> "linear-fill tokenizer OOD" framing. The method/theory sections can be
> re-numbered after these sections are inserted.
>
> **Patched-refresh warning (2026-05-01).** This drop-in file preserves the
> April 30 prose skeleton, but CSDI-dependent numbers and the floor-band
> mechanism wording have been superseded by
> `deliverable/FIGURE1_PATCHED_REFRESH.md` and the current
> `deliverable/paper/paper_draft_en.md`. Use `paper_draft_en.md` as the
> authoritative text.

> **Locked abstract sentence (do not paraphrase without re-checking
> `STORY_LOCK_2026-04-28.md`).** Pretrained chaotic forecasters fail across
> sharp sparse-observation forecastability frontiers. Inside the sparsity
> transition band, corruption-aware imputation is the only tested intervention
> that reliably moves Panda back across the frontier. In the entrance band,
> this rescue is accompanied by a large reduction in both raw-patch and
> Panda-token distance to the clean context; near the frontier floor, those
> distances become mixed, so distance-to-clean alone no longer explains the
> residual survival gain. CSDI is therefore a gap-imputation lever, not a
> generic dense-noise denoiser, and structured residuals are not fully
> interchangeable with iid noise of matched magnitude. Delay-manifold
> forecasting provides a complementary dynamics-aware route, with explicit
> scope boundaries on non-smooth systems such as Chua and scalar
> delay-differential systems such as Mackey-Glass.
>
> Three qualifiers must remain in any §1 / abstract paraphrase: (i)
> *sparse-observation* frontier; (ii) *inside the transition band*; (iii)
> *not fully interchangeable*.

---

## 1 Introduction

Pretrained time-series forecasters are usually evaluated under a generous
condition: a dense, clean context window is handed to the model, and the model
is asked to continue the trajectory. Chaotic sensing systems rarely look like
that. Weather stations drop readings, physical sensors saturate, laboratory
measurements jitter, and deployed forecasting systems usually fill missing
values before the forecasting model ever sees the context. In this paper we
study what happens when pretrained chaotic forecasters are placed behind that
realistic sparse/noisy observation interface.

Our main empirical finding is that sparse/noisy sensing creates **sharp
forecastability frontiers**. As sparsity and observation noise increase, models
such as Panda-72M, Chronos, and context-parroting baselines do not simply lose
valid prediction time smoothly. Instead, their survival probability
`Pr(VPT > threshold)` drops abruptly in a transition band. This matters because
mean VPT is highly seed-sensitive near the frontier: one long forecast can hide
four failed ones. We therefore use survival probabilities, paired bootstrap
contrasts, and Lyapunov-normalized VPT as the primary lens.

The first explanation to test is that standard linear interpolation creates
filled contexts that are out of distribution for a tokenized chaotic forecaster,
and that a learned imputer such as CSDI rescues the model by moving the context
back toward the clean trajectory. Protocol-aligned diagnostics support this
mechanism in the entrance band of the L63 frontier: at SP65, CSDI is closer to
clean than linear interpolation in local raw-patch statistics and in Panda's
patch, embedder, encoder, and pooled-latent representations. At SP82, however,
the same distances are mixed or nearly tied while CSDI retains a smaller
survival advantage.

The paper is therefore not a story about a new mask trick, nor a claim that
ambient foundation models are intrinsically doomed. It is a regime-aware failure
law: under sparse observations, preprocessing choices can move a pretrained
forecaster between survival and collapse. In the entrance band, the mechanism
looks like raw/token OOD mitigation. Near the frontier floor, distance-to-clean
metrics are no longer sufficient to explain the residual forecastability.

The regime qualification is essential. We observe three relevant regimes:

1. **Entrance-band CSDI regime.** On L63 SP65 under the Figure-1 protocol,
   `CSDI -> Panda` reaches mean VPT 2.87 versus 1.22 for `linear -> Panda`,
   with paired-bootstrap gain +1.65 Lyapunov times and CI [+1.39,+1.91].
   Raw-patch and Panda-token distances both move toward clean.
2. **Generic-regularization regime.** At some high-dimensional frontier cells,
   adding variability to filled values improves the mean. On L96 N=20 SP65,
   iid jitter, shuffled CSDI residuals, and CSDI all improve mean VPT over
   linear, but CSDI gives the best median and the strongest tail survival
   (`Pr(VPT>1.0)=60%` vs `40%` for jitter/shuffled and `20%` for linear).
3. **Floor-band CSDI regime.** Deeper in the transition band, generic noise is
   not enough and distance-to-clean metrics become less separative. On L63
   SP82, CSDI improves Panda by +0.37 Lyapunov times with 95% paired CI
   [+0.05,+0.67], while iid jitter and shuffled residuals do not cross zero.
   On L96
   N=20 SP82, CSDI improves mean VPT from 0.91 to 3.31 with paired CI
   [+0.10,+6.59], and raises `Pr(VPT>1.0)` from 20% to 60%; iid jitter and
   shuffled residuals do not cross zero.

Delay-manifold forecasting provides a complementary route through the same
frontier. DeepEDM in Takens coordinates is not the only survivor, and in several
absolute-VPT tables `CSDI -> Panda` is stronger than `CSDI -> DeepEDM`. But the
imputer-by-forecaster isolation matrix shows that delay-coordinate forecasting
also benefits from corruption-aware reconstruction and provides a
dynamics-structured companion. This framing lets us report negative boundary
cases honestly: Mackey-Glass and Chua are not hidden failures, but scope
conditions for the present smooth-attractor/delay-manifold assumptions.

### Contributions

**Failure law.** We map sparse/noisy forecastability frontiers for pretrained
chaotic forecasters using Lyapunov-normalized VPT and survival probability,
rather than mean-only degradation curves.

**Intervention law.** We isolate imputers from forecasters through a
`{linear, Kalman, CSDI} x {Panda, DeepEDM}` matrix and show that
corruption-aware reconstruction reliably improves survival inside the
transition band.

**Regime-aware mechanism.** We show that entrance-band rescue coincides with
reduced raw-patch and Panda-token OOD, while floor-band survival is not
explained by distance-to-clean alone.

**Regime taxonomy.** Jitter and shuffled-residual controls separate
entrance-band CSDI rescue, high-dimensional generic regularization, floor-band
CSDI survival, and pure-noise non-rescue, preventing an over-broad claim that
CSDI is universally beneficial.

**Dynamics-structured companion.** We retain delay-manifold forecasting as a
second, dynamics-aware route across the frontier and state its scope
conditions explicitly.

---

## 3 Sharp Forecastability Frontiers

### 3.1 Observation Model and Order Parameters

Let `x_t` be a clean trajectory sampled from a chaotic attractor. We observe

`y_t = x_t + eta_t`, with `eta_t ~ N(0, sigma^2 I)`,

and then remove a fraction `s` of time steps according to a missingness mask.
The forecaster receives only the corrupted context. For models that cannot
consume missing values natively, the context is filled before forecasting.

We measure forecastability using valid prediction time (VPT) in Lyapunov units.
For seed-stable comparison near the frontier, the primary order parameters are

- `Pr(VPT > 0.5)`,
- `Pr(VPT > 1.0)`,
- paired-bootstrap differences in VPT between matched imputers or forecasters.

Mean VPT remains useful, but it is not the only headline metric: in transition
bands it is dominated by rare long-survival seeds.

### 3.2 Coarse Frontier Evidence Across Systems

The original seven-scenario sweep already shows a frontier-like collapse. On
Lorenz63, Panda drops sharply from the clean regime into S3/S4, while the
corruption-aware pipeline degrades more gradually. On L96 N=20 and L96 N=10,
the collapse occurs at high sparsity/noise cells, with VPT often falling to
near zero for linear-filled Panda. Rössler shows lower absolute VPT because of
its small Lyapunov exponent and finite prediction window, but still exhibits
the same transition-band sensitivity. Kuramoto provides another positive
replication; Mackey-Glass and Chua are kept as boundary cases.

This is why we use the term "sharp forecastability frontier" rather than a
purely theoretical "phase transition" without qualification. The operational
frontier is the region in `(s, sigma)` where survival probabilities change
rapidly under small changes in corruption or preprocessing.

### 3.3 Pure-Sparsity Transition Bands

The v2 corruption grid separates sparsity from observation noise. Pure-sparsity
settings are especially important because they remove the easy explanation that
the learned imputer simply denoises. We use SP65 (`s=0.65, sigma=0`) and SP82
(`s=0.82, sigma=0`) as two representative points:

- SP65 often sits near the entrance to the frontier.
- SP82 is deeper in the transition band and is the cleanest CSDI-unique regime
  across L63 and L96 N=20.

The L96 N=20 SP65 smoke confirms that the band transfers to high-dimensional
chaos: `CSDI -> Panda` reaches mean VPT 2.49 versus 1.14 for `linear -> Panda`,
with `Pr(VPT>0.5)` increasing from 60% to 100% and `Pr(VPT>1.0)` from 20% to
60%. The later SP82 jitter milestone gives a sharper result: `CSDI -> Panda`
reaches 3.31 versus 0.91 for linear, paired CI [+0.10,+6.59], and
`Pr(VPT>1.0)` rises from 20% to 60%.

### 3.4 What the Frontier Is Not

The frontier is not a claim that corruption-aware reconstruction always helps
under every corruption. The pure-noise line is the counterexample we keep in
the main text: when the context is dense but noisy, CSDI does not rescue Panda
and can slightly hurt. This clarifies that CSDI is acting as a gap-imputation
lever, not as a generic dense-noise denoiser.

The frontier is also not a claim that CSDI is the only way to improve mean VPT.
L96 N=20 SP65 shows a generic-regularization regime where iid jitter and
shuffled residuals recover much of the mean gain. The important distinction is
tail survival: CSDI is still strongest on median VPT and `Pr(VPT>1.0)`.

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

- imputers: linear interpolation, Kalman/AR-Kalman, CSDI;
- forecasters: Panda-72M in ambient coordinates, DeepEDM in Takens/delay
  coordinates.

This matrix answers a reviewer-critical question: is Panda failing because
ambient foundation models are intrinsically bad at chaos, or because the
corrupted context presented to Panda is a bad forecasting object?

The answer is the second one, but with a twist. CSDI often rescues Panda. On
L96 N=20 S4, `CSDI -> Panda` raises mean VPT from 0.52 to 3.60 and
`Pr(VPT>0.5)` from 60% to 100%, with paired-bootstrap gain +3.07 Lyapunov
times and 95% CI [+0.57,+6.45]. On L96 N=10 S4, the gain is +1.11 with CI
[+0.08,+2.22]. On L63 S2, the gain is +0.82 with CI [+0.32,+1.37]. Rössler is
lower in absolute VPT but keeps positive CSDI directions, especially for
DeepEDM.

The same matrix also keeps the role of DeepEDM honest. Delay-manifold
forecasting is complementary, not uniquely dominant. CSDI improves DeepEDM in
several transition-band cells, but `CSDI -> Panda` is often the best absolute
cell. This prevents the paper from becoming a fragile "delay coordinates are
the only survivor" claim.

### 4.2 Regime-Aware Mechanism: OOD in the Entrance Band, Mixed at the Floor

We compare clean, linear-fill, and CSDI-fill contexts under the same protocol
as Figure 1. At L63 SP65, CSDI is closer to clean than linear interpolation in
all three raw-patch v2 metrics:

| Metric | Linear/CSDI W1-to-clean ratio |
|:--|--:|
| local stdev | 8.87 |
| lag-1 autocorrelation | 2.28 |
| mid-frequency power | 16.16 |

The same pattern appears inside Panda. At SP65, the linear/CSDI
paired-distance ratios to clean are 8.03 at the patch stage, 6.12 after the
DynamicsEmbedder, 6.70 after the encoder, and 8.85 in the pooled latent. This
supports a straightforward entrance-band mechanism: corruption-aware imputation
reduces raw/token OOD and restores forecastability.

At L63 SP82, the picture changes. Panda-space ratios are near one, and raw
metrics are mixed: local stdev and mid-frequency power still slightly favor
CSDI, while lag-1 autocorrelation favors linear. CSDI still improves survival,
but the improvement is smaller and no single distance-to-clean metric explains
it. This is the mechanism boundary we keep in the main text.

### 4.3 Jitter Controls and the Three Regimes

To test whether CSDI is merely stochastic regularization, we run a four-cell
Panda-only control on six `(system, scenario)` settings:

- `linear`,
- `linear + iid jitter` matched to the per-channel CSDI residual scale,
- `linear + shuffled CSDI residual` applied at missing entries,
- `CSDI`.

All controls use the same missing masks and the same forecast model.

The result separates three regimes.

**Entrance-band CSDI regime.** L63 SP65 is no longer a no-intervention example
under the Figure-1 protocol. Mean VPT is 1.22 for linear and 2.87 for CSDI,
with paired gain +1.65 and CI [+1.39,+1.91]. Iid jitter and shuffled residuals
do not reproduce the gain.

**Generic-regularization regime.** L96 N=20 SP65 admits generic improvement:
iid jitter, shuffled residuals, and CSDI all improve mean VPT over linear.
However, CSDI remains best in median and tail survival: `Pr(VPT>1.0)` is 60%
for CSDI, 40% for jitter/shuffled, and 20% for linear.

**Floor-band CSDI regime.** At SP82, generic noise does not transfer. On L63
SP82, iid jitter and shuffled residuals do not cross zero, and CSDI is the only
strictly positive intervention: +0.37 with CI [+0.05,+0.67].
On L96 N=20 SP82, CSDI is the cleanest case: mean VPT 3.31 versus 0.91 for
linear, paired CI [+0.10,+6.59], and `Pr(VPT>1.0)` 60% versus 20% for all
non-CSDI cells. Rössler shows the same positive CSDI direction, but its small
Lyapunov exponent makes `Pr(VPT>1.0)` too strict; `Pr(VPT>0.5)` is the more
appropriate tail metric there.

These controls are the reason our abstract must say "inside the transition
band." CSDI is not universally better than linear, and generic jitter explains
part of one L96 frontier cell. But across the deeper transition band, CSDI is
the only tested intervention that reliably moves models back across the
forecastability frontier; **structured CSDI residuals are not fully
interchangeable with iid noise of matched magnitude, especially in tail
survival probability rather than mean VPT**. The mean / tail asymmetry is
itself a finding: in the generic-regularization regime any plausible variability
recovers most of the mean gain, but only the structured residual recovers the
fraction of seeds that survive past one Lyapunov decorrelation time.

### 4.4 Interpretation

The mechanism we can support is:

1. Sparse/noisy sensing creates a forecastability frontier.
2. In the entrance band, CSDI moves contexts closer to the clean trajectory in
   raw-patch statistics and Panda-token geometry, and this is where the largest
   forecastability rescue occurs.
3. Near the frontier floor, distance-to-clean metrics become mixed; CSDI can
   retain a smaller survival advantage even when a single raw/token fidelity
   metric no longer separates it from linear interpolation.
4. The residual structure matters: iid noise of matched magnitude and shuffled
   residuals do not reproduce the CSDI-unique SP82 gains.

We therefore avoid the overclaim that we have fully characterized Panda's
internal failure channel. What is settled is the empirical law: the sparsity
frontier is real, CSDI crosses it in the transition band, and the mechanism is
regime-dependent rather than a universal one-line tokenizer story.

### 4.5 Scope Conditions

The delay-manifold companion assumes a smooth attractor and a useful
finite-dimensional Takens representation. Mackey-Glass and Chua violate this
comfort zone in different ways: Mackey-Glass is a scalar delay equation whose
effective state is infinite-dimensional under the observation window, and Chua
has piecewise-linear/non-smooth circuit dynamics. We report these systems as
scope boundaries. They are not used to inflate the main claim, and they prevent
the method section from sounding like a universal chaos solver.
