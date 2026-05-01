# Forecastability Frontiers Under Corrupted Chaotic Contexts

**Authors.** (anonymous for review)  **Code & data.** Released with the paper.

---

## Abstract

Pretrained chaotic forecasters fail across sharp sparse-observation forecastability
frontiers. Inside the sparsity transition band, **corpus-pretrained structured
imputation** is the lever that reliably moves Panda back across the frontier:
both CSDI (a corruption-aware diffusion imputer) and a SAITS imputer pretrained
on the same chaos corpus rescue Panda strongly, with CSDI retaining a
small but paired-CI-strict advantage at the entrance band (L63 SP65: CSDI − SAITS
= +0.41 Λ, 95 % CI [+0.05, +0.87]) and the two becoming statistically
indistinguishable at the floor band (SP82: +0.06 Λ, [−0.31, +0.59]). The
mechanism is strongest in the entrance band: CSDI produces large reductions in
both raw-patch and Panda-token distance to the clean context (linear/CSDI
distance ratios of roughly 12 to 34 across local stdev, lag-1 autocorrelation,
mid-frequency power, and Panda's patch / embedder / encoder / pooled stages on
L63 SP65). Near the floor, Panda-token distances still favor CSDI but one raw
temporal metric becomes mixed, so distance-to-clean is informative but not a
complete account of tail survival. CSDI is therefore a sparse-gap imputation
lever, not a generic dense-noise denoiser, and structured imputation residuals
are not fully interchangeable with iid noise of matched magnitude, especially in
tail survival probability rather than mean VPT. Delay-manifold forecasting
(DeepEDM in Takens coordinates) supplies a complementary dynamics-aware route
through the same frontier, with explicit scope boundaries on non-smooth systems
such as Chua and scalar delay-differential systems such as Mackey-Glass.

Headline numbers. On **L63 SP65** ($s = 0.65$, $\sigma = 0$), CSDI-filled Panda
reaches mean VPT 2.86 Lyapunov times versus 1.22 for linear-filled Panda
(paired-bootstrap CI [+1.40, +1.87]); $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ rises
from 70 % (Wilson 95 % [39 %, 90 %]) to 100 % ([72 %, 100 %]) at $n=10$. On
**L63 SP82**, the gain is +1.00 Lyapunov times (CI [+0.54, +1.51]) and
$\Pr(\mathrm{VPT}>1.0\,\Lambda)$ moves from 0 % to 60 %. On **L96 N=20 SP82**
($n = 10$ seeds, patched protocol), Panda's mean is dominated by rare
long-survival linear seeds and is reported as a non-headline statistic; the
patched-protocol headline statistics are Panda median 0.50 → 1.05,
$\Pr(\mathrm{VPT}>0.5\,\Lambda)$ 60 % → 100 %, and DeepEDM paired
CSDI − linear gain +0.43 Λ (CI [+0.29, +0.57]). On the pure-noise axis at
every tested $\sigma > 0$ with $s = 0$, CSDI is neutral or slightly hurtful
for Panda — confirming the gap-imputation framing.

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

The intervention claim is **empirical and conditional**, and we report the
underlying cells rather than introduce a regime taxonomy.

- **L63 SP65** ($s = 0.65$, $\sigma = 0$, $n = 10$ seeds). `CSDI → Panda`
  mean VPT 2.86 vs 1.22 for `linear → Panda`; paired Δ = +1.64 Λ,
  CI [+1.40, +1.87]. $\Pr(\mathrm{VPT} > 1.0\,\Lambda)$ rises from
  70 % (Wilson 95 % [40 %, 89 %]) to 100 % ([72 %, 100 %]). Raw-patch and
  Panda-token distances both move toward clean (§4.2); iid jitter and
  shuffled CSDI residuals do not reproduce the gain.
- **L63 SP82** ($s = 0.82$, $\sigma = 0$, $n = 10$). CSDI Δ = +1.00 Λ,
  CI [+0.54, +1.51]. $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ rises from 0 %
  ([0 %, 28 %]) to 60 % ([31 %, 83 %]). iid jitter and shuffled residual
  paired CIs cross zero. Panda-space distances still favor CSDI;
  lag-1 raw autocorrelation becomes mixed (§4.2).
- **L96 N=20 SP65** ($s = 0.65$, $\sigma = 0$, $n = 10$). Panda mean is
  dominated by rare long-survival linear seeds and is **not** the
  appropriate summary; we report tail survival
  $\Pr(\mathrm{VPT} > 1.0\,\Lambda) = 80\%$ ([49 %, 94 %]) for CSDI vs
  40 % ([17 %, 69 %]) for jitter / shuffled vs 40 % ([17 %, 69 %]) for
  linear. DeepEDM paired CSDI − linear is strict-positive across
  SP55–SP82.
- **L96 N=20 SP82** ($s = 0.82$, $\sigma = 0$, $n = 10$). Panda mean is
  again high-variance; the patched-protocol headline is median 0.50 →
  1.05 and $\Pr(\mathrm{VPT}>0.5\,\Lambda)$ 60 % ([31 %, 83 %]) → 100 %
  ([72 %, 100 %]). DeepEDM paired CSDI − linear = +0.43 Λ,
  CI [+0.29, +0.57].
- **Pure-noise axis** ($s = 0$, $\sigma > 0$). CSDI is neutral or slightly
  hurtful for Panda at every tested $\sigma$. CSDI is therefore a
  sparse-gap imputation lever, not a generic dense-noise denoiser.

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

**Mechanism in the entrance band.** We show that the L63 SP65 rescue coincides
with large reductions in raw-patch and Panda-token distance to clean
(linear/CSDI distance ratios of 6–22× across four representation stages and
three temporal statistics). At SP82 those distances still favor CSDI but one
raw temporal metric becomes mixed; mechanism is therefore strong but not
fully reducible to a single fidelity scalar.

**Controls.** Jitter and shuffled-residual controls show that iid noise of
matched magnitude does not reproduce the CSDI gain in the cells where the
gain is largest, and that pure-noise corruption is **not** rescued by CSDI.

**Delay-manifold companion.** Delay-coordinate forecasting (DeepEDM in Takens
coordinates) is included as a complementary route through the same frontier;
its strongest evidence is the strict-positive CSDI − linear paired CI on
L96 N=20 across SP55–SP82.

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

## 3 Empirical Forecastability Frontiers

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

### 4.4 Alt-imputer comparison

We ask whether structured imputation in general is the lever, or whether
CSDI's specific dynamics-aware diffusion prior is required, by adding a
pretrained SAITS imputer trained on the same chaos corpus that CSDI is
trained on (~64K independent-IC L63 windows of length 128, with v2-grid-
matched missingness). At inference SAITS imputes the test context in
non-overlapping length-128 chunks — its pretrained context length —
which is the natural deployment for a fixed-window-attention imputer.

| Cell | L63 SP65 (n=10) | L63 SP82 (n=10) |
|:--|:--:|:--:|
| `linear → Panda` mean VPT | 1.22 | 0.29 |
| `SAITS-pretrained → Panda` | **2.49** | **1.51** |
| `CSDI → Panda` | **2.89** | **1.57** |

| Paired contrast | SP65 Δ (95 % CI) | SP82 Δ (95 % CI) |
|:--|:-:|:-:|
| SAITS-pretrained − linear | +1.26 [+0.83, +1.64] ↑ | +1.23 [+0.86, +1.62] ↑ |
| CSDI − linear | +1.67 [+1.41, +1.92] ↑ | +1.28 [+0.73, +1.85] ↑ |
| **CSDI − SAITS-pretrained** | **+0.41 [+0.05, +0.87] ↑** | **+0.06 [−0.31, +0.59] ≈** |

Tail survival probability $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ (Wilson 95 % CI):

| Cell | SP65 | SP82 |
|:--|:-:|:-:|
| linear | 70 % [40 %, 89 %] | 0 % [0 %, 28 %] |
| SAITS-pretrained | 90 % [60 %, 98 %] | 70 % [40 %, 89 %] |
| CSDI | 100 % [72 %, 100 %] | 70 % [40 %, 89 %] |

**Cross-system replication: L96 N = 20 SP82 (n = 10).** We additionally
pretrain a SAITS imputer on the L96 N = 20 chaos corpus
(`lorenz96_clean_512k_L128_N20.npz`, 64 K windows of length 128, same
v2-grid-matched missingness, val MAE 1.07 = 0.29 × `attractor_std`).
L96 SP82 mean VPT is heavily skewed by a single linear-cell outlier
(seed 2, VPT@1.0 = 10.75 — a clean-context fluke), so per the L96
high-variance limitation noted in §6.4 we lead with median + survival:

| Cell | L96 SP82 median VPT | $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ (Wilson 95 %) |
|:--|:-:|:-:|
| `linear → Panda` | 0.50 | 30 % [11 %, 60 %] |
| `SAITS-pretrained → Panda` | 0.84 | 40 % [17 %, 69 %] |
| `CSDI → Panda` | **1.13** | **60 %** [31 %, 83 %] |

Paired-bootstrap on means: CSDI − SAITS-pretrained = +0.21 [+0.00, +0.49]
(just touching zero); CSDI − linear = −0.03 [−1.40, +0.88] and
SAITS-pretrained − linear = −0.24 [−1.55, +0.51] both straddle zero
(driven by the linear-seed-2 outlier — see Appendix C for the per-seed
table). The qualitative hierarchy on median and survival is the same as
L63 SP65: linear < SAITS-pretrained < CSDI.

Together, L63 SP65 + SP82 + L96 SP82 establish that the §1 intervention
claim narrows from "CSDI is the only tested intervention" to
"**corpus-pretrained structured imputation is the lever**, with CSDI
retaining a small but paired-CI-strict advantage in the entrance band
on L63 and on median + survival on L96, and being indistinguishable
from SAITS-pretrained in the L63 floor band". The phenomenon — that a
corpus-pretrained structured imputer crosses the sparse-observation
transition band where linear interpolation collapses — is therefore
not unique to CSDI; it generalises to at least one other corpus-pretrained
imputer trained on the same data and inference-matched to its training
context length, on two distinct chaotic systems (3-D L63 and 20-D L96).

A standalone single-trajectory SAITS / BRITS sanity check (no pretraining
corpus, per-instance fit on the test trajectory) is reported in Appendix E
as supporting observation. Per-instance training is biased against
SAITS / BRITS by design and is not the primary alt-imputer comparison.

Glocal-IB and other recent global-structure imputers are not evaluated;
they are listed in §2 as adjacent prior art on high-missingness
imputation and remain a natural follow-up.

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
$\{$Panda, DeepEDM$\}$ forecasters. To remove a Panda-tokenizer-specific
attack surface we additionally evaluate Chronos [Ansari24] on the L63
sparsity transition band (SP55–SP82) with `linear → Chronos` and
`CSDI → Chronos`; cross-foundation-model evidence is reported in §3.2.
The alt-imputer comparison in §4.4 uses a SAITS imputer pretrained on the
same chaos corpus that CSDI is trained on, so the comparison is fair on the
training-data axis.

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
that, among the imputers we tested with matched corpus pretraining (linear,
Kalman, CSDI, SAITS), CSDI gives the strongest paired CSDI − linear gain in
the L63 transition band (§4.4). We do not claim that the mechanism is fully
characterized; raw-patch and Panda-token distance to clean explains the
entrance band, while floor-band survival is informed by but not reducible to
those distances. We do not claim universality across all foundation
forecasters; Panda-72M is the headline. Chronos is reported on the L63
sparsity line as cross-foundation-model evidence (§3.2), but TimesFM /
Lag-Llama are not evaluated.

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

- **Imputation training-corpus axis.** The alt-imputer comparison in §4.4
  pairs CSDI against a SAITS model pretrained on the same chaos corpus
  (~500K independent-IC L63 windows). A standalone single-trajectory
  per-instance SAITS / BRITS sanity check is reported in Appendix E. We
  have not evaluated Glocal-IB or other recent global-structure imputers;
  these are listed as adjacent prior art in §2 and are an open follow-up.
- **Forecaster breadth.** Panda is the headline. We additionally evaluate
  Chronos-bolt-small on the L63 sparsity line (SP55–SP82, 5 seeds,
  `linear → Chronos` and `CSDI → Chronos`) at two horizons: `pred_len = 128`
  (matched to Panda) and `pred_len = 64` (Chronos's native trained
  horizon, since the Chronos library warns `prediction_length > 64` is
  outside its training distribution). At both horizons Chronos has
  substantially lower absolute VPT than Panda across the entire SP line
  (mean 0.34–0.50, $\Pr(\mathrm{VPT}>1.0)$ ≤ 20 %), and CSDI does not
  visibly improve it (paired Δ all near zero with CIs straddling zero).
  Per-seed VPTs at `pred_len = 64` are statistically indistinguishable
  from `pred_len = 128`, confirming the negative is not an artefact of
  the Chronos out-of-distribution horizon. We therefore read this as:
  "the corpus-pretrained-imputation lever is empirically observed for
  Panda; on Chronos the rescue lever is unobservable because Chronos's
  own VPT distribution sits well below the regime where the lever can
  move it." Matched-horizon evaluations of TimesFM / Lag-Llama and other
  large pretrained time-series forecasters remain future work.
- **Known-dynamics upper bound.** A model-aware reference (stochastic
  EnKF with the true L63 vector field, 100 ensemble members) saturates
  at the VPT ceiling across the entire sparse-observation transition
  band (SP55–SP82 mean 2.84–2.85, $\Pr(\mathrm{VPT}>1.0) = 100 \%$ at all
  cells; Appendix B). The frontier is therefore a property of the
  **black-box deployment interface** (no access to dynamics for
  forecasting), not of the system itself.
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

Code, CSDI checkpoints, pretrained-SAITS baselines, and the locked Figure-1 /
isolation / jitter / embedding data are released. Glocal-IB, real-data case
studies, and decoder-side Panda instrumentation remain natural camera-ready
follow-ups.

---

> Appendices below are aligned to the locked story (2026-05-01). Pre-pivot
> material describing the original four-module pipeline (M1–M4) and the
> universal tokenizer-OOD theorem family is preserved verbatim in
> `deliverable/paper/paper_draft_en_archive_2026-04-30.md` and is referenced
> here only by pointer.

## Appendix A: Theory pointer (inherited, narrowed)

The current main text relies on **empirical** entrance-band OOD reduction
(§4.2) and floor-band residual survival (§4.3) rather than on a closed-form
theorem of "ambient-predictor tokenizer-OOD failure". An earlier draft proved
a Theorem 2 with that universal phrasing; under the v2 protocol this theorem
holds only as an entrance-band statement. We retain the proof material in
`paper_draft_en_archive_2026-04-30.md` §"Appendix A: Proof sketches" for
readers interested in the formal bound, and treat it here as a *narrowed*
theoretical companion: the same patch-distribution-OOD argument applies at
L63 SP65 (where 6–22× distance-to-clean reductions in Panda token space are
measured), but not as a universal claim across the whole frontier. The
main-text intervention and frontier claims do not depend on the proof.

## Appendix B: Reproducibility and Experiment Table

### B.1 Locked v2 protocol

- **L63**: `LORENZ63_ATTRACTOR_STD = 8.51`, `dt = 0.025`, `n_ctx = 512`,
  corruption seed `1000 × seed + 5000 + grid_index` where `grid_index` is
  the cell position in `experiments/week1/configs/corruption_grid_v2.json`.
- **L96 N=20**: `lorenz96_attractor_std(N=20, F=8) = 3.6387`, `dt = 0.05`,
  `n_ctx = 512`, same seed scheme.
- **Rössler**: `ROSSLER_ATTRACTOR_STD = 4.45`, `dt = 0.1`,
  `lyap = 0.071`, same seed scheme.
- **CSDI inference**: `set_csdi_attractor_std()` matches the system, and
  `sigma_override = noise_std_frac × attractor_std` is passed to
  `impute(observed, kind="csdi", sigma_override=...)`. For pure-sparsity
  cells (σ=0), `sigma_override = 0` exactly, which leaves observed
  timestamps anchored to ~10⁻⁶ (verified in
  `deliverable/CSDI_SANITY_FINDINGS.md`).
- **VPT**: Lyapunov-time normalized; threshold-crossing definition with
  per-axis attractor standard deviation as scale.
- **CIs**: 95 % bootstrap on means (5000 resamples), Wilson 95 % on
  binomial survival probabilities at the listed seed count.

### B.2 Experiment table

All results below use the v2 protocol. JSON paths are relative to the
repository root.

| # | Experiment | System | Scenarios | Cells | Seeds | Result JSON | Aggregated to |
|---|---|---|---|---|---:|---|---|
| 1 | Figure 1 v2 grid (sparsity line) | L63 | SP00–SP97 (10) | linear/CSDI × Panda/DeepEDM (4) | 10 | `experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_{h0,h5}.json` | `deliverable/figures_main/figure1_l63_v2_10seed_patched.{png,md}` |
| 2 | Figure 1 v2 grid (noise line) | L63 | NO00–NO120 (8) | same as #1 | 10 | `experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_{h0,h5}.json` | same as #1 |
| 3 | L96 N=20 v2 cross-system | L96 N=20 | SP55/SP65/SP75/SP82 + NO010/NO020/NO050 | linear/CSDI × Panda/DeepEDM (4) | 10 (5 + 5 extension) | `pt_l96_smoke_l96N20_v2_B_patched_5seed.json`, `pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json` | `deliverable/L96_V2_B_PATCHED_N10.md` |
| 4 | L63 jitter / residual control | L63 | SP65, SP82 | linear, +iid jitter, +shuffled residual, CSDI | 10 | `panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.json` | `experiments/week1/figures/panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.md` |
| 5 | L96 N=20 jitter / residual | L96 N=20 | SP65, SP82 | same as #4 | 5 | `panda_jitter_control_l96N20_sp65_sp82_v2protocol_patched_5seed.json` | same-prefix `.md` |
| 6 | Rössler jitter / residual | Rössler | SP65, SP82 | same as #4 | 5 | `panda_jitter_control_rossler_sp65_sp82_v2protocol_patched_5seed.json` | same-prefix `.md` |
| 7 | Cross-system jitter milestone | L63+L96+Rössler | SP65, SP82 | from #4–#6 | 5–10 | merge of #4–#6 | `deliverable/figures_jitter/jitter_milestone_summary.md`, `jitter_milestone_SP{65,82}.png` |
| 8 | Panda embedding OOD diagnostic | L63 | SP65, SP82 | clean / linear / CSDI; stages: patch / embed / encoder / pooled | 5 | `panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.json` | `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.md` and `_bars.png` |
| 9 | Raw-patch diagnostic v2 | L63 | SP65, SP82 | clean / linear / CSDI; metrics: local stdev, lag-1 ρ, mid-freq power | 10 | `l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json` | `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP{65,82}.png` |
| 10 | Cross-system isolation matrix (legacy) | L63, L96 N=10/20, Rössler, Kuramoto | S0–S6 | linear/Kalman/CSDI × Panda/DeepEDM (6) | 5 | `pt_{l63,l96_iso_l96N{10,20},rossler_iso_rossler,kuramoto}_*_5seed.json` | `deliverable/figures_isolation/*_heatmap.png`, `*_bars.png`, `*.md` |
| 11 | MG / Chua scope-boundary cases | Mackey-Glass, Chua | S0–S6 | same as #10 | 5 | `pt_{mg,chua}_*_5seed.json` | `deliverable/figures_isolation/` (boundary subset) |
| 12 | Alt-imputer per-instance sanity | L63 | SP65 | linear, SAITS, BRITS, CSDI | 5 | `panda_altimputer_l63sp65_partial_5seed.json` | log-only; Appendix E sanity |
| 13 | **Pretrained alt-imputer (P1.1 + P1.5 cross-system)** | L63, L96 N=20 | L63 SP65 + SP82, L96 SP82 | linear, SAITS-pretrained, CSDI | 10 | `panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`, `panda_altimputer_l96_sp82_pretrained_10seed.json` | §4.4 + Appendix C |
| 14 | **Chronos mini-frontier (P1.2)** | L63 | SP55, SP65, SP75, SP82 | linear, CSDI (forecaster: Chronos, `pred_len ∈ {64, 128}`) | 5 | `chronos_frontier_l63_chronos_l63_sp55_sp82_5seed.json`, `..._5seed_pl64.json` | §3.2 / §6.4 cross-foundation observation; pred_len=64 confirms negative is not an artefact of Chronos OOD horizon |
| 15 | **EnKF known-dynamics upper bound (P1.3)** | L63 | SP55–SP82, NO020, NO050 | EnKF (true vector field, 100 members) | 5 | `enkf_l63_enkf_l63_v2_5seed.json` | §6.5 / Appendix B reference |

Items 1–9 are the patched-protocol locked numbers cited in §3 / §4 / §6.
Item 10 is the cross-system replication that uses the older S0–S6
corruption pipeline (`make_sparse_noisy`) and is cited as **secondary**
direction-of-effect evidence — the v2 protocol numbers in items 1–6 / 8 / 9
are authoritative. Item 11 supplies §6.3 scope conditions. Item 12 is
Appendix E sanity (per-instance training, biased against SAITS / BRITS by
design). Items 13–15 are the P1 reviewer-defense experiments: pretrained
SAITS alt-imputer comparison including the L96 survival replication
(§4.4 / Appendix C), Chronos cross-foundation mini-frontier (§3.2 / §6.4),
and EnKF known-dynamics upper bound
(§6.5 / Appendix B).

### B.3 Aggregator scripts

Each aggregator is invoked as `python -m experiments.week1.<script>`:

- `aggregate_figure1_v2.py --halves --s_tag ... --n_tag ... --out_prefix ...` →
  six-panel Figure 1 with bootstrap CI on mean and Wilson CI on Pr(VPT > θ).
- `aggregate_isolation.py --json <iso JSON> --out_prefix ...` → 2×3 heatmap
  + bar chart with paired bootstrap CI for #10 / #11.
- `aggregate_jitter_cross_system.py` → six-panel Figure 3 (mean vs Pr>1.0
  across L63 / L96 / Rössler at SP65 and SP82).
- `aggregate_corruption_grid.py` → metadata table for any v2 grid run
  (keep fraction, obs/patch, max gap in Lyapunov units).
- `aggregate_survival_summary.py` → cross-system survival probabilities at
  Pr>0.5 and Pr>1.0.

## Appendix C: Pretrained alt-imputer details

**Training.** We pretrain a SAITS [Du22] imputer on
`experiments/week2_modules/data/lorenz63_clean_64k_L128.npz` (≈ 64 K
independent-IC L63 windows, length 128). For each training window we
sample a sparsity uniformly from the v2 `fine_s_line` grid
(`{0, 0.20, 0.40, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97}`) and apply
an iid_time mask, so the SAITS training corruption distribution matches
the v2 evaluation distribution. Architecture: 2 SAITS layers, $d_{model}
= 64$, 4 heads, $d_k = d_v = 16$, $d_{ffn} = 128$. 30 epochs,
batch 64, ≈ 18 min on 1 GPU. Best checkpoint at epoch 30; final
training MAE = 0.47, validation MSE = 8.28, validation MAE on missing
entries = 1.26 (= 0.149 × $\sigma_\text{attr}$).

Checkpoint:
`experiments/week2_modules/ckpts/saits_l63_pretrained/<run-id>/SAITS.pypots`.

**Inference.** SAITS expects a fixed input length matching its
positional encoding. The test context (length 512) is split into 4
non-overlapping length-128 chunks, each is imputed independently, and
the chunks are concatenated. CSDI's natural variable-length inference
is unchanged.

**Results (10 seeds at L63 SP65 + SP82, σ = 0).**

| Cell | SP65 mean | SP82 mean | SP65 Pr>1.0 | SP82 Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.22 | 0.29 | 70 % | 0 % |
| SAITS-pretrained | 2.49 | 1.51 | 90 % | 70 % |
| CSDI | 2.89 | 1.57 | 100 % | 70 % |

| Paired contrast | SP65 Δ (95 % CI) | SP82 Δ (95 % CI) |
|:--|:-:|:-:|
| SAITS − linear | +1.26 [+0.83, +1.64] ↑ | +1.23 [+0.86, +1.62] ↑ |
| CSDI − linear | +1.67 [+1.41, +1.92] ↑ | +1.28 [+0.73, +1.85] ↑ |
| CSDI − SAITS | +0.41 [+0.05, +0.87] ↑ | +0.06 [−0.31, +0.59] ≈ |

**Reading.** The pretrained SAITS reproduces most of the L63 transition-band
rescue. CSDI retains a small but paired-CI-strict advantage at the entrance
band and is statistically indistinguishable from SAITS-pretrained at the
floor band. We therefore narrow the §1 / abstract intervention claim to
"corpus-pretrained structured imputation is the lever; CSDI is one strong
instance with a small entrance-band advantage".

**L96 N = 20 SP82 cross-system replication (10 seeds, σ = 0).** A second
SAITS imputer is pretrained on `lorenz96_clean_512k_L128_N20.npz`
(64 K windows of length 128, same v2-grid-matched missingness, same
architecture as L63). Best checkpoint at epoch 27; validation MAE on
missing entries = 1.07 (= 0.29 × $\sigma_\text{attr}^{(\mathrm{L96})}$).
Per-seed VPT@1.0:

| seed | linear | SAITS-pretrained | CSDI |
|:-:|--:|--:|--:|
| 0 | 0.50 | 0.67 | 0.76 |
| 1 | 0.00 | 0.84 | 1.85 |
| 2 | **10.75** | 4.96 | 5.12 |
| 3 | 0.42 | 0.76 | 0.76 |
| 4 | 3.78 | 4.12 | 4.12 |
| 5 | 1.09 | 1.09 | 1.01 |
| 6 | 0.50 | 1.34 | 1.26 |
| 7 | 0.25 | 0.84 | 0.92 |
| 8 | 0.50 | 0.50 | 0.50 |
| 9 | 0.25 | 0.50 | 1.43 |
| **mean** | 1.81 | 1.56 | 1.77 |
| **median** | 0.50 | 0.84 | **1.13** |

Seed 2 is a clean-context fluke for linear (linear seed-2 keep-fraction
0.15 happens to align with a forecastable Panda token sequence, so all
three cells get a long forecast there). Per the L96 high-variance
limitation noted in §6.4, mean is unreliable; median + survival is the
headline:

| Cell | $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ | Wilson 95 % CI |
|:--|:-:|:-:|
| linear | 30 % | [11 %, 60 %] |
| SAITS-pretrained | 40 % | [17 %, 69 %] |
| CSDI | **60 %** | [31 %, 83 %] |

Paired-bootstrap (5000 resamples) on means: CSDI − SAITS-pretrained
= +0.21 [+0.00, +0.49] (just touching zero); CSDI − linear and
SAITS − linear straddle zero on means (driven by the linear-seed-2
outlier). The qualitative hierarchy on median + survival mirrors L63 SP65:
linear < SAITS-pretrained < CSDI. We read this as "the structured-imputation
lever generalises to a 20-D system on the survival metric the L96 cell
uses; mean is too noisy to read directly."

Glocal-IB is not evaluated (cited in §2 as adjacent prior art on
high-missingness imputation that emphasises preserving global latent
structure); it remains a natural follow-up.

Sources:
- L63: `experiments/week1/results/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`
- L96: `experiments/week1/results/panda_altimputer_l96_sp82_pretrained_10seed.json`

## Appendix D: Figure Index

All figures referenced in the main text use the patched v2 protocol unless
explicitly marked. Main-text labels Figure 1 / 2 / 3 correspond to the three
headline panels.

### Main figures

| Label | Caption purpose | Path |
|---|---|---|
| **Figure 1** | Sparsity and noise frontier on L63 (mean / Pr>0.5 / Pr>1.0, decoupled axes, 10 seeds, 95 % bootstrap CI on mean and Wilson CI on survival) | `deliverable/figures_main/figure1_l63_v2_10seed_patched.png` (and `.md` companion table) |
| **Figure 2** | Cross-system isolation matrix (linear/Kalman/CSDI × Panda/DeepEDM heatmaps and paired-CI bars; legacy S0–S6 protocol) | `deliverable/figures_isolation/{l63,l96_iso_l96N{10,20},rossler_iso_rossler}_5seed_heatmap.png` and `_bars.png` |
| **Figure 3** | Jitter / residual controls across L63, L96 N=20, Rössler at SP65 and SP82, comparing mean vs `Pr(VPT > 1.0 Λ)` | `deliverable/figures_jitter/jitter_milestone_SP{65,82}.png` |

### §4.2 mechanism panels

| Element | Path |
|---|---|
| L63 raw-patch v2 metric histograms (local stdev / lag-1 ρ / mid-freq power, SP65 + SP82) | `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP{65,82}.png` |
| L63 raw-patch trajectory overlays (clean vs linear vs CSDI) | `experiments/week1/figures/l63_patch_ood_v2_v2protocol_traj_overlay_SP{65,82}.png` |
| Panda token-space distance bars (patch / embed / encoder / pooled, SP65 + SP82) | `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed_bars.png` |
| Panda token-space PCA scatter (per stage and scenario) | `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_5seed_SP{65,82}_{embed,encoder}_pca.png` |

### §6.3 scope-boundary panels (appendix only)

| Element | Path |
|---|---|
| Mackey-Glass S0–S6 phase-transition + trajectory | `experiments/week1/figures/iso_mackey_glass_5seed_*.png` (and trajectory plots in the same directory) |
| Chua S0–S6 phase-transition + trajectory | `experiments/week1/figures/iso_chua_5seed_*.png` |

### Pre-pivot figures (no longer cited in main text)

The previous draft cited a number of figures specific to the four-module
pipeline (M1 imputation traces, M2 τ-search Bayesian-optimisation curves,
M3 backbone comparison, M4 conformal coverage). These remain in
`deliverable/figures_extra/` and `experiments/week2_modules/figures/` but
are not part of the locked story; the archive at
`paper_draft_en_archive_2026-04-30.md` keeps their original captions.

## Appendix E: Inherited supplementary material (archived)

Three appendix sections from the pre-pivot draft are no longer load-bearing
for the locked story but remain available for readers interested in the
underlying engineering:

- **Original Appendix E — τ-search detailed evidence.** Mutual-information /
  Lyapunov objective, Bayesian-optimisation traces, ablation against random
  τ. Used to motivate the lag schedule of DeepEDM in §5.2.
- **Original Appendix F — τ-coupling complete empirical analysis.** Per-system
  sensitivity of forecastability to the lag schedule. Confirms that DeepEDM
  lag choice is not a knife-edge tuning artefact.
- **Original Appendix G — Delay-manifold perspective.** Geometric framing
  (Takens embedding, attractor reconstruction, smoothness assumptions) that
  motivates the §5.2 companion forecaster and the §6.3 scope conditions.

Source: `paper_draft_en_archive_2026-04-30.md`, sections of the same names.
None of these are required to verify the §1 / §3 / §4 / §6 claims; they
strengthen the §5 method exposition and the §6.3 scope explanation.
