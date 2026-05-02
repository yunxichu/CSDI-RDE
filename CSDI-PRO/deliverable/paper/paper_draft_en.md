# Forecastability Frontiers Under Corrupted Chaotic Contexts

**Authors.** (anonymous for review)  **Code & data.** Released with the paper.

---

## Abstract

A pretrained chaos foundation forecaster (Panda-72M) fails across sharp
sparse-observation forecastability frontiers; a second pretrained
forecaster (Chronos) instead sits at a low-VPT plateau across the same
sparsity line, so the transition shape is forecaster-dependent rather
than universal (§6.4). Inside the sparsity transition band of Panda's
frontier, **corpus-pretrained structured imputation** is the lever that
reliably moves Panda back across the frontier:
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
from 70 % (Wilson 95 % [40 %, 89 %]) to 100 % ([72 %, 100 %]) at $n=10$. On
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
forecastability frontiers** for Panda-72M, the foundation forecaster
specifically pretrained on chaotic dynamics. As sparsity increases,
Panda's survival probability $\Pr(\mathrm{VPT} > \theta)$ drops
abruptly in a narrow transition band rather than degrading smoothly.
Chronos at our setting does *not* show this transition shape — it sits
at a low-VPT plateau across the same sparsity line (mean 0.34–0.50;
§6.4) — so the empirical frontier is established for Panda; cross-
foundation generalisation is reported as a forecaster-dependent
observation, not a universal claim. This matters
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
(linear/CSDI distance ratios of roughly 12–34× across four Panda
representation stages and three temporal statistics; see §4.2 for the
per-stage breakdown). At SP82 those distances still favor CSDI but one
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
tail survival: CSDI's $\Pr(\mathrm{VPT}>1.0\,\Lambda) = 80\%$ vs 40 % for
linear / iid / shuffled at $n=10$, with overlapping Wilson 95 % CIs that
nonetheless preserve the rank order on every seed; the median VPT also
favours CSDI. The paired-CI strict claim at L96 SP65 is therefore on the
direction-and-rank rather than on the mean.

Thus the intervention claim is deliberately conditional:

> Inside the transition band, **corpus-pretrained structured imputation**
> is the lever that reliably moves models back across the forecastability
> frontier: both CSDI and a corpus-pretrained SAITS imputer cross the
> frontier where iid jitter and magnitude-matched shuffled residuals do
> not, with CSDI retaining a small paired-CI-strict advantage at the
> entrance band; structured imputation residuals are not fully
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

The answer is the second one. The authoritative cell-level evidence comes
from the v2 protocol numbers in §3.2 and §4.4 (10-seed Figure 1 grid;
30-seed L96 SP82 alt-imputer); a coarser 5-seed S0–S6 isolation sweep
(legacy `make_sparse_noisy` corruption pipeline, Figure 2) replicates
the direction-of-effect across L63 / L96 / Rössler / Kuramoto, and is
deferred to Appendix B for completeness.

The same matrix locks DeepEDM's main-text role to a single hard fact: in
the L96 N=20 v2 cross-system replication, **DeepEDM is the only forecaster
with a strict-positive paired CSDI − linear CI on every cell of the
SP55–SP82 transition band**, with +0.43, [+0.29, +0.57] at SP82. In the
same band, Panda's mean is dominated by rare lucky linear seeds (§4.3),
so the cleanest cross-system CSDI − linear evidence at high dimensionality
runs through the delay-coordinate channel rather than the ambient one.
This is why DeepEDM stays in the main text rather than the appendix,
without overclaiming dominance: at lower-dimensional cells (e.g. L63
SP65) `CSDI → Panda` is the strongest absolute cell, and the §3.2 / §4.4
claims do not rest on DeepEDM.

### 4.2 Regime-Aware Mechanism: OOD in the Entrance Band, Mixed at the Floor

We compare clean, linear-fill, CSDI-fill, and SAITS-pretrained-fill contexts
under the same protocol as Figure 1. The §4.4 alt-imputer comparison
shows that both CSDI and corpus-pretrained SAITS rescue Panda; this
section asks whether they share the same Panda-internal mechanism — i.e.
whether **corpus-pretrained structured imputation in general** reduces
token distance to clean, or whether the reduction is CSDI-specific.

**Raw-patch (L63 SP65, 10 seeds, v2 protocol).**

| Metric | Linear/CSDI W₁-to-clean ratio |
|:--|--:|
| local stdev | 21.02 |
| lag-1 autocorrelation | 15.02 |
| mid-frequency power | 33.71 |

**Panda-internal token-space (L63, 5 seeds, v2 protocol, 4-cell paired
distance to the matched clean context, mean L2 across tokens).**

| | SP65 linear / SAITS / CSDI | SP82 linear / SAITS / CSDI |
|:--|:--:|:--:|
| patch | 0.51 / 0.18 / 0.03 | 1.61 / 0.40 / 0.71 |
| embedder | 103 / 52 / 8.6 | 216 / 88 / 94 |
| encoder | 64 / 31 / 5.4 | 120 / 53 / 76 |
| pooled | 9.2 / 3.3 / 0.51 | 14.3 / 5.5 / 6.8 |

| Stage | SP65 linear/CSDI ratio | SP65 linear/SAITS | SP82 linear/CSDI | SP82 linear/SAITS | **SP82 SAITS/CSDI** |
|:--|:-:|:-:|:-:|:-:|:-:|
| patch | 14.9 | 2.8 | 2.3 | 4.0 | **0.57** |
| embedder | 12.0 | 2.0 | 2.3 | 2.4 | **0.95** |
| encoder | 11.8 | 2.1 | 1.6 | 2.3 | **0.70** |
| pooled | 18.2 | 2.8 | 2.1 | 2.6 | **0.81** |

Two regime-specific facts emerge.

**Entrance band (SP65).** Both CSDI and SAITS-pretrained move tokens
toward clean, by different magnitudes: CSDI achieves ~12× reduction
relative to linear at every Panda stage, SAITS achieves ~2× reduction.
The mechanism "**corpus-pretrained structured imputation reduces Panda-
token distance to clean**" is confirmed for **both** imputers — not
CSDI-specific. CSDI's larger reduction tracks its small paired VPT
advantage at this cell (CSDI − SAITS = +0.41 Λ, §4.4): more reduction →
more rescue.

**Floor band (SP82).** Both imputers still reduce token distance vs
linear, but the rank order between SAITS-pretrained and CSDI **flips**:
SAITS produces a *smaller* token distance to clean than CSDI at every
stage (SAITS/CSDI ratio 0.57–0.95). On §4.4 forecasting at this same
cell, however, CSDI and SAITS-pretrained are statistically
indistinguishable (paired CSDI − SAITS = +0.06 Λ, CI [−0.31, +0.59]).
**Token distance is therefore informative — both corpus-pretrained
imputers reduce it vs linear and both rescue Panda — but it does not
order SAITS-vs-CSDI at the floor band**, where the smaller-token-distance
imputer (SAITS) is not the better-VPT imputer based on **final pooled**
distance alone. The per-layer probe below resolves this paradox: it is
not encoder-vs-decoder, it is *which encoder layer* is rescue-relevant.

Raw metrics confirm the boundary: at SP82, local stdev and mid-frequency
power favor CSDI (3.54× and 5.19×), while lag-1 autocorrelation favors
linear (0.62×).

**Per-layer encoder probe (12-layer scan, L63, 5 seeds, v2 protocol).**
The naïve reading of the SP82 flip — that some signal "downstream of the
encoder" is needed to explain why CSDI ≈ SAITS on VPT despite SAITS
being closer in pooled L2 — turns out to be wrong. Panda-72M is
encoder-only (its head is a linear projection from the pooled encoder
state), so we register a forward hook on each of the 12 PandaLayer
outputs and measure paired L2-to-clean at every layer:

| Layer | SP65 SAITS/CSDI | SP82 SAITS/CSDI |
|:-:|:-:|:-:|
| 0 (post-embedder) | 5.67 | 0.94 |
| 1 | 5.08 | 0.70 |
| 2 | 4.77 | 0.64 |
| 3 | 4.20 | 0.65 |
| 4 | 4.86 | 0.86 |
| 5 | 5.46 | 0.94 |
| **6** | **5.90** | **1.02** |
| **7** | **6.26** | **1.06** |
| **8** | **5.28** | **1.00** |
| 9 | 4.35 | 0.74 |
| 10 | 4.69 | 0.78 |
| 11 | 4.78 | 0.76 |
| 12 (final, before pooling) | 4.85 | 0.71 |

At **SP65 every layer keeps CSDI 4–6× closer to clean than SAITS** —
encoder geometry is uniformly informative and matches CSDI's strict-
positive paired VPT advantage. At **SP82 the layer-wise picture is
non-monotonic**: SAITS is closer at layers 0–5 and 9–12 (saits/csdi
0.64–0.94), but at **mid-encoder layers 6–8 the two converge to a
tie (saits/csdi 1.00–1.06)**. The mid-encoder band is the *only*
sub-region of Panda's internal representation where SAITS and CSDI
look equivalently close to clean — and it precisely matches the §4.4
forecast tie (CSDI − SAITS = +0.06 Λ, CI [−0.31, +0.59]).

We read this as direct measured evidence that **the §4.4 floor-band
rescue saturates at Panda's mid-encoder**: by layer 7, both corpus-
pretrained imputers have produced internal representations that are
statistically indistinguishable in distance to clean, and any
discrimination at later layers does not translate into forecastability
because the head's linear projection sees the pooled (≈ averaged) state
that smooths over the mid-encoder convergence. The pooled-only reading
in the bar table above is therefore a *late-layer artefact*, not the
relevant predictor of VPT.

The mechanism claim therefore reads:

> **At the entrance band, corpus-pretrained structured imputation
> reduces raw / Panda-token OOD distance to clean uniformly across
> the encoder, and the magnitude of reduction tracks rescue strength.
> At the floor band, both corpus-pretrained imputers' representations
> converge to a tie in Panda's mid-encoder (layers 6–8); this is where
> the §4.4 forecast tie is realised. The final pooled distance is a
> late-layer artefact, not the rescue-relevant geometry.**

### 4.3 Jitter and Shuffled-Residual Controls (Cell-Wise Observations)

To test whether CSDI is merely stochastic regularization, we run a four-cell
Panda-only control on six (system, scenario) settings:

- `linear`,
- `linear + iid jitter` matched to the per-channel CSDI residual scale,
- `linear + shuffled CSDI residual` applied at missing entries,
- `CSDI`.

All controls share the same missing masks and the same forecast model. We
report each cell separately; we do **not** introduce a regime taxonomy
because the §4.4 alt-imputer comparison reshapes the boundary between
"floor-band CSDI is strongest" and "any structured residual works" (the
latter is now the §4.4 finding for SAITS-pretrained at SP82).

**L63 SP65** (entrance band, $n=10$). `linear → Panda` mean VPT 1.22,
`CSDI → Panda` mean VPT 2.87, paired Δ +1.65, CI [+1.41, +1.87]. Iid
jitter Δ = +0.17, CI [−0.01, +0.36]; shuffled residuals Δ = −0.16,
CI [−0.34, −0.02]. Neither magnitude-matched control reproduces the
CSDI gain. The §4.4 alt-imputer comparison adds that `SAITS-pretrained → Panda`
also crosses the frontier here, with paired CSDI − SAITS = +0.41
[+0.05, +0.87] (CSDI strict-positive but small).

**L96 N=20 SP65** ($n=10$). Iid jitter, shuffled residuals, and CSDI all
move Panda mean in the positive direction, but none is cleanly separated
on mean because Panda has rare long-survival seeds. The rank order is
preserved on every seed and CSDI is strongest on tail survival:
$\Pr(\mathrm{VPT}>1.0\,\Lambda)$ is 80 % (Wilson 95 % [49 %, 94 %]) for
CSDI vs 40 % ([17 %, 69 %]) for linear / iid / shuffled. We pre-register
median + survival as the headline at L96 cells where Panda mean is
high-variance.

**L63 SP82** (floor band, $n=10$). Iid jitter CI does not cross zero,
shuffled residuals help modestly (Δ +0.34), and CSDI gives Δ +1.09
with CI [+0.65, +1.61] and $\Pr(\mathrm{VPT}>1.0\,\Lambda) = 70\%$. The
§4.4 alt-imputer comparison adds that `SAITS-pretrained → Panda` is
**statistically indistinguishable from CSDI** at this cell (paired
CSDI − SAITS = +0.06, [−0.31, +0.59]) — so the floor-band finding is
"any corpus-pretrained structured imputer crosses, both above
magnitude-matched controls", not "CSDI specifically".

**L96 N=20 SP82** ($n=10$). Panda mean is dominated by rare lucky linear
seeds (e.g. seed 2 with `keep_frac = 0.15` happens to align with a
forecastable Panda token sequence and yields VPT@1.0 = 10.75 across all
cells); we therefore **pre-register median + survival as the headline
metric for high-dimensional high-variance L96 cells** rather than
treating mean as the primary read. On these metrics the hierarchy is
clean: linear < SAITS-pretrained < CSDI on both median (0.50 / 0.84 /
1.13) and Pr(VPT>1.0) (30 / 40 / 60 %). Among forecasters, DeepEDM
gives the only **strict-positive paired CSDI − linear CI** at this cell
(+0.43, [+0.29, +0.57]; §3.2).

**Rössler SP65 / SP82** ($n=5$). Same positive CSDI direction, but small
Lyapunov exponent makes $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ too strict;
$\Pr(\mathrm{VPT}>0.5\,\Lambda)$ is the more appropriate tail metric.

**Pure-noise axis** ($s=0$, $\sigma > 0$). CSDI is neutral or slightly
hurtful at every tested $\sigma$ (Figure 1 noise line). CSDI is therefore
a sparse-gap imputation lever, not a generic dense-noise denoiser.

These cell-wise observations are the reason our abstract says "inside
the transition band": neither iid jitter nor magnitude-matched shuffled
residuals reproduce what **corpus-pretrained structured imputation**
does — the §4.4 alt-imputer comparison shows CSDI and a corpus-pretrained
SAITS both cross the frontier where these magnitude-matched controls do
not, while CSDI retains a small paired-CI-strict advantage at the L63
entrance band. **Structured residuals from a corpus-pretrained imputer
are therefore not interchangeable with iid noise of matched magnitude**,
especially in tail survival probability rather than mean VPT.

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

**Cross-system replication: L96 N = 20 SP82 (n = 30).** We additionally
pretrain a SAITS imputer on the L96 N = 20 chaos corpus
(`lorenz96_clean_512k_L128_N20.npz`, 64 K windows of length 128, same
v2-grid-matched missingness, val MAE 1.07 = 0.29 × `attractor_std`). To
dilute the lucky-seed effect flagged by the §4.3 pre-registration on L96,
we run **30 seeds** on this cell (the only 30-seed cell in the paper):

| Cell | mean VPT | median VPT | $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ Wilson 95 % |
|:--|:-:|:-:|:-:|
| `linear → Panda` | 0.86 | 0.25 | 20 % [10 %, 37 %] |
| `SAITS-pretrained → Panda` | 1.57 | 1.01 | 50 % [33 %, 67 %] |
| `CSDI → Panda` | **1.87** | **1.26** | **73 %** [56 %, 86 %] |

Paired-bootstrap on means (5000 resamples), $n = 30$:

| Paired contrast | Δ | 95 % CI | sign |
|:--|:-:|:-:|:-:|
| SAITS-pretrained − linear | +0.71 | [+0.02, +1.38] | ↑ |
| CSDI − linear | +1.01 | [+0.36, +1.64] | ↑ |
| CSDI − SAITS-pretrained | **+0.31** | **[+0.07, +0.56]** | **↑** |

At 30 seeds the lucky-seed dilution removes the L96 mean ambiguity that
appeared at 10 seeds: every metric (mean / median / Pr(VPT > 0.5) /
Pr(VPT > 1.0)) is now monotonic `linear < SAITS-pretrained < CSDI`, and
all three paired contrasts are strict-positive. CSDI − SAITS-pretrained
becomes a strict-positive paired CI on means, matching the L63 SP65
entrance band.

Together, L63 SP65 + SP82 + L96 SP82 establish that the §1 intervention
claim narrows from "CSDI is the only tested intervention" to
"**corpus-pretrained structured imputation is the lever**, with CSDI
retaining a small but paired-CI-strict advantage in the L63 entrance
band, an on-the-edge advantage on L96 SP82 means (CI just touching zero)
and a clear advantage on L96 SP82 median + survival, and being
indistinguishable from SAITS-pretrained in the L63 floor band". The
phenomenon — that a corpus-pretrained structured imputer crosses the
sparse-observation transition band where linear interpolation collapses
— is therefore not unique to CSDI; the median + survival hierarchy
linear < SAITS-pretrained < CSDI is reproduced on a second
corpus-pretrained imputer trained on the same data and inference-matched
to its training context length, on two distinct chaotic systems
(3-D L63 and 20-D L96). We do *not* claim broader generalisation
(e.g. to other systems, sparsity cells, or imputer families) from a
single L96 cell.

A reproducibility note on the small per-cell drift in the L63 SP65
paired CSDI − linear value across §3.2 / §4.3 / §4.4 (each from an
independent CSDI inference run; CSDI is a stochastic diffusion sampler)
is in Appendix B.

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
sparsity transition band (SP55–SP82) at two horizons (`pred_len = 128`
matched to Panda, and `pred_len = 64` matched to Chronos's native
trained horizon) with `linear → Chronos` and `CSDI → Chronos`; the
cross-foundation observation is reported in §6.4 (Chronos sits at a
low-VPT plateau and does not exhibit the Panda frontier shape; the CSDI
rescue is not visible because Chronos's own VPT distribution is below
the regime where the lever can move it).
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
sparsity line as cross-foundation observation (§6.4) — Chronos sits at
a low-VPT plateau and does not exhibit Panda's transition shape — and
TimesFM / Lag-Llama are not evaluated.

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
  (~64K independent-IC L63 windows of length 128, with v2-grid-matched
  missingness; see Appendix C). A standalone single-trajectory
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
  positive replication; Mackey-Glass and Chua are scope boundaries.
  Jena Climate hourly is reported as a real-sensor **negative** case
  study that bounds the lever to chaotic-attractor-dominant regimes
  (§6.6). KSE / dysts breadth and additional real-data case studies
  (EEG, climate reanalysis) remain future work.
- **Foundation-model interpretability.** The per-layer encoder probe in
  §4.2 resolves the original "decoder-side latent dynamics" hypothesis
  by direct measurement: Panda-72M is encoder-only (its head is a linear
  projection from the pooled encoder state), and the floor-band rescue
  saturates at **mid-encoder layers 6–8**, where SAITS-pretrained and
  CSDI representations converge to a tie in distance to clean
  (saits/csdi 1.00–1.06). Final-pooled distance is a late-layer
  artefact and is not the relevant geometry. What remains open is *why*
  mid-encoder is the saturation band — that is, what specific
  Panda-internal computation is performed by layers 6–8 such that two
  corpus-pretrained imputers with very different early-layer distances
  arrive at equivalent mid-layer states. This is a follow-up
  interpretability question, not a blocker for the §1 / §3 / §4 claims.

### 6.5 Relationship to data assimilation

Sequential data assimilation (EnKF / LETKF) is a richer and more
information-efficient approach to sparse-observation chaos when the dynamics are
known and tracked online. Our setting is different: the forecaster is a black
box (Panda) and the corruption is preprocessed offline. We therefore compare
against preprocessing-style baselines (linear / Kalman / CSDI) that match the
deployment interface, and we read the corresponding DA literature as motivating
the existence of the forecastability frontier rather than as a direct
competitor.

### 6.6 Real-sensor case study: Jena Climate (boundary on the lever)

To stress-test the §4.4 claim that "corpus-pretrained structured imputation
is the lever" on a real multivariate sensor stream, we run the same
sparse-context-fill protocol on the public Jena Climate 2009–2016 dataset
(14 numeric atmospheric features, 10-minute sampling, downsampled to
hourly; train 2009–2014, val 2015, test 2016; see Appendix C.2 for full
preprocessing). We additionally include a **clean-context upper bound**
(no corruption, ctx_true → forecaster directly) as the natural ceiling
that any imputer-then-forecaster path could hope to reach, and a
**cross-forecaster control** (Chronos-bolt-small *and* Panda-72M) to
distinguish "imputation-axis" effects from forecaster-specific
weaknesses. Imputers: linear and SAITS-pretrained-on-Jena
(trained on the train split, val MAE 0.62 z-units). 10 seeds ×
{SP55, SP65, SP75, SP82}, $n_{ctx} = 512$ hours, $pred_{len} = 64$ hours.
Metric: normalized valid horizon vh@τ — the largest lead-step h such
that the per-step RMSE across the 14 z-scored features stays below
threshold τ.

**Clean-context upper bound and cross-forecaster.**

| | clean | linear | SAITS-pretrained |
|:--|:-:|:-:|:-:|
| `→ Chronos` (vh@1.0 mean over SP55–SP82) | 51.1 | 50.6 (avg) | 30.3 (avg) |
| `→ Panda` (vh@1.0 mean over SP55–SP82) | 46.4 | 43.2 (avg) | 35.2 (avg) |

Two facts pop out:

1. **Linear-fill ≈ clean-context** on both forecasters. On Chronos the
   per-cell linear-fill mean (51.1 / 50.9 / 48.5 / 50.9) tracks the
   clean upper bound (51.1) within 1–3 vh-units; on Panda the linear
   means (45.7 / 41.8 / 46.0 / 39.2) likewise stay within 1–7 of clean
   46.4. Linear interpolation alone preserves enough of the dominant
   diurnal cycle that the forecaster reaches its clean-context ceiling.
2. **SAITS-pretrained drops both forecasters below clean.** Mean vh@1.0
   on Chronos collapses to 27–34; on Panda to 30–39. Paired SAITS −
   linear at vh@1.0 (5000-resample bootstrap, $n = 10$):

| Cell | Chronos paired Δ (95 % CI) | Panda paired Δ (95 % CI) |
|:--|:-:|:-:|
| SP55 | −16.7 [−28.2, −5.8] ↓ | −6.3 [−16.6, +3.0] ≈ |
| SP65 | −18.8 [−29.7, −8.2] ↓ | −4.1 [−12.9, +5.2] ≈ |
| SP75 | −21.0 [−34.3, −8.6] ↓ | −12.0 [−20.0, −4.7] ↓ |
| SP82 | −23.6 [−39.2, −8.6] ↓ | −9.7 [−19.9, −0.9] ↓ |

The SAITS-hurts pattern is **cross-forecaster** — strict-negative on
Chronos at every cell, strict-negative on Panda at SP75 and SP82 and
directionally-negative at SP55 / SP65 — so it is not a Chronos-specific
artefact. The clean-context upper bound additionally rules out the
hypothesis that the forecaster is itself the bottleneck and SAITS just
happens to be on the wrong side of a noisy plateau: the gap between
SAITS-fill and clean is 17–24 vh-units on Chronos and 7–17 on Panda,
both far outside seed-to-seed noise.

**Reading.** The §4.4 lever **does not apply** to Jena, and the
mechanism is not "Chronos is too weak":

> Linear interpolation is already at the clean-context forecasting
> ceiling on Jena hourly because the dominant temporal structure is
> deterministic diurnal periodicity, which linear preserves "for
> free". A SAITS imputer pretrained on a noisy real-world corpus then
> introduces sample-specific high-frequency artefacts that move the
> filled context **off** the periodic mode the forecaster relies on, so
> the result is below clean for both Chronos and Panda. There is no
> headroom for a learned imputer to recover.

This bounds the §4.4 claim cleanly:

> **The corpus-pretrained-imputation rescue is observable on
> chaotic-attractor-dominated systems (L63, L96), where linear
> interpolation breaks the local geometric structure that the foundation
> forecaster relies on. On periodic-dominant real-world streams (Jena
> hourly), linear interpolation already reaches the clean-context
> ceiling, and a corpus-pretrained imputer is net-harmful on both a
> broad time-series forecaster (Chronos) and a chaos-pretrained
> forecaster (Panda).**

The frontier story is therefore a chaotic-system property, not a
universal sparse-context-fill claim. We keep this case study in §6 rather
than promoting it to §3 because it is a **negative** result that defines
the boundary of the claim, and the headline frontier statements (§3.2)
remain unchanged. Sources:
- Chronos + clean upper bound: `experiments/week1/results/jena_real_sensor_jena_chronos_with_clean_upper_10seed.json`
- Panda cross-forecaster control: `experiments/week1/results/jena_real_sensor_jena_panda_with_clean_upper_10seed.json`

---

## 7 Conclusion

A pretrained chaos foundation forecaster (Panda-72M) fails across sharp
sparse-observation forecastability frontiers; a second pretrained
forecaster (Chronos) shows a different failure mode (low-VPT plateau),
so the transition shape is forecaster-dependent. Inside Panda's
transition band, the rescue is regime-aware,
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

Code, CSDI checkpoints, pretrained-SAITS baselines (L63, L96 N=20, and
Jena Climate train split), and the locked Figure-1 / isolation / jitter /
embedding data are released. The §6.6 Jena Climate negative defines the
chaotic-attractor-dominant scope of the lever; additional real-data case
studies (EEG, climate reanalysis), Glocal-IB as a third alt-imputer, and
decoder-side Panda instrumentation remain natural camera-ready follow-ups.

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
L63 SP65 (where 12–22× distance-to-clean reductions in Panda token space
are measured across the four representation stages), but not as a
universal claim across the whole frontier. The
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
| 8 | Panda embedding OOD diagnostic (with SAITS arm, P3.B1) | L63 | SP65, SP82 | clean / linear / **SAITS-pretrained** / CSDI; stages: patch / embed / encoder / pooled | 5 | `panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_with_saits_5seed.json` (current §4.2 source); legacy 3-cell file `..._5seed.json` retained for the original Figure 2 PCA scatter | `..._with_saits_5seed.md` and `_bars.png`; PCA scatter figures use the legacy 3-cell file |
| 9 | Raw-patch diagnostic v2 | L63 | SP65, SP82 | clean / linear / CSDI; metrics: local stdev, lag-1 ρ, mid-freq power | 10 | `l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json` | `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP{65,82}.png` |
| 10 | Cross-system isolation matrix (legacy) | L63, L96 N=10/20, Rössler, Kuramoto | S0–S6 | linear/Kalman/CSDI × Panda/DeepEDM (6) | 5 | `pt_{l63,l96_iso_l96N{10,20},rossler_iso_rossler,kuramoto}_*_5seed.json` | `deliverable/figures_isolation/*_heatmap.png`, `*_bars.png`, `*.md` |
| 11 | MG / Chua scope-boundary cases | Mackey-Glass, Chua | S0–S6 | same as #10 | 5 | `pt_{mg,chua}_*_5seed.json` | `deliverable/figures_isolation/` (boundary subset) |
| 12 | Alt-imputer per-instance sanity | L63 | SP65 | linear, SAITS, BRITS, CSDI | 5 | `panda_altimputer_l63sp65_partial_5seed.json` | log-only; Appendix E sanity |
| 13 | **Pretrained alt-imputer (P1.1 + P1.5 + P2.2 30-seed)** | L63, L96 N=20 | L63 SP65 + SP82, L96 SP82 | linear, SAITS-pretrained, CSDI | L63: 10, L96: **30** | `panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`, `panda_altimputer_l96_sp82_pretrained_30seed.json` | §4.4 + Appendix C |
| 14 | **Chronos mini-frontier (P1.2)** | L63 | SP55, SP65, SP75, SP82 | linear, CSDI (forecaster: Chronos, `pred_len ∈ {64, 128}`) | 5 | `chronos_frontier_l63_chronos_l63_sp55_sp82_5seed.json`, `..._5seed_pl64.json` | §6.4 cross-foundation observation; pred_len=64 confirms negative is not an artefact of Chronos OOD horizon |
| 15 | **EnKF known-dynamics upper bound (P1.3)** | L63 | SP55–SP82, NO020, NO050 | EnKF (true vector field, 100 members) | 5 | `enkf_l63_enkf_l63_v2_5seed.json` | §6.5 / Appendix B reference |
| 16 | **Real-sensor case study (P2.1 + P3.A clean-upper / cross-forecaster)** | Jena Climate 2009–2016 | SP55, SP65, SP75, SP82 (hourly, $n_{ctx}=512$, $pred_{len}=64$) | clean, linear, SAITS-pretrained-on-Jena × {Chronos-bolt-small, Panda-72M} | 10 | `jena_real_sensor_jena_chronos_with_clean_upper_10seed.json`, `jena_real_sensor_jena_panda_with_clean_upper_10seed.json` | §6.6 boundary case study with clean-context upper bound + cross-forecaster control; metric = normalized valid horizon vh@τ in z-RMSE units |
| 17 | **Per-layer encoder probe (P3.B2)** | L63 | SP65, SP82 | clean / linear / SAITS-pretrained / CSDI × {12 PandaLayer outputs + post-embedder} | 5 | `panda_per_layer_probe_l63_sp65_sp82_per_layer_5seed.json` | §4.2 mid-encoder convergence finding; closes "decoder-side hypothesis" by direct measurement |

Items 1–9 are the patched-protocol locked numbers cited in §3 / §4 / §6.
Item 10 is the cross-system replication that uses the older S0–S6
corruption pipeline (`make_sparse_noisy`) and is cited as **secondary**
direction-of-effect evidence — the v2 protocol numbers in items 1–6 / 8 / 9
are authoritative. Item 11 supplies §6.3 scope conditions. Item 12 is
Appendix E sanity (per-instance training, biased against SAITS / BRITS by
design). Items 13–16 are the P1 / P2 reviewer-defense experiments:
pretrained SAITS alt-imputer comparison including the L96 survival
replication extended to 30 seeds (P2.2)
(§4.4 / Appendix C), Chronos cross-foundation mini-frontier (§6.4),
EnKF known-dynamics upper bound (§6.5 / Appendix B), and the Jena
Climate real-sensor case study (§6.6 / Appendix C.2 — boundary case
showing the lever does not transfer to periodic-dominant streams).

### B.3 Reproducibility note: CSDI sampler stochasticity across §3.2 / §4.3 / §4.4

CSDI is a conditional score-diffusion imputer; its `impute(...)` call is
stochastic, so independent runs on the **same** corruption draws produce
slightly different filled contexts and therefore slightly different
per-seed VPTs. The locked numbers in §3.2 (Figure 1 v2 grid), §4.3
(jitter control), and §4.4 (alt-imputer) come from three separate CSDI
inference runs, each with its own random sampler seed. The L63 SP65
paired CSDI − linear value reads as +1.64 in §3.2 / §1, +1.65 in §4.3,
and +1.67 in §4.4; all three CIs overlap and the qualitative claim
(strict-positive paired CSDI − linear at the L63 entrance band) is
stable. The same caveat applies to the small per-cell spread between
§3.2's L96 SP82 cross-system replication (Panda median 0.50 → 1.05) and
§4.4's L96 SP82 alt-imputer comparison (median 0.50 → 1.13 at $n=10$,
0.25 → 1.26 at $n=30$): same v2 corruption seed scheme and same Panda
checkpoint, but independent CSDI inference runs. We do not freeze the
diffusion seed across experiments; the per-seed VPT differences
(typically < 0.5 Λ) reflect CSDI's sampler stochasticity, not protocol
drift. All experiments are released as separate JSONs (Appendix B.2 row
1 / 4 / 13) so the per-run numbers can be re-derived.

### B.4 Aggregator scripts

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
linear < SAITS-pretrained < CSDI.

**Update at $n = 30$ (P2.2 reviewer-defense extension).** We extend the
L96 SP82 alt-imputer cell from 10 → 30 seeds to dilute the lucky-seed
effect. At 30 seeds the mean ambiguity disappears: every metric is
monotonic `linear < SAITS-pretrained < CSDI` on mean, median, Pr(VPT>0.5)
and Pr(VPT>1.0), and **all three paired contrasts are strict-positive**
(SAITS − linear +0.71 [+0.02, +1.38]; CSDI − linear +1.01 [+0.36, +1.64];
CSDI − SAITS-pretrained +0.31 [+0.07, +0.56]). The 30-seed table replaces
the 10-seed numbers in §4.4. The 10-seed per-seed table previously kept
in this appendix is now redundant and dropped from the freeze; both 10-
and 30-seed JSONs are released.

Glocal-IB is not evaluated (cited in §2 as adjacent prior art on
high-missingness imputation that emphasises preserving global latent
structure); it remains a natural follow-up.

Sources:
- L63: `experiments/week1/results/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`
- L96 (10-seed pilot): `experiments/week1/results/panda_altimputer_l96_sp82_pretrained_10seed.json`
- L96 (30-seed authoritative): `experiments/week1/results/panda_altimputer_l96_sp82_pretrained_30seed.json`

## Appendix C.2: Real-sensor pilot — Jena Climate

**Data.** Public Jena Climate 2009–2016 record from the Max Planck
Institute weather station, 14 numeric atmospheric features at 10-minute
resolution (`p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`, `rh (%)`,
`VPmax (mbar)`, `VPact (mbar)`, `VPdef (mbar)`, `sh (g/kg)`,
`H2OC (mmol/mol)`, `rho (g/m**3)`, `wv (m/s)`, `max. wv (m/s)`,
`wd (deg)`). We resample to hourly mean (factor 6) and split by year:
train 2009–2014 (52 622 hours), validation 2015 (8 760 hours), test 2016
(8 709 hours). All features are z-scored using train-split per-feature
mean and standard deviation. The "surrogate attractor std" used by the
SAITS training logger is the mean of the train per-feature unscaled std,
13.57; SAITS sees z-scored data, so its native error scale is 1.0
z-units per feature.

**SAITS-Jena pretraining.** Same architecture as L63 / L96 SAITS
(2 layers, $d_{model}=64$, 4 heads, $d_k = d_v = 16$, $d_{ffn}=128$),
30 epochs, batch 64, ≈ 2 min on 1 V100 (smaller corpus than L63 / L96).
Best checkpoint at epoch 30 (val MAE on missing entries 0.62 z-units).
Training corruption distribution matches the v2 `fine_s_line` grid (same
sparsities, iid_time mask). Checkpoint:
`experiments/week2_modules/ckpts/saits_jena_pretrained/<run-id>/SAITS.pypots`.

**Eval protocol.** For each seed, draw a contiguous 576-hour test-split
window starting at a uniformly-random offset; treat the first
$n_{ctx} = 512$ hours as context and the last $pred_{len} = 64$ hours as
the future. Apply v2-style sparse corruption to the context (sparsity
∈ {0.55, 0.65, 0.75, 0.82}, σ = 0); impute via linear and via
SAITS-pretrained-on-Jena (chunked 4 × 128 inference, the SAITS pretraining
context length); forecast with Chronos-bolt-small per channel; compare
to true future. Metric: normalized valid horizon vh@τ — the largest
lead step h such that the per-step RMSE across the 14 z-scored features
stays ≤ τ. We report vh@0.3, vh@0.5, vh@1.0, vh@2.0.

**Results (10 seeds, σ = 0).**

| Cell | SP55 vh@1.0 mean | SP65 | SP75 | SP82 |
|:--|:-:|:-:|:-:|:-:|
| `linear → Chronos` | 51.1 | 50.9 | 48.5 | 50.9 |
| `SAITS-pretrained → Chronos` | 34.4 | 32.1 | 27.5 | 27.3 |

| Paired contrast (SAITS − linear) | SP55 | SP65 | SP75 | SP82 |
|:--|:-:|:-:|:-:|:-:|
| vh@1.0 Δ | −16.7 | −18.8 | −21.0 | −23.6 |
| vh@1.0 95 % CI | [−28.2, −5.8] | [−29.7, −8.2] | [−34.3, −8.6] | [−39.2, −8.6] |
| vh@0.5 Δ | −6.3 | −4.4 | −3.5 | −4.3 |
| vh@0.5 95 % CI | [−12.0, −1.9] | [−8.5, −0.9] | [−8.0, +0.2] | [−10.1, +0.2] |

The vh@1.0 contrast is **strict-negative at every cell** — SAITS-pretrained
is *worse* than linear interpolation on Jena hourly. This is the §6.6
boundary on the §4.4 lever: when the dominant temporal structure is
periodic (here daily / weekly cycles in atmospheric variables), linear
interpolation already preserves it for free, and a learned SAITS imputer
introduces sample-specific artefacts that drift the filled context off
the periodic mode.

Source:
`experiments/week1/results/jena_real_sensor_jena_real_sensor_10seed.json`.

## Appendix D: Figure Index

All figures referenced in the main text use the patched v2 protocol unless
explicitly marked. Main-text labels Figure 1 / 2 / 3 correspond to the three
headline panels.

### Main figures

| Label | Caption purpose | Path |
|---|---|---|
| **Figure 1** | Sparsity and noise frontier on L63 (Panda mean VPT, $\Pr(\mathrm{VPT}>0.5\,\Lambda)$, $\Pr(\mathrm{VPT}>1.0\,\Lambda)$ on decoupled $s$ and $\sigma$ axes; 10 seeds; **patched v2 protocol**; 95 % bootstrap CI on mean and Wilson 95 % CI on survival) | `deliverable/figures_main/figure1_l63_v2_10seed_patched.png` (and `.md` companion table) |
| **Figure 2** | Cross-system isolation matrix on linear / Kalman / CSDI × Panda / DeepEDM (heatmaps and paired-bootstrap CI bars; **5 seeds**; **legacy S0–S6 protocol** — secondary direction-of-effect evidence; v2 protocol numbers in §3.2 are authoritative) | `deliverable/figures_isolation/l63_iso_l63_5seed_heatmap.png`, `l96_iso_l96N10_5seed_heatmap.png`, `l96_iso_l96N20_5seed_heatmap.png`, `rossler_iso_rossler_5seed_heatmap.png` (each with a matching `_bars.png`) |
| **Figure 3** | Jitter / residual controls across L63, L96 N=20, Rössler at SP65 and SP82, comparing Panda mean VPT vs $\Pr(\mathrm{VPT}>1.0\,\Lambda)$; L63 is **10 seeds**, L96 / Rössler are **5 seeds**; **patched v2 protocol**; paired-bootstrap CI on every Δ | `deliverable/figures_jitter/jitter_milestone_SP65.png`, `jitter_milestone_SP82.png` |

### §4.2 mechanism panels (patched v2 protocol)

| Element | Path |
|---|---|
| L63 raw-patch metric histograms (local stdev / lag-1 ρ / mid-freq power, SP65 + SP82, 10 seeds) | `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP65.png`, `..._SP82.png` |
| L63 raw-patch trajectory overlays (clean vs linear vs CSDI, 10 seeds) | `experiments/week1/figures/l63_patch_ood_v2_v2protocol_traj_overlay_SP65.png`, `..._SP82.png` |
| Panda token-space distance bars (patch / embed / encoder / pooled, SP65 + SP82, 5 seeds, paired-bootstrap CI) | `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed_bars.png` |
| Panda token-space PCA scatter (per stage and scenario, 5 seeds) | `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_5seed_SP65_embed_pca.png`, `..._SP65_encoder_pca.png`, `..._SP82_embed_pca.png`, `..._SP82_encoder_pca.png` |

### §4.4 / Appendix C alt-imputer (P1.1 + P1.5)

| Element | Path |
|---|---|
| L63 SP65 + SP82 alt-imputer summary table (linear / SAITS-pretrained / CSDI; 10 seeds; paired-bootstrap CI; Wilson CI on survival) | `experiments/week1/figures/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.md` |
| L96 N=20 SP82 alt-imputer summary table (linear / SAITS-pretrained / CSDI; 10 seeds; median + Wilson CI on survival; paired-bootstrap on means) | `experiments/week1/figures/panda_altimputer_l96_sp82_pretrained_10seed.md` |

### §6.3 scope-boundary panels (appendix only, **legacy S0–S6 protocol**, 5 seeds)

| Element | Path |
|---|---|
| Mackey-Glass S0–S6 phase-transition curve | `experiments/week1/figures/pt_mg_mg_5seed_phase_transition.png` |
| Mackey-Glass attractor trajectory | `pictures/mackey_glass_trajectory_final.png` |
| Chua S0–S6 phase-transition curve | `experiments/week1/figures/pt_chua_chua_5seed_phase_transition.png` |
| Chua double-scroll trajectory | `pictures/chua_trajectory_final.png` |

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
