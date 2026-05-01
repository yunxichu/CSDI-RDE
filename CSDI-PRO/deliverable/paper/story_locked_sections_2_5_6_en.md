# Story-Locked Draft Sections: §2, §5, §6

> Drop-in companion to `story_locked_sections_1_3_4_en.md`. Same lock rules:
> all three abstract qualifiers must remain — sparse-observation frontier,
> inside the transition band, structured residuals not fully interchangeable.
> The earlier paper-draft §2 / §5 / §6 (which still describe a 4-module
> pipeline and a tokenizer-OOD mechanism) are superseded by these.
>
> **Patched-refresh warning (2026-05-01).** This file preserves the April 30
> support-section skeleton, but CSDI-dependent wording in §6 has been
> superseded by the current `deliverable/paper/paper_draft_en.md`. Use the main
> draft as authoritative.

---

## 2 Related Work

**Pretrained chaotic / time-series forecasters.** Chronos [Ansari24],
TimesFM [Das23], Lag-Llama [Rasul23], TimeGPT [Garza23], and the
chaos-specific Panda-72M [Wang25] pretrain decoder Transformers on large
time-series corpora and are evaluated primarily on dense, clean context
windows. Our work asks what happens at the realistic interface where the
context is sparse and noisy and is filled before forecasting. We pick
Panda-72M as the headline because it is already specialized for chaotic
dynamics and therefore the strongest available candidate for "the
forecaster does not need a corruption-aware front-end".

**Time-series imputation under missingness.** BRITS [Cao18] frames
imputation as a bidirectional recurrent process; SAITS [Du22] uses
diagonally-masked self-attention; CSDI [Tashiro21] introduces score-based
diffusion imputation conditioned on observed points; recent work such as
Glocal-IB argues that high-missingness imputation must preserve a global
latent structure and not just minimize pointwise reconstruction error.
We extend this conversation downstream: in our chaotic-forecasting setting,
even imputations that are *closer* to the clean trajectory in raw or
foundation-model token space do not always forecast better, and structured
residuals are not interchangeable with iid noise of matched magnitude.
That is, the relevant objective is not imputation fidelity but whether
the reconstructed context lies on the forecastable side of a sharp
forecastability frontier.

**Delay-coordinate forecasting.** Classical Takens-style delay-embedding
plus local linear or kernel prediction goes back to
[Farmer-Sidorowich87, Casdagli89]. Echo-state networks [Jaeger01,
Pathak18], reservoir computing, and operator-theoretic approaches
[Brunton16, Lu21] also provide forecasters whose state is constructed
from delay or projected coordinates. DeepEDM / LETS-Forecast [Majeedi25]
recasts delay-coordinate prediction as softmax-attention-as-learned-kernel
on Takens tokens; we use it as a complementary dynamics-structured route
that does not depend on a foundation model's tokenizer.

**Phase transitions and survival probabilities in forecasting.** Although
the words "phase transition" appear informally in chaos literature, our
operational claim — that survival probability $\Pr(\mathrm{VPT}>\theta\Lambda)$
collapses non-smoothly across a narrow band of sparsity — does not assume
a thermodynamic critical exponent. We follow numerical-weather-forecasting
practice [Bauer15] in treating tail-survival as the operational quantity,
and use the language *sharp forecastability frontier* rather than *phase
transition* to keep the empirical claim distinct from a physics analogy.

**Data assimilation as classical sparse-noisy baseline.** EnKF / LETKF
[Hunt07] is the standard baseline for sparse-noisy chaos but is typically
deployed jointly with the model dynamics, not as a preprocessing step
feeding a black-box forecaster. We do not include it as a primary
baseline because the comparison would conflate state-estimation quality
with the forecastability question we isolate; we discuss the relationship
in §6.

---

## 5 Method

We deliberately keep this section short. The main contributions of the
paper are the failure law, the intervention law, and the
reconstruction-forecastability mismatch, not a new modular pipeline.

### 5.1 Corruption-aware imputation (M1)

We use a CSDI-style score-based diffusion imputer [Tashiro21] to fill the
sparse-noisy context. A clean attractor scale $\sigma_\mathrm{attr}$
(per-axis-mean over a long reference trajectory) is used to normalize
inputs and the diffusion noise schedule; mismatched normalization leads to
under- or over-noised samples and was the origin of an earlier protocol
inconsistency that we report transparently in our reproducibility appendix.
Only one imputation is required per context; we use the diffusion median
across a small sample budget for the deterministic VPT panels and the full
sample distribution for tail survival.

### 5.2 Delay-manifold forecaster (DeepEDM) as companion

Our companion forecaster predicts the next state from a fixed-length
delay vector
$X_t = [x_t, x_{t-\tau_1}, \dots, x_{t-\tau_L}]$
using a softmax-attention learned-kernel head [Majeedi25] trained on
delay/next-state pairs derived from the imputed context. Lags $\{\tau_i\}$
are selected by a mutual-information / Lyapunov objective (an MI-Lyap
schedule we describe in Appendix A) to balance injectivity and stretch
rate. DeepEDM is included primarily because it does not depend on a
foundation-model tokenizer; this is the route the §4 isolation matrix
shows can also be improved by corruption-aware imputation.

### 5.3 Forecasters and forecasters under test

The forecaster under test in the main figures is Panda-72M [Wang25]. The
isolation matrix in §4 spans $\{$linear, AR-Kalman, CSDI$\}$ imputers
times $\{$Panda, DeepEDM$\}$ forecasters. The alt-imputer reviewer-defense
table in §4 adds SAITS and BRITS at the strongest CSDI-decisive cells;
neither is pretrained on the chaos corpus, so the comparison is biased
against them — we keep this caveat in §6.

### 5.4 Operational metrics

We report three metrics throughout the main paper:

- mean valid-prediction time (VPT) in Lyapunov-time units;
- survival probability $\Pr(\mathrm{VPT} > 0.5\,\Lambda)$ and
  $\Pr(\mathrm{VPT} > 1.0\,\Lambda)$, the latter being the
  operational tail metric;
- paired-bootstrap mean differences between matched cells of the
  isolation matrix.

CIs are 95 % bootstrap on the mean and Wilson 95 % on survival
probabilities.

---

## 6 Discussion and Limitations

### 6.1 What the paper claims, restated

Inside the sparse-observation transition band, corruption-aware imputation
reliably moves Panda back across the forecastability frontier. In the
entrance band we can attribute this to a large reduction in raw-patch and
Panda-token distance to the clean context; near the frontier floor those
distances become mixed and the residual survival gain is no longer
explained by distance-to-clean alone. CSDI is therefore a sparse-gap
imputation lever, not a generic dense-noise denoiser. Delay-manifold
forecasting is a complementary dynamics-aware route through the same
frontier.

### 6.2 What the paper does not claim

We do not claim that pretrained chaotic forecasters are intrinsically
broken; the rescue results show they are highly recoverable when the
filled context is corruption-aware. We do not claim CSDI is the only
imputer that works — only the only *tested* intervention that reliably
rescues across the band; alt-imputer results in §4 set the current
boundary. We do not claim that the mechanism is fully characterised;
distance-to-clean explains the entrance band but not the floor band.
We do not claim universality across all foundation forecasters;
Panda-72M is the headline and Chronos / context-parroting are present
only as side baselines.

### 6.3 Scope conditions

The delay-manifold companion assumes a smooth attractor and a useful
finite-dimensional Takens representation. Mackey-Glass and Chua are
reported as appendix scope boundaries. Mackey-Glass is a scalar
delay-differential system whose effective state is infinite-dimensional
under the observation window; the available CSDI training corpus and
delay configuration do not span the relevant history dimension. Chua is
a piecewise-linear, non-smooth circuit; the smooth-attractor
assumptions implicit in M1 / M3 are violated. These are honest
boundaries, not hidden failures.

### 6.4 Limitations

- **Corpus asymmetry.** Our CSDI is pretrained on each system separately
  (∼500K independent-IC windows). SAITS and BRITS in our alt-imputer
  experiment are trained per-instance from a single 512-step trajectory.
  A pretrained SAITS / BRITS comparison is the natural follow-up.
- **Single-forecaster headline.** Panda is the only foundation
  forecaster fully evaluated under the v2 corruption grid. Replicating
  the frontier curve with TimesFM / Chronos / Lag-Llama would extend the
  external validity of Pillar 1.
- **Pure-noise axis.** The paper's intervention claim is restricted to
  the sparse-observation axis. CSDI is neutral or slightly hurtful on
  the dense-noise axis; a denoising-aware variant is an open follow-up.
- **System breadth.** L63, L96 N=10/20, Rössler, and Kuramoto cover the
  positive replication; Mackey-Glass and Chua are scope boundaries. KSE
  / dysts breadth and real-data case studies (EEG, climate reanalysis)
  remain future work.
- **Foundation-model interpretability.** Why CSDI's residual produces a
  forecastable context near the floor band, when raw-patch and
  Panda-token distance to clean no longer separate it from linear, is
  open. A natural hypothesis is that the relevant geometric quantity
  lives in Panda's deeper latent dynamics (decoder side rather than
  encoder side), which we have not instrumented.

### 6.5 Relationship to data assimilation

Sequential data assimilation (EnKF / LETKF) is a richer and more
information-efficient approach to sparse-noisy chaos when the dynamics
are known and tracked online. Our setting is different: the forecaster is
a black box (Panda) and the corruption is preprocessed offline. We
therefore compare against preprocessing-style baselines (linear / Kalman
/ CSDI) that match the deployment interface, and we read the
corresponding DA literature as motivating the existence of the
forecastability frontier rather than as a direct competitor.
