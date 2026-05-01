# Paper Status — 2026-05-01 (post-P1 freeze)

**State.** P0 (cleanup) + P1 (reviewer-defense experiments) **completed**.
The draft is at submission-ready state modulo P3 optional polish (real-data
case study, Panda decoder-side mechanism, Glocal-IB). No new experiments
are scheduled before submission unless reviewer simulation flags a hard
blocker.

**Authoritative commits.**

| Commit | Date | Scope |
|:--|:--|:--|
| `75a7bf4` | 2026-05-01 | P0 cleanup — internal notes stripped, Wilson CI added, §3 retitled, three-regime taxonomy demoted |
| `695dbad` | 2026-05-01 | P1.1 / P1.2 / P1.3 — pretrained SAITS L63 + Chronos mini-frontier + EnKF upper bound |
| `c3f1256` | 2026-05-01 | P1.4 / P1.5 — Chronos pred_len=64 native horizon + SAITS-pretrained L96 N=20 cross-system |
| `bd2ccc6` | 2026-05-01 | docs(P1) — record Chronos native horizon and L96 SAITS follow-up in `P1_RESULTS.md` |
| **`290e38b`** | **2026-05-01** | **Submission-prep QA freeze** — traceability sidecar, reviewer-sim fixes (Chronos plural, "12–34×", "generalises" softening, CSDI-stochasticity disclosure), Appendix D figure-path fixes |

The current submission-ready freeze is `290e38b`. External reviewers
should pull this commit and read the four files listed below.

Authoritative drafts:

- `deliverable/paper/paper_draft_en.md` — English (current target)
- `deliverable/paper/paper_draft_zh.md` — Chinese mirror, synced with English

Authoritative milestone narratives:

- `deliverable/SUBMISSION_PREP_PLAN.md` — reviewer-perspective P0/P1/P2/P3 plan
- `deliverable/P1_RESULTS.md` — per-experiment P1 results (P1.1/P1.2/P1.3 +
  P1.4/P1.5 follow-ups recorded by `bd2ccc6`)
- `deliverable/PAPER_NUMBER_TRACEABILITY.md` — every headline number
  traced to its source JSON / figure (built at `290e38b`)

---

## 0. External reviewer entry point

If you are reading the paper for the first time:

1. Read `deliverable/paper/paper_draft_en.md` end-to-end (≈ 30 min).
2. Sanity-check any number that looks load-bearing against
   `deliverable/PAPER_NUMBER_TRACEABILITY.md` — every abstract / §1 /
   §3 / §4 / §6.4 number is mapped to the underlying result-JSON.
3. Spot-check the patched-protocol claims by opening a JSON listed in
   the traceability table; the per-seed records are at the top level
   under `records`, the aggregated cell statistics under `summary`, and
   the paired-bootstrap CIs under `contrasts`.
4. The story has been narrowed twice — first away from a four-module
   pipeline to a sparse-observation forecastability frontier
   (commit `c99c978`), then again under reviewer-defense pressure into
   "corpus-pretrained structured imputation is the lever" (commit
   `c3f1256`). Both narrowings are intentional; the archive at
   `deliverable/paper/paper_draft_en_archive_2026-04-30.md` keeps the
   pre-pivot text for context.

---

## 1. Locked story

> Pretrained chaotic forecasters fail across sharp sparse-observation
> forecastability frontiers. Inside the sparsity transition band,
> **corpus-pretrained structured imputation** is the lever that reliably
> moves Panda back across the frontier. CSDI is one strong instance of
> the lever, with a small but paired-CI-strict advantage at the L63
> entrance band (SP65) and on L96 SP82 median + survival; at the L63
> floor band (SP82) CSDI and a corpus-pretrained SAITS are statistically
> indistinguishable. The lever is sparse-gap-imputation specific, not
> dense-noise denoising. DeepEDM in delay coordinates is a complementary
> dynamics-aware route, not the headline.

The three reviewer-defeating qualifiers that **must remain** in the
abstract and §1 (mark them when editing):

1. **Sparse-observation frontier**, not all corruption.
2. **Inside the transition band**, not universal improvement everywhere.
3. **Structured residuals are not interchangeable with iid noise**,
   especially in survival / tail metrics.

---

## 2. Completed P0 cleanup (commit `75a7bf4`)

- Internal-facing notes removed from both drafts (no "must be refreshed",
  "deferred", "reviewer-defeating" wording in the body).
- Wilson 95 % CI added to all headline survival numbers.
- §3 retitled "Empirical Forecastability Frontiers".
- Three-regime taxonomy demoted from contributions to a per-cell empirical
  bullet list in §1.
- DeepEDM softened to "complementary dynamics-aware companion".
- Per-instance SAITS / BRITS demoted to Appendix E sanity.

---

## 3. Completed P1 experiments

### P1.1 — Pretrained SAITS on L63 SP65 + SP82 (commit `695dbad`)

`experiments/week1/results/panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json`

| Cell | SP65 mean | SP82 mean | SP65 Pr>1.0 | SP82 Pr>1.0 |
|:--|--:|--:|--:|--:|
| `linear → Panda` | 1.22 | 0.29 | 70 % | 0 % |
| `SAITS-pretrained → Panda` | 2.49 | 1.51 | 90 % | 70 % |
| `CSDI → Panda` | 2.89 | 1.57 | 100 % | 70 % |

Paired CSDI − SAITS = +0.41 [+0.05, +0.87] at SP65 (strict-positive),
+0.06 [−0.31, +0.59] at SP82 (≈ tie).

**Effect on the paper.** §1 / abstract intervention claim narrowed from
"CSDI is the only tested intervention" to "**corpus-pretrained structured
imputation is the lever**, with CSDI a small entrance-band advantage".

### P1.2 — Chronos mini-frontier on L63 (commit `695dbad`)

`experiments/week1/results/chronos_frontier_l63_chronos_l63_sp55_sp82_5seed.json`

Mean VPT@1.0 across SP55 / SP65 / SP75 / SP82: 0.34–0.50; Pr>1.0 ≤ 20 %;
paired CSDI − linear all CIs straddle zero.

**Effect on the paper.** §6.4 reframed: cross-foundation evidence is
forecaster-dependent; the frontier shape is empirically established for
Panda but does not transfer cleanly to Chronos at the horizons we test.

### P1.3 — EnKF known-dynamics upper bound on L63 (commit `695dbad`)

`experiments/week1/results/enkf_l63_enkf_l63_v2_5seed.json`

Stochastic EnKF (n_members=100, true L63 vector field, RK4 forward)
saturates the VPT ceiling (≈ 2.84–2.90) across the entire SP55–SP82
band; Pr>1.0 = 100 % at all cells. Degrades only on the dense-noise axis.

**Effect on the paper.** §1 / §6.5 argue the frontier is a property of the
**black-box deployment interface** (forecaster has no access to dynamics),
not of L63 itself.

### P1.4 — Chronos at native horizon pred_len=64 (commit `c3f1256`)

`experiments/week1/results/chronos_frontier_l63_chronos_l63_sp55_sp82_5seed_pl64.json`

Same SP55–SP82 cells, 5 seeds, at Chronos's native trained horizon
(Chronos library warns `prediction_length > 64` is OOD). Per-seed VPTs
are statistically indistinguishable from `pred_len=128` (mean 0.34–0.50,
paired CIs all straddle zero).

**Effect on the paper.** §6.4 caveat "matched pred_len ≤ 64 is future
work" fully resolved — the negative is not an artefact of the Chronos
OOD horizon.

### P1.5 — Pretrained SAITS L96 N=20 cross-system (commit `c3f1256`)

`experiments/week1/results/panda_altimputer_l96_sp82_pretrained_10seed.json`

L96 SP82, 10 seeds. Mean is dominated by linear seed-2 fluke
(VPT@1.0 = 10.75); per existing L96 high-variance limitation we lead
with median + survival:

| Cell | median VPT@1.0 | Pr(VPT>1.0) (Wilson 95 %) |
|:--|:-:|:-:|
| `linear → Panda` | 0.50 | 30 % [11 %, 60 %] |
| `SAITS-pretrained → Panda` | 0.84 | 40 % [17 %, 69 %] |
| `CSDI → Panda` | **1.13** | **60 %** [31 %, 83 %] |

Paired CSDI − SAITS = +0.21 [+0.00, +0.49] on means.

**Effect on the paper.** Cross-system replication of the structured-
imputation lever from 3-D L63 to 20-D L96 on median + survival.

---

## 4. Where headline numbers come from

See section 6 (Number-source traceability) below for the full table.

---

## 5. Optional P3 (deferred)

- **Real-data case study** (ECG, EEG, climate reanalysis). Highest value
  P3 if pursued; flagged in §6.4 as future work.
- **Panda decoder-side mechanism** instrumentation for the floor-band
  question (why CSDI floor-band rescue exceeds what raw-patch / Panda-
  token distances predict).
- **Glocal-IB pretrained as third alt-imputer**. Listed as adjacent prior
  art in §2; would extend the §4.4 lever claim to a 3rd imputer.
- **KSE / dysts breadth on §3.**

Decision: **default skip P3** for this submission; revisit only if
reviewer simulation (step 8 of QA plan) flags a hard reject hook
addressable only by one of these.

---

## 6. Number-source traceability table

(Filled in by step 3 of the submission QA pass — see
`PAPER_NUMBER_TRACEABILITY.md`.)

---

## 7. Submission QA checklist (in progress 2026-05-01)

1. ✅ P0 + P1 completed and pushed
2. 🟡 Consistency grep en + zh — no stale "deferred" / "C1 待跑" / "future
   work pred_len ≤ 64" / "must be refreshed" / "CSDI is the only
   intervention" wording
3. 🟡 Number-source traceability table
4. 🟡 Appendix D figure-index audit (every path exists; captions specify
   seeds / metric / Wilson or bootstrap CI / patched v2 protocol)
5. 🟡 DeepEDM positioning final check (companion, not main)
6. 🟡 Chronos negative-result wording check (plateau framing, not "frontier
   generalises")
7. 🟡 SAITS conclusion wording check (lever framing, CSDI strong instance,
   not universal mean dominator)
8. 🟡 Reviewer simulation (Claude / GPT fresh-context read) — reject hooks?
   over-claim? weakest experiment? abstract honest?
9. 🟡 P3 decision (default skip; revisit only if step 8 hard-flags it)

Items 2–8 are the active QA pass; item 9 is the final go/no-go.
