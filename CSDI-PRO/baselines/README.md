# Foundation-model baselines

Weights + adapters for head-to-head comparisons against our v2 pipeline.

## panda-72M (blocked, needs upstream source)

Panda (Lai / Bao / Gilpin, ICLR 2026) is the reference foundation model for chaotic dynamics. We
downloaded the 72M-parameter checkpoint to `panda-72M/` (weights ignored via `.gitignore`) and
hand-reimplemented the arch in `panda_model.py` by reading the safetensors key layout:

  - `DynamicsEmbedder` (freq_weights [16,188] + freq_biases [1,1,1,188] + projection [768,768])
  - Dual-attention layer: `temporal_self_attn` (RoPE 75%, `max_wavelength=500`) + `channel_self_attn`
  - RMSNorm × 3 per block (pre-norm), FFN with `ffn_dim=d_model=768`
  - Prediction head: mean-pool over patches + linear [128, 768]

**Status: functionally incorrect**. The reimplementation loads all 280 weights without
`missing`/`unexpected` errors and param count matches 71.6 M, but zero-shot NRMSE at h=1 on clean
Lorenz63 is ~0.80 (should be ≪ 0.1). Predictions collapse to a per-instance mean — layer-0
activations blow up 3000× through attention, which suggests a custom normalisation/scaling in
Panda's attention that is not visible from weights alone (e.g. QK-norm, residual scaling, or a
different embedder concat order).

**Unblock**: we need the Panda source (`github.com/abao1999/panda`, not accessible from the
sandbox). Once cloned next to this repo, the right fix is to `from panda.models.patchtst import
PatchTSTForPrediction` and load the HF weights via `transformers.AutoModel.from_pretrained`. Our
adapter will then just wrap `model.generate()`.

```bash
# Run this OUTSIDE Claude's sandbox:
cd /home/rhl/Github && git clone https://github.com/abao1999/panda panda-src
```

Until then, the `panda_model.py` file is a placeholder that documents the arch layout — do not
use it for paper numbers.
