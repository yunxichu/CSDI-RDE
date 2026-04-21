"""Quick Chronos zero-shot smoke test on a Lorenz63 slice.

Downloads amazon/chronos-t5-small (~50MB) on first run; cached afterwards.
Run:
  CUDA_VISIBLE_DEVICES=2 python experiments/week1/smoke_chronos.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def main() -> None:
    from dysts.flows import Lorenz
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj = np.asarray(Lorenz().make_trajectory(600, resample=True))
    ctx = traj[:500, 0]
    true_future = traj[500:, 0]
    print(f"[data] ctx.shape={ctx.shape} future.shape={true_future.shape}")

    try:
        from chronos import ChronosPipeline  # type: ignore

        src = "chronos"
    except ImportError:
        src = None

    if src == "chronos":
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map=device, torch_dtype=torch.float32)
        t0 = time.time()
        forecast = pipe.predict(torch.tensor(ctx, dtype=torch.float32).unsqueeze(0), prediction_length=100, num_samples=20)
        t1 = time.time() - t0
        fc = forecast[0].cpu().numpy()  # (num_samples, prediction_length)
        mean = fc.mean(0)
        print(f"[chronos] via ChronosPipeline inference {t1:.2f}s mean.shape={mean.shape}")
    else:
        from transformers import pipeline

        t0 = time.time()
        pipe = pipeline("text-generation", model="amazon/chronos-t5-small", device=device)
        t1 = time.time() - t0
        print(f"[chronos] fallback transformers pipeline not ideal. load={t1:.2f}s -- Chronos needs chronos-forecasting pkg")

    rmse = float(np.sqrt(((mean - true_future) ** 2).mean())) if src == "chronos" else float("nan")
    print(f"[chronos] zero-shot rmse on 100-step Lorenz63 x-component: {rmse:.3f}")


if __name__ == "__main__":
    main()
