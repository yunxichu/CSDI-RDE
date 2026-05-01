"""Pretrain a SAITS imputer on the L63 chaos corpus, with a missingness
distribution matched to the v2 corruption grid.

Output a PyPOTS-format SAITS checkpoint that the alt-imputer evaluator
loads at inference time.

Run:
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  python -u -m experiments.week2_modules.train_saits_l63 \
      --corpus experiments/week2_modules/data/lorenz63_clean_64k_L128.npz \
      --epochs 30 --batch 64 \
      --out experiments/week2_modules/ckpts/saits_l63_pretrained
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]


# fine_s_line sparsity values (mirrors corruption_grid_v2.json).
SPARSITY_GRID = (0.00, 0.20, 0.40, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97)


def make_training_masks(n: int, seq_len: int, n_features: int,
                         seed: int = 0) -> np.ndarray:
    """For each of n windows, pick a sparsity uniformly from SPARSITY_GRID
    and apply an iid_time mask. Returns a (n, seq_len, n_features) float32
    array shaped to be applied multiplicatively (NaN where mask=0)."""
    rng = np.random.default_rng(seed)
    s_per_window = rng.choice(SPARSITY_GRID, size=n)
    keep = (rng.random((n, seq_len)) > s_per_window[:, None]).astype(np.float32)
    keep_full = np.repeat(keep[:, :, None], n_features, axis=2)
    return keep_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="path to a .npz with key 'clean' shape (N, T, D)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_k", type=int, default=16)
    ap.add_argument("--d_v", type=int, default=16)
    ap.add_argument("--d_ffn", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=None,
                    help="optional cap on training windows (for fast iter)")
    ap.add_argument("--n_val", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True,
                    help="output directory; PyPOTS will save the model under this path")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly per RUN_PLAN_V2.md.")

    out_dir = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[saits-train] loading corpus {args.corpus}")
    d = np.load(args.corpus)
    clean = d["clean"].astype(np.float32)  # (N, T, D)
    N_total, T, D = clean.shape
    n_train = N_total - args.n_val
    if args.n_train is not None:
        n_train = min(args.n_train, n_train)
    print(f"[saits-train] corpus shape {clean.shape}; train={n_train} val={args.n_val}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N_total)
    train_idx = perm[: n_train]
    val_idx = perm[n_train : n_train + args.n_val]

    train_clean = clean[train_idx]
    val_clean = clean[val_idx]
    train_keep = make_training_masks(n_train, T, D, seed=args.seed)
    val_keep = make_training_masks(args.n_val, T, D, seed=args.seed + 1)

    train_X = train_clean.copy()
    train_X[train_keep == 0] = np.nan
    val_X = val_clean.copy()
    val_X[val_keep == 0] = np.nan

    print(f"[saits-train] train missing fraction = {np.isnan(train_X).mean():.3f}")
    print(f"[saits-train] val   missing fraction = {np.isnan(val_X).mean():.3f}")

    from pypots.imputation import SAITS

    saits = SAITS(
        n_steps=T, n_features=D,
        n_layers=args.n_layers, d_model=args.d_model,
        n_heads=args.n_heads, d_k=args.d_k, d_v=args.d_v, d_ffn=args.d_ffn,
        batch_size=args.batch, epochs=args.epochs,
        verbose=True, device="cuda",
        saving_path=str(out_dir),
        model_saving_strategy="best",
    )

    print(f"[saits-train] starting fit ({args.epochs} epochs, batch={args.batch})")
    fit_t0 = time.time()
    # PyPOTS expects val_set = {"X": with_NaN, "X_ori": clean reference, "indicating_mask": ?}
    # Use the artificial-missing scheme: X_ori = clean, X = clean with NaN at val_keep==0;
    # SAITS uses X_ori internally to score validation imputation MAE.
    saits.fit({"X": train_X},
               val_set={"X": val_X, "X_ori": val_clean})
    print(f"[saits-train] fit done in {time.time() - fit_t0:.1f}s")

    # Final imputation MAE on a validation slice (sanity)
    val_imp = saits.impute({"X": val_X})
    val_imp = np.asarray(val_imp).reshape(args.n_val, T, D)
    miss_mask = np.isnan(val_X)
    if miss_mask.any():
        mae = float(np.mean(np.abs(val_imp[miss_mask] - val_clean[miss_mask])))
        attr_std = float(d["attractor_std"]) if "attractor_std" in d.files else 1.0
        print(f"[saits-train] val MAE on missing = {mae:.4f} "
              f"(attractor_std={attr_std:.3f}, normalized {mae/max(attr_std,1e-9):.4f})")

    meta = {
        "corpus": args.corpus,
        "n_train": int(n_train), "n_val": int(args.n_val),
        "epochs": int(args.epochs), "batch": int(args.batch),
        "n_layers": int(args.n_layers), "d_model": int(args.d_model),
        "n_heads": int(args.n_heads), "d_k": int(args.d_k), "d_v": int(args.d_v),
        "d_ffn": int(args.d_ffn),
        "sparsity_grid": list(SPARSITY_GRID),
        "saving_path": str(out_dir),
        "wall_seconds": float(time.time() - t0),
    }
    (out_dir.parent / f"{out_dir.name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[saits-train] saved meta -> {out_dir.parent / (out_dir.name + '_meta.json')}")
    print(f"[saits-train] saved checkpoint under -> {out_dir} (PyPOTS layout)")


if __name__ == "__main__":
    main()
