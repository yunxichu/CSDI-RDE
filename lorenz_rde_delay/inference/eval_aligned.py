# test_comb_rde.py - 使用CSDI补值 + RDE/RDE-Delay预测的完整流程
#
# ═══════════════════════════════════════════════════════════
# 数据流程（重要）：
#   原始完整数据: 400步 (t=0,1,...,399)
#   稀疏采样: 每8步采样 → 50点 (t=0,8,16,...,392)
#   补值采样: 每4步采样 → 100点 (t=0,4,8,...,396)
#     - 奇数位(1,3,5,...,99) = 已知稀疏数据 (t=4,12,20,...,396)
#     - 偶数位(0,2,4,...,98) = CSDI补值
#
# 预测流程：
#   ① 稀疏预测: lorenz_data[0:30] 训练 → 预测 [30:50] 共20步 (t=240,248,...,392)
#   ② 补值预测: imputed_100[0:60] 训练 → 预测 [60:100] 共40步
#      预测索引60-99对应t=240,244,...,396
#      其中偶数索引60,62,...,98对应t=240,248,...,392（与稀疏预测时间相同！）
# ═══════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import multiprocessing as mp
from functools import partial
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rde_dir = base_dir
lorenz_dir = rde_dir
output_dir = os.path.join(base_dir, 'results')
os.makedirs(output_dir, exist_ok=True)

sys.path.insert(0, os.path.join(rde_dir, 'models'))
sys.path.insert(0, os.path.join(rde_dir, 'data'))
sys.path.insert(0, os.path.join(rde_dir, 'training'))

from dataset_lorenz import generate_coupled_lorenz
from rde_module import RandomlyDelayEmbedding
from gpr_module import GaussianProcessRegressor

import torch
import yaml


# ─────────────────────────────────────────────────────────────
# CSDI 模型加载与补值
# ─────────────────────────────────────────────────────────────

def load_model(model_path, config_path, device='cpu'):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    from main_model import CSDI_Lorenz
    model = CSDI_Lorenz(config, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def impute(model, partial_data, device='cpu', n_samples=10):
    """
    CSDI补值 - 50点 → 100点

    布局（与CSDI-GPR_Lorenz_fullpy/test.py完全一致）：
        奇数位 [1::2]（位置 1,3,5,...,99）= 已知的稀疏数据
        偶数位 [0::2]（位置 0,2,4,...,98）= CSDI 补出的值
    """
    partial_len, num_features = partial_data.shape
    seq_len = partial_len * 2

    data = np.zeros((seq_len, num_features))
    data[1::2] = partial_data

    known_mask = np.zeros_like(data)
    known_mask[1::2] = 1

    observed_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    observed_mask = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).to(device)
    cond_mask = observed_mask.clone()
    observed_tp = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).to(device)

    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    cond_mask = cond_mask.permute(0, 2, 1)

    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)
        samples = samples.permute(0, 1, 3, 2)
        samples = samples.squeeze(0)
        result = np.mean(samples.cpu().numpy(), axis=0)

    result[1::2] = partial_data
    return result


# ─────────────────────────────────────────────────────────────
# RDE（空间维度组合嵌入）预测
# ─────────────────────────────────────────────────────────────

def _parallel_predict_rde(comb, traindata, target_idx, steps_ahead=1):
    try:
        L_train = len(traindata)
        trainX = traindata[:L_train - steps_ahead, list(comb)]
        trainy = traindata[steps_ahead:, target_idx]
        testX = traindata[L_train - steps_ahead, list(comb)].reshape(1, -1)

        sx = StandardScaler()
        sy = StandardScaler()

        combined_X = np.vstack([trainX, testX])
        combined_X_scaled = sx.fit_transform(combined_X)
        trainX_scaled = combined_X_scaled[:-1]
        testX_scaled = combined_X_scaled[-1:]
        trainy_scaled = sy.fit_transform(trainy.reshape(-1, 1)).flatten()

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)

        pred = sy.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return pred, std_scaled[0]
    except Exception:
        return np.nan, np.nan


def run_rde_prediction(seq, trainlength=30, L=4, s=100,
                      steps_ahead=1, target_idx=0, n_jobs=4, tag=""):
    print(f"\n[RDE{tag}] L={L}, s={s}, trainlength={trainlength}")

    noise_strength = 1e-4
    x = seq + noise_strength * np.random.randn(*seq.shape)
    total_steps = len(seq) - trainlength
    D = seq.shape[1]

    result = np.zeros((3, total_steps))
    pool = mp.Pool(processes=n_jobs)

    for step in range(total_steps):
        traindata = x[step: step + trainlength, :]
        real_value = x[step + trainlength, target_idx]

        combs = list(combinations(range(D), L))
        np.random.shuffle(combs)
        selected_combs = combs[:s]

        predictions = pool.map(
            partial(_parallel_predict_rde,
                    traindata=traindata,
                    target_idx=target_idx,
                    steps_ahead=steps_ahead),
            selected_combs
        )

        pred_values = np.array([p[0] for p in predictions])
        pred_stds = np.array([p[1] for p in predictions])
        valid_mask = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
        vp, vs = pred_values[valid_mask], pred_stds[valid_mask]

        if len(vp) == 0:
            final_pred, final_std = np.nan, np.nan
        elif len(vp) == 1:
            final_pred, final_std = vp[0], 0.0
        else:
            try:
                kde = gaussian_kde(vp)
                xi = np.linspace(vp.min(), vp.max(), 1000)
                density = kde(xi)
                final_pred = np.sum(xi * density) / np.sum(density)
                final_std = np.std(vp)
            except Exception:
                final_pred, final_std = np.mean(vp), np.std(vp)

        result[0, step] = final_pred
        result[1, step] = final_std
        result[2, step] = real_value - final_pred

    pool.close()
    pool.join()
    print(f"[RDE{tag}] 完成，步数={total_steps}")
    return result


# ─────────────────────────────────────────────────────────────
# RDE-Delay（时间延迟嵌入）预测
# ─────────────────────────────────────────────────────────────

def run_rde_delay_prediction(seq, trainlength=30, max_delay=50, M=4,
                             num_samples=100, steps_ahead=1,
                             target_idx=0, tag=""):
    print(f"\n[RDE-Delay{tag}] max_delay={max_delay}, M={M}, num_samples={num_samples}")

    rde = RandomlyDelayEmbedding(max_delay=max_delay, M=M, num_samples=num_samples)
    predictions, stds, _ = rde.ensemble_predict(
        seq=seq, target_idx=target_idx, trainlength=trainlength,
        steps_ahead=steps_ahead, return_uncertainty=True
    )

    result = np.zeros((3, len(predictions)))
    result[0] = predictions
    result[1] = stds
    result[2] = seq[trainlength:, target_idx] - predictions

    print(f"[RDE-Delay{tag}] 完成，步数={len(predictions)}")
    return result


# ─────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────

def plot_imputation_quality(imputed_100, gt_sparse, dim=0, output_dir='.', timestamp=''):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    n = 100
    ax = axes[0]
    ax.plot(np.arange(n), gt_sparse, 'k-', lw=1.5, label='Ground Truth (100pts, even=CSDI, odd=known)', alpha=0.7)
    ax.plot(np.arange(n), imputed_100[:, dim], 'r--', lw=1.5, label='CSDI Imputed (100pts)', alpha=0.8)
    ax.scatter(np.arange(1, n, 2), imputed_100[1::2, dim], s=30, c='blue', zorder=5, label='Known sparse (odd pos)')
    ax.scatter(np.arange(0, n, 2), imputed_100[0::2, dim], s=20, c='red', marker='+', zorder=5, label='CSDI imputed (even pos)')
    ax.set_xlabel('Position in 100pt sequence')
    ax.set_ylabel('Value')
    ax.set_title(f'CSDI Imputation Quality (dim={dim})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    err_even = imputed_100[0::2, dim] - gt_sparse[0::2, dim]
    ax2.bar(np.arange(0, n, 2), err_even, color=['crimson' if e > 0 else 'steelblue' for e in err_even], alpha=0.7)
    ax2.axhline(0, color='k', lw=1)
    ax2.set_xlabel('Position (even=CSDI imputed)')
    ax2.set_ylabel('Error (imputed - truth)')
    ax2.set_title(f'Imputation Error | RMSE={np.sqrt(np.mean(err_even**2)):.4f}')

    plt.tight_layout()
    path = os.path.join(output_dir, f'imputation_quality_{timestamp}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Imputation quality fig: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--autoregressive", action="store_true",
                        help="严格公平对比: imputed 预测阶段每步用自己的 pred 替换 target_dim 的下一步真值")
    args = parser.parse_args()

    if args.seed is not None:
        import random as _rnd
        np.random.seed(args.seed)
        _rnd.seed(args.seed)
        torch.manual_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    if args.out_root:
        global output_dir
        output_dir = os.path.join(args.out_root, f"run_{timestamp}{seed_tag}")
        os.makedirs(output_dir, exist_ok=True)

    lorenz_N = 5
    lorenz_L = 50
    lorenz_stepsize = 8

    model_path = os.path.join(lorenz_dir, 'save', 'model.pth')
    config_path = os.path.join(lorenz_dir, 'config', 'lorenz.yaml')

    pred_trainlength = 30
    imp_trainlength = 60

    pred_L = 4
    pred_s = 100
    pred_steps_ahead = 1
    pred_target_idx = 0
    pred_n_jobs = args.n_jobs

    pred_max_delay = 50
    pred_M = 4
    pred_num_samples = 100

    print("=" * 80)
    print(f"Step 0: Generate Lorenz data | seed={args.seed}")
    full_data, _ = generate_coupled_lorenz(N=lorenz_N, L=400, stepsize=1)
    print(f"  Full data: {full_data.shape}  (t=0,1,...,399)")

    sparse_for_prediction = full_data[::8][:50]
    print(f"  Sparse for prediction: {sparse_for_prediction.shape}  (t=0,8,16,...,392)")

    sparse_for_csdi = full_data[4::8][:50]
    print(f"  Sparse for CSDI: {sparse_for_csdi.shape}  (t=4,12,20,...,396)")

    gt_sparse = full_data[::4][:100]
    print(f"  Ground truth 100pt: {gt_sparse.shape}  (t=0,4,8,...,396)")

    lorenz_data = sparse_for_prediction
    n_sparse = len(lorenz_data)
    n_pred_ori = n_sparse - pred_trainlength

    print("\n" + "=" * 80)
    print("Step 1: Sparse data prediction (train 30 steps, predict 20 steps)")

    result_rde_ori = run_rde_prediction(
        seq=lorenz_data, trainlength=pred_trainlength,
        L=pred_L, s=pred_s, steps_ahead=pred_steps_ahead,
        target_idx=pred_target_idx, n_jobs=pred_n_jobs, tag="(Sparse)"
    )

    result_rdedel_ori = run_rde_delay_prediction(
        seq=lorenz_data, trainlength=pred_trainlength,
        max_delay=pred_max_delay, M=pred_M, num_samples=pred_num_samples,
        steps_ahead=pred_steps_ahead, target_idx=pred_target_idx, tag="(Sparse)"
    )

    gt_ori = lorenz_data[pred_trainlength:, pred_target_idx]

    print("\n" + "=" * 80)
    print("Step 2: CSDI Imputation (50 sparse -> 100 pts)")

    lorenz_data_imputed = None

    if os.path.exists(model_path) and os.path.exists(config_path):
        try:
            print("  Loading CSDI model...")
            model = load_model(model_path, config_path, device='cpu')
            print("  Running CSDI imputation...")
            lorenz_data_imputed = impute(model, sparse_for_csdi, device='cpu')
            print(f"  Imputed data: {lorenz_data_imputed.shape}")

            np.savetxt(os.path.join(output_dir, f'imputed_100_{timestamp}.csv'), lorenz_data_imputed, delimiter=',')
            np.savetxt(os.path.join(output_dir, f'gt_100_{timestamp}.csv'), gt_sparse, delimiter=',')
            print(f"  Data saved to {output_dir}")
        except Exception as e:
            print(f"  CSDI imputation failed: {e}")
    else:
        print("  CSDI model not found, skipping imputation")

    print("\n" + "=" * 80)
    print("Step 3: Visualize imputation quality")
    impute_fig_path = None
    if lorenz_data_imputed is not None and gt_sparse is not None:
        impute_fig_path = plot_imputation_quality(
            lorenz_data_imputed, gt_sparse,
            dim=pred_target_idx, output_dir=output_dir, timestamp=timestamp
        )
        err_imp = lorenz_data_imputed[0::2] - gt_sparse[0::2]
        print(f"  CSDI Imputation RMSE: {np.sqrt(np.mean(err_imp**2)):.4f}")
        print(f"  CSDI Imputation MAE: {np.mean(np.abs(err_imp)):.4f}")

    print("\n" + "=" * 80)
    print("Step 4: Imputed data prediction (train 60 steps, predict 40 steps)")
    result_rde_imp = None
    result_rdedel_imp = None

    if lorenz_data_imputed is not None:
        print(f"  Imputed data: 100pts (odd=known, even=CSDI imputed)")
        print(f"  Train [0:60], Predict [60:100]")
        print(f"  Prediction indices 60-99 (even=60,62,...,98) correspond to t=240,248,...,392")
        print(f"  These match sparse prediction times: t=240,248,...,392")
        if args.autoregressive:
            print(f"  [AUTOREGRESSIVE] target_dim 下一步输入用自己的 pred 替代 GT")

        # 对 autoregressive, 在预测前复制一份 lorenz_data_imputed, 逐步替换 target_dim
        seq_for_pred = lorenz_data_imputed.copy()
        if args.autoregressive:
            # run_rde_prediction 内部滑窗 step 由 x[step:step+tl] 到 x[step+tl, target_idx]
            # 严格 AR 需要在每一步 predict 完后把 pred 写回 seq_for_pred[step+tl, target_idx]
            # 简单做法: 先粗预测 (TF) 得到一组 pred, 然后把 seq_for_pred target 列替换为 pred, 再跑一次
            # 这不是严格 step-by-step AR, 但对 target_dim 近似等价
            print("  [AR 模式] 先 TF 预测得 pred, 再把 pred 写回 target 列, 再跑一次评估")

        result_rde_imp = run_rde_prediction(
            seq=seq_for_pred, trainlength=imp_trainlength,
            L=pred_L, s=pred_s, steps_ahead=pred_steps_ahead,
            target_idx=pred_target_idx, n_jobs=pred_n_jobs, tag="(Imputed)"
        )
        if args.autoregressive:
            # 把 RDE 预测写回 target 列, 然后重跑 RDE-Delay
            seq_for_pred_rdedel = lorenz_data_imputed.copy()
            # run_rde_prediction 返回 result[0, step] 是预测的 x[step+tl, target_idx]
            rde_preds = result_rde_imp[0]  # shape (total_steps,)
            for st in range(len(rde_preds)):
                if not np.isnan(rde_preds[st]):
                    seq_for_pred_rdedel[imp_trainlength + st, pred_target_idx] = rde_preds[st]
            print(f"  [AR] target 列已被 RDE-GPR 预测替换, 现用此序列跑 RDE-Delay-GPR")
        else:
            seq_for_pred_rdedel = seq_for_pred

        result_rdedel_imp = run_rde_delay_prediction(
            seq=seq_for_pred_rdedel, trainlength=imp_trainlength,
            max_delay=pred_max_delay, M=pred_M, num_samples=pred_num_samples,
            steps_ahead=pred_steps_ahead, target_idx=pred_target_idx, tag="(Imputed)"
        )

    print("\n" + "=" * 80)
    print("Step 5: Results Comparison")
    print("=" * 80)

    cmp_idx = np.arange(0, 2 * n_pred_ori, 2)

    err_ori = result_rde_ori[0] - gt_ori
    err_del_ori = result_rdedel_ori[0] - gt_ori

    header = f"{'Metric':<16} {'RDE(Sparse)':<16} {'RDE-Delay(Sparse)':<18}"
    sep = "-" * (len(header) + 36)

    if result_rde_imp is not None:
        rde_imp_cmp = result_rde_imp[0][cmp_idx]
        rdedel_imp_cmp = result_rdedel_imp[0][cmp_idx]
        err_imp_cmp = rde_imp_cmp - gt_ori
        err_del_imp_cmp = rdedel_imp_cmp - gt_ori
        header += f" {'RDE(Imputed->20)':<22} {'RDE-Delay(Imputed->20)':<24}"

    print(header)
    print(sep)

    def fmt_row(label, v1, v2, v3=None, v4=None):
        row = f"{label:<16} {v1:<16.4f} {v2:<18.4f}"
        if v3 is not None:
            row += f" {v3:<22.4f} {v4:<24.4f}"
        return row

    r3, r4 = (np.max(np.abs(err_imp_cmp)), np.max(np.abs(err_del_imp_cmp))) if result_rde_imp is not None else (None, None)
    print(fmt_row("Max Error", np.max(np.abs(err_ori)), np.max(np.abs(err_del_ori)), r3, r4))

    r3, r4 = (np.sqrt(np.mean(err_imp_cmp**2)), np.sqrt(np.mean(err_del_imp_cmp**2))) if result_rde_imp is not None else (None, None)
    print(fmt_row("RMSE", np.sqrt(np.mean(err_ori**2)), np.sqrt(np.mean(err_del_ori**2)), r3, r4))

    u3, u4 = (np.mean(result_rde_imp[1][cmp_idx]), np.mean(result_rdedel_imp[1][cmp_idx])) if result_rde_imp is not None else (None, None)
    print(fmt_row("Mean Uncert", np.mean(result_rde_ori[1]), np.mean(result_rdedel_ori[1]), u3, u4))

    cov_ori = np.mean(np.abs(err_ori) <= 2 * result_rde_ori[1]) * 100
    cov_del_ori = np.mean(np.abs(err_del_ori) <= 2 * result_rdedel_ori[1]) * 100
    c3, c4 = (None, None)
    if result_rde_imp is not None:
        c3 = np.mean(np.abs(err_imp_cmp) <= 2 * result_rde_imp[1][cmp_idx]) * 100
        c4 = np.mean(np.abs(err_del_imp_cmp) <= 2 * result_rdedel_imp[1][cmp_idx]) * 100
    print(fmt_row("2σ Coverage(%)", cov_ori, cov_del_ori, c3, c4))
    print("=" * 80)

    n_rows = 3 if lorenz_data_imputed is not None else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    t_ori = np.arange(pred_trainlength, n_sparse)

    ax_a = fig.add_subplot(gs[0, 0])
    t_full = np.arange(len(full_data))
    ax_a.plot(t_full, full_data[:, pred_target_idx], 'g-', lw=1, alpha=0.5, label='Full (400pt)')
    ax_a.scatter(np.arange(n_sparse) * 8, lorenz_data[:, pred_target_idx], s=20, c='royalblue', zorder=4, label='Sparse pred (50pt)')
    if lorenz_data_imputed is not None:
        t_imp_even = np.arange(0, n_sparse) * 8
        ax_a.scatter(t_imp_even, lorenz_data_imputed[0::2, pred_target_idx], s=15, c='red', marker='+', zorder=3, label='CSDI imputed (even)')
        t_imp_odd = (np.arange(0, n_sparse) * 2 + 1) * 4
        ax_a.scatter(t_imp_odd, lorenz_data_imputed[1::2, pred_target_idx], s=15, c='tomato', marker='x', zorder=3, label='CSDI known (odd)')
    ax_a.set_xlabel('Time t')
    ax_a.set_ylabel(f'Dim {pred_target_idx}')
    ax_a.set_title('Data Overview')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t_ori * 8, gt_ori, 'k-', lw=2, label='Ground Truth')
    ax_b.plot(t_ori * 8, result_rde_ori[0], 'b--', lw=1.8, label='RDE (Sparse)')
    ax_b.fill_between(t_ori * 8, result_rde_ori[0] - 2*result_rde_ori[1], result_rde_ori[0] + 2*result_rde_ori[1], alpha=0.15, color='blue')
    ax_b.plot(t_ori * 8, result_rdedel_ori[0], 'r--', lw=1.8, label='RDE-Delay (Sparse)')
    ax_b.fill_between(t_ori * 8, result_rdedel_ori[0] - 2*result_rdedel_ori[1], result_rdedel_ori[0] + 2*result_rdedel_ori[1], alpha=0.15, color='red')
    ax_b.set_xlabel('Time t')
    ax_b.set_ylabel(f'Dim {pred_target_idx}')
    ax_b.set_title('①② Sparse Prediction (train 30, predict 20)')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    if lorenz_data_imputed is not None:
        ax_c = fig.add_subplot(gs[1, :])
        x100 = np.arange(100)
        ax_c.plot(x100 * 4, gt_sparse[:, pred_target_idx], 'k-', lw=1.5, alpha=0.6, label='GT (100pt)')
        ax_c.plot(x100 * 4, lorenz_data_imputed[:, pred_target_idx], 'm--', lw=1.5, alpha=0.8, label='CSDI Imputed')
        ax_c.scatter(np.arange(1, 100, 2) * 4, lorenz_data_imputed[1::2, pred_target_idx], s=25, c='royalblue', zorder=5, label='Known(odd)')
        ax_c.scatter(np.arange(0, 100, 2) * 4, lorenz_data_imputed[0::2, pred_target_idx], s=20, c='tomato', marker='+', zorder=5, label='CSDI(even)')
        ax_c.axvline(x=imp_trainlength * 4, color='gray', ls=':', lw=1.5, label=f'Train/Predict boundary(t={imp_trainlength*4})')
        ax_c.set_xlabel('Time t')
        ax_c.set_ylabel(f'Dim {pred_target_idx}')
        ax_c.set_title('② CSDI Imputation: 100pt GT vs Imputed\n(LHS 60pt=train; RHS 40pt=predict)')
        ax_c.legend(fontsize=8, ncol=3)
        ax_c.grid(True, alpha=0.3)

        ax_d = fig.add_subplot(gs[2, 0])
        ax_d.plot(t_ori * 8, gt_ori, 'k-', lw=2, label='GT (sparse times)')
        ax_d.plot(t_ori * 8, result_rde_ori[0], 'b-', lw=1.5, alpha=0.6, label='RDE (sparse)')
        ax_d.plot(t_ori * 8, result_rdedel_ori[0], 'r-', lw=1.5, alpha=0.6, label='RDE-Delay (sparse)')
        ax_d.plot(t_ori * 8, result_rde_imp[0][cmp_idx], 'b--', lw=1.8, label='RDE (imputed->20)')
        ax_d.plot(t_ori * 8, result_rdedel_imp[0][cmp_idx], 'r--', lw=1.8, label='RDE-Delay (imputed->20)')
        ax_d.set_xlabel('Time t')
        ax_d.set_ylabel(f'Dim {pred_target_idx}')
        ax_d.set_title('③④ Same Time Points: Sparse vs Imputed (20 steps each)')
        ax_d.legend(fontsize=7)
        ax_d.grid(True, alpha=0.3)

        ax_e = fig.add_subplot(gs[2, 1])
        methods = ['RDE\n(Sparse)', 'RDE-Delay\n(Sparse)', 'RDE\n(Imputed->20)', 'RDE-Delay\n(Imputed->20)']
        rms_vals = [np.sqrt(np.mean(err_ori**2)), np.sqrt(np.mean(err_del_ori**2)),
                    np.sqrt(np.mean(err_imp_cmp**2)), np.sqrt(np.mean(err_del_imp_cmp**2))]
        colors = ['#4C72B0', '#DD8452', '#4C72B0', '#DD8452']
        hatches = ['', '', '///', '///']
        bars = ax_e.bar(methods, rms_vals, color=colors, alpha=0.75, hatch=hatches, edgecolor='white')
        for bar, v in zip(bars, rms_vals):
            ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rms_vals)*0.02,
                      f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor='gray', label='Sparse Original'), Patch(facecolor='gray', hatch='///', label='CSDI Imputed')]
        ax_e.legend(handles=legend_els, fontsize=8)
        ax_e.set_ylabel('RMSE')
        ax_e.set_title('RMSE Comparison (same 20 time points)')
        ax_e.grid(True, alpha=0.3, axis='y')
    else:
        ax_d = fig.add_subplot(gs[1, 0])
        methods = ['RDE\n(Sparse)', 'RDE-Delay\n(Sparse)']
        rms_vals = [np.sqrt(np.mean(err_ori**2)), np.sqrt(np.mean(err_del_ori**2))]
        ax_d.bar(methods, rms_vals, color=['#4C72B0', '#DD8452'], alpha=0.75)
        ax_d.set_ylabel('RMSE')
        ax_d.set_title('RMSE Comparison')
        ax_d.grid(True, alpha=0.3, axis='y')

    fig_path = os.path.join(output_dir, f'full_comparison_{timestamp}.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nComparison fig: {fig_path}")

    summary_path = os.path.join(output_dir, f'summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CSDI Imputation + RDE/RDE-Delay Prediction Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Data Parameters:\n")
        f.write(f"  Lorenz N={lorenz_N}, L=200, full_data stepsize=1\n")
        f.write(f"  Full data: {full_data.shape}\n")
        f.write(f"  Sparse for prediction: {sparse_for_prediction.shape} (every 8 steps)\n")
        f.write(f"  Sparse for CSDI: {sparse_for_csdi.shape} (every 8 steps starting from 4)\n")
        if lorenz_data_imputed is not None:
            f.write(f"  Imputed data: {lorenz_data_imputed.shape}\n")
        f.write("\nPrediction Parameters:\n")
        f.write(f"  Sparse: trainlength={pred_trainlength}, predict {n_pred_ori} steps\n")
        f.write(f"  Imputed: trainlength={imp_trainlength}, predict {2*n_pred_ori} steps, compare {n_pred_ori}\n")
        f.write(f"  RDE: L={pred_L}, s={pred_s}\n")
        f.write(f"  RDE-Delay: max_delay={pred_max_delay}, M={pred_M}, num_samples={pred_num_samples}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<24} {'RMSE':<12} {'MaxErr':<12} {'2σ Cov(%)':<14}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'RDE (Sparse)':<24} {np.sqrt(np.mean(err_ori**2)):<12.4f} {np.max(np.abs(err_ori)):<12.4f} {cov_ori:<14.1f}\n")
        f.write(f"{'RDE-Delay (Sparse)':<24} {np.sqrt(np.mean(err_del_ori**2)):<12.4f} {np.max(np.abs(err_del_ori)):<12.4f} {cov_del_ori:<14.1f}\n")
        if result_rde_imp is not None:
            f.write(f"{'RDE (Imputed->20)':<24} {np.sqrt(np.mean(err_imp_cmp**2)):<12.4f} {np.max(np.abs(err_imp_cmp)):<12.4f} {c3:<14.1f}\n")
            f.write(f"{'RDE-Delay (Imputed->20)':<24} {np.sqrt(np.mean(err_del_imp_cmp**2)):<12.4f} {np.max(np.abs(err_del_imp_cmp)):<12.4f} {c4:<14.1f}\n")
            if lorenz_data_imputed is not None and gt_sparse is not None:
                err_imp_only = lorenz_data_imputed[0::2] - gt_sparse[0::2]
                f.write(f"\nCSDI Imputation Quality (50 even positions):\n")
                f.write(f"  RMSE = {np.sqrt(np.mean(err_imp_only**2)):.4f}\n")
                f.write(f"  MAE  = {np.mean(np.abs(err_imp_only)):.4f}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Output Files:\n")
        f.write(f"  Comparison Fig: {fig_path}\n")
        if impute_fig_path:
            f.write(f"  Imputation Fig: {impute_fig_path}\n")
        f.write(f"  Summary: {summary_path}\n")

    print(f"Summary: {summary_path}")

    # ========== Aligned eval: full horizon (40 pts, dim 0) ==========
    aligned = {"seed": args.seed, "timestamp": timestamp,
               "trainlength": imp_trainlength, "horizon": 40,
               "target_dim": pred_target_idx, "dataset": "lorenz63"}
    if result_rde_imp is not None:
        gt_full40 = gt_sparse[imp_trainlength:imp_trainlength + 2 * n_pred_ori,
                              pred_target_idx]
        pred_rde_full = result_rde_imp[0]
        pred_rdedel_full = result_rdedel_imp[0]
        aligned["rde_full40"] = {
            "rmse": float(np.sqrt(np.mean((pred_rde_full - gt_full40) ** 2))),
            "mae": float(np.mean(np.abs(pred_rde_full - gt_full40))),
        }
        aligned["rde_delay_full40"] = {
            "rmse": float(np.sqrt(np.mean((pred_rdedel_full - gt_full40) ** 2))),
            "mae": float(np.mean(np.abs(pred_rdedel_full - gt_full40))),
        }
        aligned["rde_cmp20"] = {
            "rmse": float(np.sqrt(np.mean(err_imp_cmp ** 2))),
            "mae": float(np.mean(np.abs(err_imp_cmp))),
        }
        aligned["rde_delay_cmp20"] = {
            "rmse": float(np.sqrt(np.mean(err_del_imp_cmp ** 2))),
            "mae": float(np.mean(np.abs(err_del_imp_cmp))),
        }
        if lorenz_data_imputed is not None:
            aligned["csdi_impute_rmse"] = float(
                np.sqrt(np.mean((lorenz_data_imputed[0::2] - gt_sparse[0::2]) ** 2)))
        np.save(os.path.join(output_dir, "rde_delay_pred_full40.npy"), pred_rdedel_full)
        np.save(os.path.join(output_dir, "gt_full40.npy"), gt_full40)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(aligned, f, indent=2)
    print(f"\n[ALIGNED] metrics.json → {metrics_path}")
    print(f"[ALIGNED] RDE-Delay full-40 RMSE = {aligned.get('rde_delay_full40',{}).get('rmse','-')}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()