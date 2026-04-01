# test_comb_rde.py - 使用CSDI补值 + RDE/RDE-Delay预测的完整流程
#
# ═══════════════════════════════════════════════════════════
# 数据流示意：
#   full_data    : (400, 15)  完整轨迹，时间步 t=0,1,...,399
#   lorenz_data  : ( 50, 15)  每隔8步采样，t=0,8,16,...,392
#   imputed_100  : (100, 15)  CSDI补值后，步长4，t=0,4,8,...,396
#                    偶数位 [0::2] = lorenz_data（已知，t=0,8,...）
#                    奇数位 [1::2] = CSDI补出（t=4,12,...）
#   gt_100       : (100, 15)  imputed_100 对应的真实值
#                    gt_100[0::2] = full_data[0::8]  (= lorenz_data)
#                    gt_100[1::2] = full_data[4::8]
#
# 预测流程：
#   ① 稀疏预测  : lorenz_data[0:30] 训练 → 预测 [30:50] 共20步
#   ② 补值预测  : imputed_100[0:60] 训练 → 预测 [60:100] 共40步
#                    其中偶数偏移 [0,2,4,...,38] 与①对应同一时间点
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
rde_dir = os.path.join(base_dir)
lorenz_dir = os.path.join(base_dir, '..', 'lorenz_rde')
output_dir = os.path.join(base_dir, 'results')
os.makedirs(output_dir, exist_ok=True)

sys.path.insert(0, rde_dir)
sys.path.insert(0, os.path.join(rde_dir, 'models'))
sys.path.insert(0, os.path.join(lorenz_dir, 'models'))
sys.path.insert(0, os.path.join(lorenz_dir, 'data'))

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
    将 50 个稀疏数据点补值为 100 个点。

    布局约定（保证时间有序）：
        偶数位 [0::2]（位置 0,2,4,...,98）= 已知的稀疏数据  t=0,8,16,...
        奇数位 [1::2]（位置 1,3,5,...,99）= CSDI 补出的值    t=4,12,20,...

    Returns
    -------
    result : np.ndarray, shape (2*L, D)
        result[0::2] = partial_data  (还原)
        result[1::2] = CSDI 补值
    """
    partial_len, num_features = partial_data.shape   # (50, 15)
    seq_len = partial_len * 2                         # 100

    data = np.zeros((seq_len, num_features))
    data[0::2] = partial_data          # 已知值放在偶数位（时间有序：t=0,8,...）

    known_mask = np.zeros_like(data)
    known_mask[0::2] = 1               # 1=已知, 0=待补（奇数位）

    observed_data = torch.tensor(data,       dtype=torch.float32).unsqueeze(0).to(device)
    observed_mask = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).to(device)
    cond_mask     = observed_mask.clone()
    observed_tp   = torch.arange(seq_len,    dtype=torch.float32).unsqueeze(0).to(device)

    # 模型期望 (B, D, T)
    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    cond_mask     = cond_mask.permute(0, 2, 1)

    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples   = model.impute(observed_data, cond_mask, side_info, n_samples)
        samples   = samples.permute(0, 1, 3, 2)   # (B, n_samples, T, D)
        samples   = samples.squeeze(0)
        result    = np.mean(samples.cpu().numpy(), axis=0)   # (T, D)

    result[0::2] = partial_data   # 已知位置还原为精确值
    return result


def build_ground_truth_100(full_data, stepsize=8):
    """
    根据完整数据构造 100 点真实值，与 imputed_100 的布局对齐。

    偶数位 → full_data[0::stepsize]     (= lorenz_data，t=0,8,...)
    奇数位 → full_data[stepsize//2::stepsize]  (t=4,12,...)
    """
    num_features = full_data.shape[1]
    sparse   = full_data[0::stepsize]                     # (50, D)
    midpoint = full_data[stepsize // 2::stepsize]         # (50, D)
    n = min(len(sparse), len(midpoint))
    gt_100 = np.zeros((2 * n, num_features))
    gt_100[0::2] = sparse[:n]
    gt_100[1::2] = midpoint[:n]
    return gt_100


# ─────────────────────────────────────────────────────────────
# RDE（空间维度组合嵌入）预测
# ─────────────────────────────────────────────────────────────

def _parallel_predict_rde(comb, traindata, target_idx, steps_ahead=1):
    try:
        L_train = len(traindata)
        trainX  = traindata[:L_train - steps_ahead, list(comb)]
        trainy  = traindata[steps_ahead:,            target_idx]
        testX   = traindata[L_train - steps_ahead,   list(comb)].reshape(1, -1)

        sx = StandardScaler()
        sy = StandardScaler()

        combined_X        = np.vstack([trainX, testX])
        combined_X_scaled = sx.fit_transform(combined_X)
        trainX_scaled     = combined_X_scaled[:-1]
        testX_scaled      = combined_X_scaled[-1:]
        trainy_scaled     = sy.fit_transform(trainy.reshape(-1, 1)).flatten()

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled,
               init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)

        pred = sy.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return pred, std_scaled[0]
    except Exception:
        return np.nan, np.nan


def run_rde_prediction(seq, trainlength=30, L=4, s=100,
                       steps_ahead=1, target_idx=0, n_jobs=4, tag=""):
    print(f"\n[RDE{tag}] 空间维度嵌入预测  L={L}, s={s}, trainlength={trainlength}")

    noise_strength = 1e-4
    x = seq + noise_strength * np.random.randn(*seq.shape)
    total_steps = len(seq) - trainlength
    D = seq.shape[1]

    result = np.zeros((3, total_steps))
    pool   = mp.Pool(processes=n_jobs)

    for step in range(total_steps):
        traindata  = x[step: step + trainlength, :]
        real_value = x[step + trainlength, target_idx]

        combs          = list(combinations(range(D), L))
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
        pred_stds   = np.array([p[1] for p in predictions])
        valid_mask  = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
        vp, vs      = pred_values[valid_mask], pred_stds[valid_mask]

        if len(vp) == 0:
            final_pred, final_std = np.nan, np.nan
        elif len(vp) == 1:
            final_pred, final_std = vp[0], 0.0
        else:
            try:
                kde     = gaussian_kde(vp)
                xi      = np.linspace(vp.min(), vp.max(), 1000)
                density = kde(xi)
                final_pred = np.sum(xi * density) / np.sum(density)
                final_std  = np.std(vp)
            except Exception:
                final_pred, final_std = np.mean(vp), np.std(vp)

        result[0, step] = final_pred
        result[1, step] = final_std
        result[2, step] = real_value - final_pred

    pool.close()
    pool.join()
    print(f"[RDE{tag}] 预测完成，步数={total_steps}")
    return result


# ─────────────────────────────────────────────────────────────
# RDE-Delay（时间延迟嵌入）预测
# ─────────────────────────────────────────────────────────────

def run_rde_delay_prediction(seq, trainlength=30, max_delay=50, M=4,
                             num_samples=100, steps_ahead=1,
                             target_idx=0, tag=""):
    print(f"\n[RDE-Delay{tag}] 时间延迟嵌入预测  "
          f"max_delay={max_delay}, M={M}, num_samples={num_samples}, trainlength={trainlength}")

    rde = RandomlyDelayEmbedding(max_delay=max_delay, M=M, num_samples=num_samples)
    predictions, stds, _ = rde.ensemble_predict(
        seq=seq,
        target_idx=target_idx,
        trainlength=trainlength,
        steps_ahead=steps_ahead,
        return_uncertainty=True
    )

    result = np.zeros((3, len(predictions)))
    result[0] = predictions
    result[1] = stds
    result[2] = seq[trainlength:, target_idx] - predictions

    print(f"[RDE-Delay{tag}] 预测完成，步数={len(predictions)}")
    return result


# ─────────────────────────────────────────────────────────────
# 可视化辅助
# ─────────────────────────────────────────────────────────────

def plot_imputation_quality(lorenz_data, imputed_100, gt_100,
                            dim=0, output_dir='.', timestamp='', stepsize=8):
    """
    可视化 CSDI 补值质量：
      - 上方：整体100点对比（补值 vs 真实）
      - 下方：仅奇数位（补出的点）误差
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'CSDI 补值质量对比（维度 {dim}）', fontsize=14, fontweight='bold')

    n = 100
    x_even = np.arange(0, n, 2)   # 已知位置
    x_odd  = np.arange(1, n, 2)   # 补值位置

    ax = axes[0]
    ax.plot(np.arange(n), gt_100[:, dim],     'k-',  lw=1.5, label='真实值 (100点)', alpha=0.7)
    ax.plot(np.arange(n), imputed_100[:, dim],'r--', lw=1.5, label='补值结果 (100点)', alpha=0.8)
    ax.scatter(x_even, lorenz_data[:, dim],   s=30,  c='blue', zorder=5, label='稀疏已知点', alpha=0.9)
    ax.scatter(x_odd,  imputed_100[1::2, dim], s=20, c='red',  zorder=5,
               marker='+', label='CSDI补出点', alpha=0.9)
    ax.set_xlabel('序列位置（偶=已知，奇=补值）')
    ax.set_ylabel('值')
    ax.set_title('100点序列：真实值 vs 补值结果')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    err_odd = imputed_100[1::2, dim] - gt_100[1::2, dim]
    ax2.bar(x_odd, err_odd, color=['crimson' if e > 0 else 'steelblue' for e in err_odd],
            alpha=0.7, width=0.8)
    ax2.axhline(0, color='k', lw=1)
    ax2.set_xlabel('序列位置（奇数=补值点）')
    ax2.set_ylabel('误差（补值 − 真实）')
    ax2.set_title(f'CSDI 补值误差  |  RMSE = {np.sqrt(np.mean(err_odd**2)):.4f}  '
                  f'| MAE = {np.mean(np.abs(err_odd)):.4f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'imputation_quality_{timestamp}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"补值质量图已保存: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Lorenz 生成参数 ──────────────────────────────────────
    lorenz_N      = 5        # 耦合系统数
    lorenz_L      = 50       # 稀疏步数（采样后）
    lorenz_stepsize = 8      # 采样步长 → 完整序列 = 50*8 = 400 步

    model_path  = os.path.join(lorenz_dir, 'save',   'model.pth')
    config_path = os.path.join(lorenz_dir, 'config', 'lorenz.yaml')

    # ── 预测参数 ─────────────────────────────────────────────
    pred_trainlength  = 30    # 稀疏数据训练长度（50中用30训练，预测20步）
    imp_trainlength   = 60    # 补值数据训练长度（100中用60训练，预测40步）

    pred_L            = 4
    pred_s            = 100
    pred_steps_ahead  = 1
    pred_target_idx   = 0
    pred_n_jobs       = 4

    pred_max_delay    = 50
    pred_M            = 4
    pred_num_samples  = 100

    # ════════════════════════════════════════════════════════
    # Step 0: 生成数据
    # ════════════════════════════════════════════════════════
    print("=" * 80)
    print("Step 0: 生成 Lorenz 系统数据")
    lorenz_data, full_data = generate_coupled_lorenz(
        N=lorenz_N, L=lorenz_L, stepsize=lorenz_stepsize
    )
    # lorenz_data : (50,  15)  t=0,8,16,...,392
    # full_data   : (400, 15)  t=0,1,2,...,399
    print(f"  稀疏采样数据: {lorenz_data.shape}  (50步，每步间隔{lorenz_stepsize}个原始步)")
    print(f"  完整数据:     {full_data.shape}")

    n_sparse   = len(lorenz_data)                  # 50
    n_pred_ori = n_sparse - pred_trainlength       # 20  （预测步数，稀疏）
    n_pred_imp = 2 * n_pred_ori                    # 40  （预测步数，补值）

    # ════════════════════════════════════════════════════════
    # Step 1: 稀疏数据预测（方法①②）
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Step 1: 对稀疏 50 步数据进行预测（训练30步，预测20步）")

    result_rde_ori = run_rde_prediction(
        seq=lorenz_data,
        trainlength=pred_trainlength,
        L=pred_L, s=pred_s,
        steps_ahead=pred_steps_ahead,
        target_idx=pred_target_idx,
        n_jobs=pred_n_jobs,
        tag="（稀疏）"
    )

    result_rdedel_ori = run_rde_delay_prediction(
        seq=lorenz_data,
        trainlength=pred_trainlength,
        max_delay=pred_max_delay,
        M=pred_M, num_samples=pred_num_samples,
        steps_ahead=pred_steps_ahead,
        target_idx=pred_target_idx,
        tag="（稀疏）"
    )

    # 稀疏预测对应的真实值
    gt_ori = lorenz_data[pred_trainlength:, pred_target_idx]   # shape (20,)

    # ════════════════════════════════════════════════════════
    # Step 2: CSDI 补值  50 → 100
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Step 2: CSDI 补值（50 稀疏 → 100 点）")

    lorenz_data_imputed = None
    gt_100              = None

    if os.path.exists(model_path) and os.path.exists(config_path):
        try:
            print("  加载 CSDI 模型...")
            model = load_model(model_path, config_path, device='cpu')
            print("  执行补值...")
            lorenz_data_imputed = impute(model, lorenz_data, device='cpu')
            # lorenz_data_imputed : (100, 15)
            #   [0::2] = lorenz_data（已知，t=0,8,...）
            #   [1::2] = CSDI 补出（t=4,12,...）
            print(f"  补值完成: {lorenz_data_imputed.shape}")

            # 100 点真实值（用于对比补值质量）
            gt_100 = build_ground_truth_100(full_data, stepsize=lorenz_stepsize)
            print(f"  100点真实值形状: {gt_100.shape}")

            np.savetxt(
                os.path.join(output_dir, f'imputed_100_{timestamp}.csv'),
                lorenz_data_imputed, delimiter=','
            )
            np.savetxt(
                os.path.join(output_dir, f'gt_100_{timestamp}.csv'),
                gt_100, delimiter=','
            )
            print(f"  数据已保存至 {output_dir}")
        except Exception as e:
            print(f"  CSDI 补值失败: {e}")
    else:
        print("  未找到 CSDI 模型文件，跳过补值")

    # ════════════════════════════════════════════════════════
    # Step 3: 可视化补值质量（100点原始 vs 补值）
    # ════════════════════════════════════════════════════════
    impute_fig_path = None
    if lorenz_data_imputed is not None and gt_100 is not None:
        print("\n" + "=" * 80)
        print("Step 3: 可视化补值质量（100点真实 vs 补值）")
        impute_fig_path = plot_imputation_quality(
            lorenz_data, lorenz_data_imputed, gt_100,
            dim=pred_target_idx,
            output_dir=output_dir,
            timestamp=timestamp,
            stepsize=lorenz_stepsize
        )

        # 打印补值误差摘要（仅奇数位=补出的点）
        err_imp = lorenz_data_imputed[1::2, :] - gt_100[1::2, :]
        print(f"  补值RMSE（所有维度均值）: {np.sqrt(np.mean(err_imp**2)):.4f}")
        print(f"  补值MAE （所有维度均值）: {np.mean(np.abs(err_imp)):.4f}")
        print(f"  已知点误差（应≈0）:       {np.max(np.abs(lorenz_data_imputed[0::2] - gt_100[0::2])):.2e}")

    # ════════════════════════════════════════════════════════
    # Step 4: 补值数据预测（方法③④）
    # ════════════════════════════════════════════════════════
    result_rde_imp    = None
    result_rdedel_imp = None

    if lorenz_data_imputed is not None:
        print("\n" + "=" * 80)
        print("Step 4: 对补值 100 步数据进行预测（训练60步，预测40步）")
        print(f"  → 预测步数40中，偶数偏移 [0,2,...,38] 对应稀疏预测的20个时间点")

        result_rde_imp = run_rde_prediction(
            seq=lorenz_data_imputed,
            trainlength=imp_trainlength,
            L=pred_L, s=pred_s,
            steps_ahead=pred_steps_ahead,
            target_idx=pred_target_idx,
            n_jobs=pred_n_jobs,
            tag="（补值）"
        )

        result_rdedel_imp = run_rde_delay_prediction(
            seq=lorenz_data_imputed,
            trainlength=imp_trainlength,
            max_delay=pred_max_delay,
            M=pred_M, num_samples=pred_num_samples,
            steps_ahead=pred_steps_ahead,
            target_idx=pred_target_idx,
            tag="（补值）"
        )

    # ════════════════════════════════════════════════════════
    # Step 5: 结果对比分析
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Step 5: 结果对比分析")
    print("=" * 80)

    # ── 对应索引说明 ────────────────────────────────────────
    # imputed_100 布局：偶数位=已知，奇数位=补值
    # 训练段 [0:60]：
    #   偶数位 0,2,...,58 → lorenz_data[0:30]（稀疏训练集完全对应）
    # 预测段 [60:100]（40步）：
    #   偶数偏移 0,2,...,38 → 位置 60,62,...,98 → lorenz_data[30:50]（与稀疏预测同时间点）
    #   奇数偏移 1,3,...,39 → 位置 61,63,...,99 → 补值预测点（时间点在中间）
    cmp_idx = np.arange(0, 2 * n_pred_ori, 2)   # [0,2,4,...,38]，长度=20

    err_ori      = result_rde_ori[0]    - gt_ori
    err_del_ori  = result_rdedel_ori[0] - gt_ori

    header = f"{'指标':<16} {'RDE（稀疏）':<16} {'RDE-Delay（稀疏）':<18}"
    sep    = "-" * (len(header) + 36)

    if result_rde_imp is not None:
        # 补值预测在对应位置的值
        rde_imp_cmp    = result_rde_imp[0][cmp_idx]
        rdedel_imp_cmp = result_rdedel_imp[0][cmp_idx]
        err_imp_cmp    = rde_imp_cmp    - gt_ori
        err_del_imp_cmp= rdedel_imp_cmp - gt_ori
        header += f" {'RDE（补值→对应20步）':<22} {'RDE-Delay（补值→对应20步）':<24}"

    print(header)
    print(sep)

    def fmt_row(label, v1, v2, v3=None, v4=None):
        row = f"{label:<16} {v1:<16.4f} {v2:<18.4f}"
        if v3 is not None:
            row += f" {v3:<22.4f} {v4:<24.4f}"
        return row

    # 最大误差
    r3, r4 = (np.max(np.abs(err_imp_cmp)), np.max(np.abs(err_del_imp_cmp))) \
              if result_rde_imp is not None else (None, None)
    print(fmt_row("最大误差",
                  np.max(np.abs(err_ori)),
                  np.max(np.abs(err_del_ori)), r3, r4))

    # RMS 误差
    r3, r4 = (np.sqrt(np.mean(err_imp_cmp**2)), np.sqrt(np.mean(err_del_imp_cmp**2))) \
              if result_rde_imp is not None else (None, None)
    print(fmt_row("RMSE",
                  np.sqrt(np.mean(err_ori**2)),
                  np.sqrt(np.mean(err_del_ori**2)), r3, r4))

    # 平均不确定性
    u3, u4 = (np.mean(result_rde_imp[1][cmp_idx]), np.mean(result_rdedel_imp[1][cmp_idx])) \
              if result_rde_imp is not None else (None, None)
    print(fmt_row("平均不确定性",
                  np.mean(result_rde_ori[1]),
                  np.mean(result_rdedel_ori[1]), u3, u4))

    # 2σ 覆盖率
    cov_ori     = np.mean(np.abs(err_ori)     <= 2 * result_rde_ori[1]) * 100
    cov_del_ori = np.mean(np.abs(err_del_ori) <= 2 * result_rdedel_ori[1]) * 100
    c3, c4 = (None, None)
    if result_rde_imp is not None:
        c3 = np.mean(np.abs(err_imp_cmp)     <= 2 * result_rde_imp[1][cmp_idx]) * 100
        c4 = np.mean(np.abs(err_del_imp_cmp) <= 2 * result_rdedel_imp[1][cmp_idx]) * 100
    print(fmt_row("2σ覆盖率(%)",
                  cov_ori, cov_del_ori, c3, c4))
    print("=" * 80)

    # ════════════════════════════════════════════════════════
    # Step 6: 绘制综合对比图
    # ════════════════════════════════════════════════════════
    n_rows = 3 if lorenz_data_imputed is not None else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 时间轴（以稀疏步为单位）
    t_ori  = np.arange(pred_trainlength, n_sparse)            # [30,...,49] 20点
    # 补值序列中对应的时间（换算回稀疏步尺度，偶数位=整数稀疏步）
    t_imp_full = np.linspace(imp_trainlength / 2,
                             (imp_trainlength + n_pred_imp - 1) / 2,
                             n_pred_imp)                       # 40点，步长0.5

    # ── Panel A: 原始数据概览 ─────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    t_full   = np.linspace(0, n_sparse - 1, len(full_data))
    ax_a.plot(t_full, full_data[:, pred_target_idx],
              'g-', lw=1, alpha=0.5, label='完整轨迹(400点)')
    ax_a.scatter(np.arange(n_sparse), lorenz_data[:, pred_target_idx],
                 s=20, c='royalblue', zorder=4, label='稀疏采样(50点)')
    if lorenz_data_imputed is not None:
        t_imp_even = np.arange(0, n_sparse)                   # 偶数位→整数稀疏步
        t_imp_odd  = np.arange(0, n_sparse) + 0.5             # 奇数位→半步
        ax_a.scatter(t_imp_odd, lorenz_data_imputed[1::2, pred_target_idx],
                     s=15, c='tomato', marker='+', zorder=3, label='CSDI补值(50点)')
    ax_a.set_xlabel('稀疏步（1步=8原始步）')
    ax_a.set_ylabel(f'维度 {pred_target_idx}')
    ax_a.set_title('数据概览')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # ── Panel B: 稀疏预测对比 ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t_ori, gt_ori, 'k-', lw=2, label='真实值')
    ax_b.plot(t_ori, result_rde_ori[0],    'b--', lw=1.8, label='RDE（稀疏）')
    ax_b.fill_between(t_ori,
                      result_rde_ori[0] - 2 * result_rde_ori[1],
                      result_rde_ori[0] + 2 * result_rde_ori[1],
                      alpha=0.15, color='blue')
    ax_b.plot(t_ori, result_rdedel_ori[0], 'r--', lw=1.8, label='RDE-Delay（稀疏）')
    ax_b.fill_between(t_ori,
                      result_rdedel_ori[0] - 2 * result_rdedel_ori[1],
                      result_rdedel_ori[0] + 2 * result_rdedel_ori[1],
                      alpha=0.15, color='red')
    ax_b.set_xlabel('稀疏步')
    ax_b.set_ylabel(f'维度 {pred_target_idx}')
    ax_b.set_title('①② 稀疏数据预测（训练30步，预测20步）')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    if lorenz_data_imputed is not None:
        # ── Panel C: 补值 vs 真实（100点细节）──────────────
        ax_c = fig.add_subplot(gs[1, :])
        x100 = np.arange(100)
        ax_c.plot(x100, gt_100[:, pred_target_idx],
                  'k-', lw=1.5, alpha=0.6, label='100点真实值')
        ax_c.plot(x100, lorenz_data_imputed[:, pred_target_idx],
                  'm--', lw=1.5, alpha=0.8, label='100点补值结果')
        ax_c.scatter(np.arange(0, 100, 2), lorenz_data[:, pred_target_idx],
                     s=25, c='royalblue', zorder=5, label='已知稀疏点(偶数位)')
        ax_c.scatter(np.arange(1, 100, 2), lorenz_data_imputed[1::2, pred_target_idx],
                     s=20, c='tomato', marker='+', zorder=5, label='CSDI补出点(奇数位)')
        ax_c.axvline(x=imp_trainlength, color='gray', ls=':', lw=1.5,
                     label=f'训练/预测分界 (pos={imp_trainlength})')
        ax_c.set_xlabel('100点序列位置（偶=已知，奇=补值；步长=4原始步）')
        ax_c.set_ylabel(f'维度 {pred_target_idx}')
        ax_c.set_title('② CSDI补值质量：100点真实值 vs 补值结果\n'
                       '（左侧60步=训练区域；右侧40步=预测区域）')
        ax_c.legend(fontsize=8, ncol=3)
        ax_c.grid(True, alpha=0.3)

        # ── Panel D: 补值预测 vs 稀疏预测（同时间点对比）──
        ax_d = fig.add_subplot(gs[2, 0])
        ax_d.plot(t_ori, gt_ori, 'k-', lw=2, label='真实值（稀疏时间点）')
        # 稀疏预测
        ax_d.plot(t_ori, result_rde_ori[0], 'b-', lw=1.5,
                  alpha=0.6, label='RDE（稀疏原始）')
        ax_d.plot(t_ori, result_rdedel_ori[0], 'r-', lw=1.5,
                  alpha=0.6, label='RDE-Delay（稀疏原始）')
        # 补值预测（对应点）
        ax_d.plot(t_ori, result_rde_imp[0][cmp_idx], 'b--', lw=1.8,
                  label='RDE（补值→对应20步）')
        ax_d.plot(t_ori, result_rdedel_imp[0][cmp_idx], 'r--', lw=1.8,
                  label='RDE-Delay（补值→对应20步）')
        ax_d.set_xlabel('稀疏步')
        ax_d.set_ylabel(f'维度 {pred_target_idx}')
        ax_d.set_title('③④ 相同时间点对比：\n稀疏预测 vs 补值预测（各取20步）')
        ax_d.legend(fontsize=7)
        ax_d.grid(True, alpha=0.3)

        # ── Panel E: RMSE 柱状图对比 ─────────────────────
        ax_e = fig.add_subplot(gs[2, 1])
        methods = ['RDE\n(稀疏)', 'RDE-Delay\n(稀疏)',
                   'RDE\n(补值对应)', 'RDE-Delay\n(补值对应)']
        rms_vals = [np.sqrt(np.mean(err_ori**2)),
                    np.sqrt(np.mean(err_del_ori**2)),
                    np.sqrt(np.mean(err_imp_cmp**2)),
                    np.sqrt(np.mean(err_del_imp_cmp**2))]
        colors = ['#4C72B0', '#DD8452', '#4C72B0', '#DD8452']
        hatches = ['', '', '///', '///']
        bars = ax_e.bar(methods, rms_vals, color=colors, alpha=0.75,
                        hatch=hatches, edgecolor='white')
        for bar, v in zip(bars, rms_vals):
            ax_e.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(rms_vals) * 0.02,
                      f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor='gray', label='稀疏原始'),
                      Patch(facecolor='gray', hatch='///', label='CSDI补值')]
        ax_e.legend(handles=legend_els, fontsize=8)
        ax_e.set_ylabel('RMSE')
        ax_e.set_title('RMSE 对比（相同20个时间点）')
        ax_e.grid(True, alpha=0.3, axis='y')
    else:
        # 没有补值结果时，仅画 RMSE 对比
        ax_d = fig.add_subplot(gs[1, 0])
        methods   = ['RDE\n(稀疏)', 'RDE-Delay\n(稀疏)']
        rms_vals  = [np.sqrt(np.mean(err_ori**2)),
                     np.sqrt(np.mean(err_del_ori**2))]
        ax_d.bar(methods, rms_vals, color=['#4C72B0', '#DD8452'], alpha=0.75)
        ax_d.set_ylabel('RMSE')
        ax_d.set_title('RMSE 对比')
        ax_d.grid(True, alpha=0.3, axis='y')

    fig_path = os.path.join(output_dir, f'full_comparison_{timestamp}.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n综合对比图已保存: {fig_path}")

    # ════════════════════════════════════════════════════════
    # Step 7: 文本摘要
    # ════════════════════════════════════════════════════════
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CSDI Imputation + RDE/RDE-Delay Prediction — Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"实验时间: {timestamp}\n\n")
        f.write("数据参数:\n")
        f.write(f"  Lorenz N={lorenz_N}, L={lorenz_L}, stepsize={lorenz_stepsize}\n")
        f.write(f"  完整数据: {full_data.shape}   稀疏数据: {lorenz_data.shape}\n")
        if lorenz_data_imputed is not None:
            f.write(f"  补值数据: {lorenz_data_imputed.shape}\n")
        f.write("\n预测参数:\n")
        f.write(f"  稀疏预测: trainlength={pred_trainlength}, 预测{n_pred_ori}步\n")
        f.write(f"  补值预测: trainlength={imp_trainlength}, 预测{n_pred_imp}步, "
                f"取偶数偏移 [0,2,...,{2*(n_pred_ori-1)}] 作对比\n")
        f.write(f"  RDE: L={pred_L}, s={pred_s}\n")
        f.write(f"  RDE-Delay: max_delay={pred_max_delay}, M={pred_M}, "
                f"num_samples={pred_num_samples}\n\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'方法':<24} {'RMSE':<12} {'MaxErr':<12} {'2σ覆盖率(%)':<14}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'RDE（稀疏原始）':<24} "
                f"{np.sqrt(np.mean(err_ori**2)):<12.4f} "
                f"{np.max(np.abs(err_ori)):<12.4f} "
                f"{cov_ori:<14.1f}\n")
        f.write(f"{'RDE-Delay（稀疏原始）':<24} "
                f"{np.sqrt(np.mean(err_del_ori**2)):<12.4f} "
                f"{np.max(np.abs(err_del_ori)):<12.4f} "
                f"{cov_del_ori:<14.1f}\n")
        if result_rde_imp is not None:
            f.write(f"{'RDE（补值→对应20步）':<24} "
                    f"{np.sqrt(np.mean(err_imp_cmp**2)):<12.4f} "
                    f"{np.max(np.abs(err_imp_cmp)):<12.4f} "
                    f"{c3:<14.1f}\n")
            f.write(f"{'RDE-Delay（补值→对应20步）':<24} "
                    f"{np.sqrt(np.mean(err_del_imp_cmp**2)):<12.4f} "
                    f"{np.max(np.abs(err_del_imp_cmp)):<12.4f} "
                    f"{c4:<14.1f}\n")
            if lorenz_data_imputed is not None and gt_100 is not None:
                err_imp_only = lorenz_data_imputed[1::2] - gt_100[1::2]
                f.write(f"\n补值质量（仅补出的50点）:\n")
                f.write(f"  RMSE = {np.sqrt(np.mean(err_imp_only**2)):.4f}\n")
                f.write(f"  MAE  = {np.mean(np.abs(err_imp_only)):.4f}\n")
        f.write("=" * 80 + "\n\n")
        f.write("输出文件:\n")
        f.write(f"  综合对比图  : {fig_path}\n")
        if impute_fig_path:
            f.write(f"  补值质量图  : {impute_fig_path}\n")
        f.write(f"  摘要        : {summary_path}\n")

    print(f"摘要已保存: {summary_path}")
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

