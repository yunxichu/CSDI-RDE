# -*- coding: utf-8 -*-
"""
EEG 预测方法对比脚本

对比 RDE-GPR 与 基线方法（GRU/LSTM）的预测效果
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    rmse = np.sqrt(np.mean((y_true_m - y_pred_m) ** 2))
    mae = np.mean(np.abs(y_true_m - y_pred_m))
    return {"rmse": rmse, "mae": mae}

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    print(result.stdout)
    return result.stdout

def load_results(folder, method_name):
    pred_file = os.path.join(folder, "future_pred.csv")
    if os.path.exists(pred_file):
        preds = pd.read_csv(pred_file).values
        return preds
    return None

def plot_comparison(results_dict, ground_truth, target_dims, out_dir):
    n_dims = len(target_dims)
    horizon = ground_truth.shape[0]

    fig, axes = plt.subplots(n_dims, 2, figsize=(16, 4 * n_dims))

    if n_dims == 1:
        axes = axes.reshape(1, -1)

    colors = {
        'RDE-GPR': '#e74c3c',
        'GRU': '#3498db',
        'LSTM': '#2ecc71',
        'NeuralCDE': '#9b59b6',
        'GRU-ODE-Bayes': '#f39c12',
    }

    for i, dim in enumerate(target_dims):
        ax_left = axes[i, 0]
        ax_right = axes[i, 1]

        ax_left.plot(ground_truth[:, i], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)

        for method, result in results_dict.items():
            if result is not None and not np.isnan(result['rmse']):
                preds = result['predictions'][:, i]
                ax_left.plot(preds, '--', color=colors.get(method, None), linewidth=1.5,
                           label=f'{method} (RMSE={result["rmse"]:.2f})', alpha=0.7)

        ax_left.set_xlabel('Time Step')
        ax_left.set_ylabel('Value')
        ax_left.set_title(f'Channel {dim} - Forecast Comparison')
        ax_left.legend(loc='upper right', fontsize=8)
        ax_left.grid(True, alpha=0.3)

        errors = {}
        for method, result in results_dict.items():
            if result is not None and not np.isnan(result['rmse']):
                errors[method] = np.abs(ground_truth[:, i] - result['predictions'][:, i])

        if errors:
            for method, error in errors.items():
                ax_right.plot(error, '-', color=colors.get(method, None),
                            linewidth=1.5, label=method, alpha=0.7)
            ax_right.set_xlabel('Time Step')
            ax_right.set_ylabel('Absolute Error')
            ax_right.set_title(f'Channel {dim} - Prediction Error')
            ax_right.legend(loc='upper right', fontsize=8)
            ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eeg_forecast_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved: {os.path.join(out_dir, 'eeg_forecast_comparison.png')}")

    fig, ax = plt.subplots(figsize=(10, 6))
    methods = []
    rmses = []
    maes = []

    for method, result in results_dict.items():
        if result is not None and not np.isnan(result['rmse']):
            methods.append(method)
            rmses.append(result['rmse'])
            maes.append(result['mae'])

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, rmses, width, label='RMSE', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, maes, width, label='MAE', color='#3498db', alpha=0.8)

    ax.set_xlabel('Method')
    ax.set_ylabel('Error')
    ax.set_title('EEG Forecast Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'eeg_forecast_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved: {os.path.join(out_dir, 'eeg_forecast_metrics.png')}")

def main():
    parser = argparse.ArgumentParser(description="EEG Forecast Comparison")
    parser.add_argument("--imputed_path", type=str,
                       default="./save/eeg_imputed_random_ratio0.5_seed42_20260331_131907/eeg_imputed.npy")
    parser.add_argument("--ground_path", type=str,
                       default="./data/eeg/eeg_ground.npy")
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=100)
    parser.add_argument("--target_dims", type=str, default="0,1,2")
    parser.add_argument("--rdegpr_L", type=int, default=7)
    parser.add_argument("--rdegpr_s", type=int, default=50)
    parser.add_argument("--rdegpr_trainlength", type=int, default=100)
    parser.add_argument("--rdegpr_max_delay", type=int, default=20)
    parser.add_argument("--gru_epochs", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="./save/eeg_comparison")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_teacher_forcing", action="store_true", help="Use ground truth for sliding window (single-step rolling prediction)")
    parser.add_argument("--use_ground_truth_train", action="store_true", help="Use ground truth for training (fair comparison with RDE-GPR)")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    target_dims = [int(x) for x in args.target_dims.split(',')]

    ground_truth = np.load(args.ground_path)
    if ground_truth.ndim == 3:
        ground_truth = ground_truth.reshape(-1, ground_truth.shape[-1])
    ground_truth = ground_truth[args.history_timesteps:args.history_timesteps + args.horizon_steps]
    if target_dims:
        ground_truth = ground_truth[:, target_dims]

    results = {}

    rdegpr_dir = os.path.join(args.out_dir, "rdegpr")
    ensure_dir(rdegpr_dir)
    cmd = [
        "python", "rde_gpr/eeg_CSDIimpute_after-RDEgpr.py",
        "--imputed_path", args.imputed_path,
        "--ground_path", args.ground_path,
        "--horizon_steps", str(args.horizon_steps),
        "--L", str(args.rdegpr_L),
        "--s", str(args.rdegpr_s),
        "--trainlength", str(args.rdegpr_trainlength),
        "--n_jobs", str(args.n_jobs),
        "--use_delay_embedding",
        "--max_delay", str(args.rdegpr_max_delay),
        "--target_indices", args.target_dims,
        "--out_dir", rdegpr_dir
    ]
    run_command(cmd, "RDE-GPR Forecast")
    rdegpr_preds = load_results(rdegpr_dir, "RDE-GPR")
    if rdegpr_preds is not None:
        preds_aligned = rdegpr_preds[:, target_dims]
        m = compute_metrics(ground_truth, preds_aligned)
        results["RDE-GPR"] = {"predictions": preds_aligned, **m}
        print(f"RDE-GPR - RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}")

    gru_dir = os.path.join(args.out_dir, "gru")
    ensure_dir(gru_dir)
    cmd = [
        "python", "baselines/eeg_baseline_forecast.py",
        "--imputed_path", args.imputed_path,
        "--ground_path", args.ground_path,
        "--horizon_steps", str(args.horizon_steps),
        "--history_timesteps", str(args.history_timesteps),
        "--hidden_size", "64",
        "--num_layers", "2",
        "--epochs", str(args.gru_epochs),
        "--batch_size", "64",
        "--lr", "1e-3",
        "--model", "gru",
        "--train_window", "48",
        "--target_dims", args.target_dims,
        "--out_dir", gru_dir,
        "--device", args.device
    ]
    if args.use_teacher_forcing:
        cmd.append("--use_teacher_forcing")
    if args.use_ground_truth_train:
        cmd.append("--use_ground_truth_train")
    run_command(cmd, "GRU Baseline Forecast")
    gru_preds = load_results(gru_dir, "GRU")
    if gru_preds is not None:
        preds_aligned = gru_preds[:, target_dims]
        m = compute_metrics(ground_truth, preds_aligned)
        results["GRU"] = {"predictions": preds_aligned, **m}
        print(f"GRU - RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}")

    lstm_dir = os.path.join(args.out_dir, "lstm")
    ensure_dir(lstm_dir)
    cmd = [
        "python", "baselines/eeg_baseline_forecast.py",
        "--imputed_path", args.imputed_path,
        "--ground_path", args.ground_path,
        "--horizon_steps", str(args.horizon_steps),
        "--history_timesteps", str(args.history_timesteps),
        "--hidden_size", "64",
        "--num_layers", "2",
        "--epochs", str(args.gru_epochs),
        "--batch_size", "64",
        "--lr", "1e-3",
        "--model", "lstm",
        "--train_window", "48",
        "--target_dims", args.target_dims,
        "--out_dir", lstm_dir,
        "--device", args.device
    ]
    if args.use_teacher_forcing:
        cmd.append("--use_teacher_forcing")
    if args.use_ground_truth_train:
        cmd.append("--use_ground_truth_train")
    run_command(cmd, "LSTM Baseline Forecast")
    lstm_preds = load_results(lstm_dir, "LSTM")
    if lstm_preds is not None:
        preds_aligned = lstm_preds[:, target_dims]
        m = compute_metrics(ground_truth, preds_aligned)
        results["LSTM"] = {"predictions": preds_aligned, **m}
        print(f"LSTM - RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}")

    neuralcde_dir = os.path.join(args.out_dir, "neuralcde")
    ensure_dir(neuralcde_dir)
    cmd = [
        "python", "baselines/eeg_neuralcde_forecast.py",
        "--imputed_path", args.imputed_path,
        "--ground_path", args.ground_path,
        "--horizon_steps", str(args.horizon_steps),
        "--history_timesteps", str(args.history_timesteps),
        "--hidden_channels", "64",
        "--epochs", str(args.gru_epochs),
        "--lr", "1e-3",
        "--target_dims", args.target_dims,
        "--out_dir", neuralcde_dir,
        "--device", args.device
    ]
    if args.use_teacher_forcing:
        cmd.append("--use_teacher_forcing")
    if args.use_ground_truth_train:
        cmd.append("--use_ground_truth_train")
    run_command(cmd, "NeuralCDE Forecast")
    neuralcde_preds = load_results(neuralcde_dir, "NeuralCDE")
    if neuralcde_preds is not None:
        preds_aligned = neuralcde_preds[:, target_dims]
        m = compute_metrics(ground_truth, preds_aligned)
        results["NeuralCDE"] = {"predictions": preds_aligned, **m}
        print(f"NeuralCDE - RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}")

    gruode_dir = os.path.join(args.out_dir, "gruodebayes")
    ensure_dir(gruode_dir)
    cmd = [
        "python", "baselines/eeg_gruodebayes_forecast.py",
        "--imputed_path", args.imputed_path,
        "--ground_path", args.ground_path,
        "--horizon_steps", str(args.horizon_steps),
        "--history_timesteps", str(args.history_timesteps),
        "--hidden_size", "64",
        "--epochs", str(args.gru_epochs),
        "--lr", "1e-3",
        "--target_dims", args.target_dims,
        "--out_dir", gruode_dir,
        "--device", args.device
    ]
    if args.use_teacher_forcing:
        cmd.append("--use_teacher_forcing")
    if args.use_ground_truth_train:
        cmd.append("--use_ground_truth_train")
    run_command(cmd, "GRU-ODE-Bayes Forecast")
    gruode_preds = load_results(gruode_dir, "GRU-ODE-Bayes")
    if gruode_preds is not None:
        preds_aligned = gruode_preds[:, target_dims]
        m = compute_metrics(ground_truth, preds_aligned)
        results["GRU-ODE-Bayes"] = {"predictions": preds_aligned, **m}
        print(f"GRU-ODE-Bayes - RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}")

    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)

    summary = []
    for method, result in results.items():
        if result is not None:
            summary.append({
                "Method": method,
                "RMSE": result['rmse'],
                "MAE": result['mae']
            })
            print(f"{method:15s} - RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "comparison_summary.csv"), index=False)

        plot_comparison(results, ground_truth, target_dims, args.out_dir)

        all_results = {}
        for method, result in results.items():
            if result is not None:
                all_results[method] = {
                    "rmse": float(result['rmse']),
                    "mae": float(result['mae']),
                    "predictions": result['predictions'].tolist() if result['predictions'] is not None else None
                }

        with open(os.path.join(args.out_dir, "comparison_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)

    print(f"\nResults saved to: {args.out_dir}")

if __name__ == "__main__":
    main()