#!/bin/bash
# PM2.5 预测网格搜索 - 快速测试版
# 仅测试少量参数组合，用于验证脚本是否正常工作

echo "=========================================="
echo "PM2.5 网格搜索 - 快速测试"
echo "=========================================="
echo ""
echo "该脚本将测试以下参数组合："
echo "  L: 5, 11 (2个值)"
echo "  trainlength: 200, 400 (2个值)"
echo "  总计: 4 种组合"
echo ""
echo "预计运行时间: 10-20分钟"
echo ""

# 检查必需参数
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "用法: $0 <history_imputed.csv路径> <ground.txt路径>"
    echo ""
    echo "示例:"
    echo "  $0 ./data/history_imputed.csv ./data/pm25_ground.txt"
    exit 1
fi

IMPUTED_PATH="$1"
GROUND_PATH="$2"

# 检查文件是否存在
if [ ! -f "$IMPUTED_PATH" ]; then
    echo "错误: 找不到文件 $IMPUTED_PATH"
    exit 1
fi

if [ ! -f "$GROUND_PATH" ]; then
    echo "错误: 找不到文件 $GROUND_PATH"
    exit 1
fi

echo "输入文件验证通过"
echo "  - history_imputed: $IMPUTED_PATH"
echo "  - ground: $GROUND_PATH"
echo ""

# 创建快速测试脚本（临时Python脚本，修改参数范围）
cat > /tmp/pm25_quick_test.py << 'EOFPYTHON'
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '/home/claude')
from pm25_grid_search import *

# 覆盖网格搜索函数，使用更小的参数范围
def quick_run_grid_search(args):
    set_global_seed(args.seed)
    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_out_dir = f"./save/pm25_quick_test_{now}/"
    ensure_dir(main_out_dir)
    
    safe_json_dump(vars(args), os.path.join(main_out_dir, "args.json"))
    
    print("=" * 80)
    print("PM2.5 预测网格搜索 - 快速测试版")
    print("=" * 80)
    
    print("\n读取数据...")
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True).sort_index()
    hist_full, fut_full, meta = time_split_df(df_full, args.split_ratio)
    
    df_hist_imputed = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index()
    
    assert_or_raise(df_hist_imputed.index.equals(hist_full.index), "索引不一致")
    assert_or_raise(list(df_hist_imputed.columns) == list(hist_full.columns), "列不一致")
    
    history = df_hist_imputed.values.astype(np.float64)
    assert_or_raise(not np.isnan(history).any(), "history包含NaN")
    
    full_horizon = meta["fut_len"]
    horizon = full_horizon
    if args.horizon_days and args.horizon_days > 0:
        steps_per_day = infer_steps_per_day_from_index(fut_full.index, 24)
        horizon = int(round(args.horizon_days * steps_per_day))
    elif args.horizon_steps and args.horizon_steps > 0:
        horizon = int(args.horizon_steps)
    
    horizon = max(1, min(horizon, full_horizon))
    fut_full = fut_full.iloc[:horizon].copy()
    y_true = fut_full.values.astype(np.float64)
    
    if args.target_indices and args.target_indices.strip():
        target_indices = [int(x) for x in args.target_indices.split(",") if x.strip()]
    else:
        target_indices = None
    
    # 快速测试：只测试2个L值和2个trainlength值
    L_values = [5, 11]
    trainlength_values = [200, 400]
    trainlength_values = [t for t in trainlength_values if t <= len(history)]
    
    print(f"\n快速测试参数:")
    print(f"  L: {L_values}")
    print(f"  trainlength: {trainlength_values}")
    print(f"  总组合数: {len(L_values) * len(trainlength_values)}")
    
    all_results = []
    total = len(L_values) * len(trainlength_values)
    
    with tqdm(total=total, desc="Quick Test Progress") as pbar:
        for L in L_values:
            for trainlength in trainlength_values:
                D = history.shape[1]
                if L > D:
                    pbar.update(1)
                    continue
                
                param_dir = os.path.join(main_out_dir, f"L{L}_trainlen{trainlength}")
                ensure_dir(param_dir)
                
                start_time = time.time()
                
                try:
                    preds, stds = rdegpr_forecast_multivariate(
                        history=history, horizon=horizon, trainlength=trainlength,
                        L=L, s=args.s, steps_ahead=args.steps_ahead,
                        n_jobs=args.n_jobs, seed=args.seed,
                        noise_strength=args.noise_strength,
                        optimize_hyp=(not args.no_optimize_hyp),
                        target_indices=target_indices,
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    df_pred = pd.DataFrame(preds, index=fut_full.index, columns=df_full.columns)
                    df_std = pd.DataFrame(stds, index=fut_full.index, columns=df_full.columns)
                    df_pred.to_csv(os.path.join(param_dir, "future_pred.csv"))
                    df_std.to_csv(os.path.join(param_dir, "future_pred_std.csv"))
                    
                    overall = compute_metrics(y_true, preds)
                    
                    per_dim = []
                    for j, col in enumerate(df_full.columns):
                        m = compute_metrics(y_true[:, j], preds[:, j])
                        per_dim.append({"dim": j, "name": str(col), "rmse": m["rmse"], "mae": m["mae"]})
                    pd.DataFrame(per_dim).to_csv(os.path.join(param_dir, "metrics_per_dim.csv"), index=False)
                    
                    save_plots(param_dir, fut_full.index, y_true, preds, L, trainlength, [0, 1])
                    
                    result = {
                        "L": L, "trainlength": trainlength,
                        "rmse": overall["rmse"], "mae": overall["mae"],
                        "valid_points": overall["valid_points"],
                        "elapsed_time": elapsed_time,
                        "status": "success", "output_dir": param_dir
                    }
                    safe_json_dump(result, os.path.join(param_dir, "result.json"))
                    
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    result = {
                        "L": L, "trainlength": trainlength,
                        "rmse": np.nan, "mae": np.nan, "valid_points": 0,
                        "elapsed_time": elapsed_time,
                        "status": "failed", "error": str(e), "output_dir": param_dir
                    }
                    safe_json_dump(result, os.path.join(param_dir, "result.json"))
                
                all_results.append(result)
                pbar.update(1)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(main_out_dir, "quick_test_results.csv"), index=False)
    
    successful = results_df[results_df['status'] == 'success']
    
    if len(successful) > 0:
        best_rmse = successful.loc[successful['rmse'].idxmin()]
        summary = {
            "total_combinations": total,
            "successful_runs": len(successful),
            "failed_runs": len(results_df) - len(successful),
            "best_rmse": {
                "L": int(best_rmse['L']),
                "trainlength": int(best_rmse['trainlength']),
                "rmse": float(best_rmse['rmse']),
                "mae": float(best_rmse['mae'])
            }
        }
        safe_json_dump(summary, os.path.join(main_out_dir, "summary.json"))
        
        print("\n" + "=" * 80)
        print("快速测试完成！")
        print("=" * 80)
        print(f"\n成功: {len(successful)}/{total}")
        print(f"\n最优参数:")
        print(f"  L = {summary['best_rmse']['L']}")
        print(f"  trainlength = {summary['best_rmse']['trainlength']}")
        print(f"  RMSE = {summary['best_rmse']['rmse']:.4f}")
        print(f"\n结果保存在: {main_out_dir}")
    else:
        print("\n所有运行都失败了！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon_days", type=float, default=1.0)
    parser.add_argument("--horizon_steps", type=int, default=0)
    parser.add_argument("--s", type=int, default=30)
    parser.add_argument("--steps_ahead", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--no_optimize_hyp", action="store_true")
    parser.add_argument("--target_indices", type=str, default="0,1,2")
    
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    quick_run_grid_search(args)
EOFPYTHON

echo "开始运行快速测试..."
echo ""

# 运行快速测试（使用较小的参数以加快速度）
python /tmp/pm25_quick_test.py \
  --imputed_history_path "$IMPUTED_PATH" \
  --ground_path "$GROUND_PATH" \
  --split_ratio 0.5 \
  --horizon_days 1.0 \
  --s 30 \
  --n_jobs 4 \
  --target_indices 0,1,2

echo ""
echo "快速测试完成！"
echo ""
echo "如果测试成功，可以运行完整的网格搜索："
echo "  python pm25_grid_search.py \\"
echo "    --imputed_history_path $IMPUTED_PATH \\"
echo "    --ground_path $GROUND_PATH \\"
echo "    --split_ratio 0.5 \\"
echo "    --horizon_days 1 \\"
echo "    --s 50 \\"
echo "    --n_jobs 8 \\"
echo "    --target_indices 0,1,2"
