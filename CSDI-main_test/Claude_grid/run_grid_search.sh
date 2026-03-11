#!/bin/bash
# PM2.5 预测网格搜索 - 完整版运行脚本
# 
# 网格参数：
#   L: 5, 8, 11, 14, 17, 20 (6个值)
#   trainlength: 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000 (10个值)
#   总计: 60 种组合
#
# 预计运行时间: 5-10小时（取决于硬件配置）

echo "=========================================="
echo "PM2.5 网格搜索 - 完整版"
echo "=========================================="
echo ""
echo "该脚本将测试60种参数组合"
echo "预计运行时间: 5-10小时"
echo ""

# 检查必需参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <history_imputed.csv路径> <ground.txt路径> [可选参数]"
    echo ""
    echo "必需参数:"
    echo "  1. history_imputed.csv路径"
    echo "  2. ground.txt路径"
    echo ""
    echo "可选参数（在位置参数后添加）:"
    echo "  --split_ratio <值>       数据划分比例 (默认: 0.5)"
    echo "  --horizon_days <值>      预测天数 (默认: 1.0)"
    echo "  --s <值>                 每步抽样数 (默认: 50)"
    echo "  --n_jobs <值>            并行进程数 (默认: 8)"
    echo "  --target_indices <值>    目标维度 (默认: 0,1,2)"
    echo "  --out_dir <路径>         输出目录 (默认: 自动生成)"
    echo ""
    echo "示例:"
    echo "  $0 ./data/history_imputed.csv ./data/pm25_ground.txt"
    echo "  $0 ./data/history_imputed.csv ./data/pm25_ground.txt --n_jobs 16 --s 100"
    exit 1
fi

IMPUTED_PATH="$1"
GROUND_PATH="$2"
shift 2  # 移除前两个位置参数

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

# 默认参数
SPLIT_RATIO=0.5
HORIZON_DAYS=1.0
S_VALUE=50
N_JOBS=8
TARGET_INDICES="0,1,2"
OUT_DIR=""
SEED=42
NOISE=0.0

# 解析可选参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --split_ratio)
            SPLIT_RATIO="$2"
            shift 2
            ;;
        --horizon_days)
            HORIZON_DAYS="$2"
            shift 2
            ;;
        --s)
            S_VALUE="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --target_indices)
            TARGET_INDICES="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --noise_strength)
            NOISE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "运行参数:"
echo "  split_ratio = $SPLIT_RATIO"
echo "  horizon_days = $HORIZON_DAYS"
echo "  s (抽样数) = $S_VALUE"
echo "  n_jobs (并行) = $N_JOBS"
echo "  target_indices = $TARGET_INDICES"
echo "  seed = $SEED"
echo "  noise_strength = $NOISE"
if [ -n "$OUT_DIR" ]; then
    echo "  out_dir = $OUT_DIR"
fi
echo ""

# 确认运行
read -p "确认开始运行？这可能需要5-10小时。(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消运行"
    exit 0
fi

echo ""
echo "开始网格搜索..."
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 构建命令
CMD="python pm25_grid_search.py \
  --imputed_history_path \"$IMPUTED_PATH\" \
  --ground_path \"$GROUND_PATH\" \
  --split_ratio $SPLIT_RATIO \
  --horizon_days $HORIZON_DAYS \
  --s $S_VALUE \
  --steps_ahead 1 \
  --n_jobs $N_JOBS \
  --seed $SEED \
  --noise_strength $NOISE \
  --target_indices \"$TARGET_INDICES\""

if [ -n "$OUT_DIR" ]; then
    CMD="$CMD --out_dir \"$OUT_DIR\""
fi

# 记录命令到日志
echo "执行命令:" > grid_search.log
echo "$CMD" >> grid_search.log
echo "" >> grid_search.log

# 运行并同时输出到终端和日志
eval "$CMD" 2>&1 | tee -a grid_search.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "网格搜索成功完成！"
    echo "=========================================="
    echo ""
    echo "日志已保存到: grid_search.log"
    echo ""
    echo "下一步:"
    echo "  1. 查看 summary.json 了解最优参数"
    echo "  2. 查看热力图: heatmap_rmse.png, heatmap_mae.png"
    echo "  3. 查看趋势图: trend_plots.png"
    echo "  4. 检查最优参数组的详细结果"
else
    echo "=========================================="
    echo "网格搜索失败，退出码: $EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "请查看 grid_search.log 了解详细错误信息"
    exit $EXIT_CODE
fi
