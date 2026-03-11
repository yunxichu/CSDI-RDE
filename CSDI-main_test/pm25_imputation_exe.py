# -*- coding: utf-8 -*-模型训练主程序，用于PM2.5数据的补值任务（随机时间划分版本）
"""
CSDI PM2.5 补值主程序（随机时间划分版本）

修改说明：
1. 使用新的随机时间划分数据集
2. 移除validationindex参数（不再按月份划分）
3. 增加数据集划分相关参数
"""

# 导入命令行参数解析库
import argparse
# 导入PyTorch深度学习框架
import torch
# 导入日期时间处理库
import datetime
# 导入JSON数据处理库
import json
# 导入YAML配置文件解析库
import yaml
# 导入操作系统接口库
import os

# 从新的数据集模块导入PM2.5数据加载器
from pm25_imputation_dataset import get_dataloader
# 从自定义模块导入CSDI模型（用于PM2.5数据的条件扩散模型）
from main_model import CSDI_PM25
# 从自定义模块导入训练和评估函数
from utils import train, evaluate

# 创建命令行参数解析器，程序描述为"CSDI"
parser = argparse.ArgumentParser(description="CSDI - PM2.5 Imputation with Random Time Split")

# ========== 基础参数 ==========
parser.add_argument(
    "--config", 
    type=str, 
    default="base.yaml",
    help="配置文件路径"
)
parser.add_argument(
    '--device', 
    default='cuda:0', 
    help='训练设备 (例如: cuda:0, cuda:1, cpu)'
)
parser.add_argument(
    "--modelfolder", 
    type=str, 
    default="",
    help="预训练模型文件夹路径（为空则从头训练）"
)

# ========== 模型参数 ==========
parser.add_argument(
    "--targetstrategy", 
    type=str, 
    default="mix", 
    choices=["mix", "random", "historical"],
    help="目标mask策略"
)
parser.add_argument(
    "--nsample", 
    type=int, 
    default=100,
    help="测试时的采样次数"
)
parser.add_argument(
    "--unconditional", 
    action="store_true",
    help="是否使用无条件生成模式"
)

# ========== 数据集划分参数（新增） ==========
parser.add_argument(
    "--split_ratio", 
    type=float, 
    default=0.5,
    help="时间切分比例：前split_ratio用于补值，后(1-split_ratio)用于预测 (默认: 0.5)"
)
parser.add_argument(
    "--train_ratio", 
    type=float, 
    default=0.7,
    help="训练集比例（在前半部分数据中）(默认: 0.7)"
)
parser.add_argument(
    "--valid_ratio", 
    type=float, 
    default=0.15,
    help="验证集比例（在前半部分数据中）(默认: 0.15)"
)
parser.add_argument(
    "--missing_ratio", 
    type=float, 
    default=0.1,
    help="训练时人工mask的缺失比例（CSDI方法）(默认: 0.1)"
)
parser.add_argument(
    "--seed", 
    type=int, 
    default=42,
    help="随机种子（用于可复现的数据划分）(默认: 42)"
)

# 解析命令行参数
args = parser.parse_args()

# 打印分隔线和参数信息
print("=" * 80)
print("CSDI PM2.5 补值训练 - 随机时间划分版本")
print("=" * 80)
print("\n命令行参数:")
print(json.dumps(vars(args), indent=4))

# ========== 加载配置文件 ==========
config_path = "config/" + args.config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 根据命令行参数设置模型配置
config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

print("\n配置文件内容:")
print(json.dumps(config, indent=4))

# ========== 创建模型保存文件夹 ==========
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 构建文件夹名称（包含关键参数信息）
foldername = (
    f"./save/pm25_imputation_"
    f"split{args.split_ratio}_"
    f"train{args.train_ratio}_"
    f"valid{args.valid_ratio}_"
    f"mask{args.missing_ratio}_"
    f"seed{args.seed}_"
    f"{current_time}/"
)

print(f"\n模型保存路径: {foldername}")
os.makedirs(foldername, exist_ok=True)

# 保存完整配置（包括命令行参数和配置文件）
full_config = {
    "args": vars(args),
    "model_config": config
}
with open(foldername + "config.json", "w") as f:
    json.dump(full_config, f, indent=4)

# ========== 加载数据 ==========
print("\n" + "=" * 80)
print("加载数据集")
print("=" * 80)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    batch_size=config["train"]["batch_size"],
    device=args.device,
    missing_ratio=args.missing_ratio,
    split_ratio=args.split_ratio,
    train_ratio=args.train_ratio,
    valid_ratio=args.valid_ratio,
    seed=args.seed
)

print(f"\n数据集统计:")
print(f"  训练集: {len(train_loader.dataset)} 个样本, {len(train_loader)} 个batch")
print(f"  验证集: {len(valid_loader.dataset)} 个样本, {len(valid_loader)} 个batch")
print(f"  测试集: {len(test_loader.dataset)} 个样本, {len(test_loader)} 个batch")

# ========== 创建模型 ==========
print("\n" + "=" * 80)
print("创建模型")
print("=" * 80)

model = CSDI_PM25(config, args.device).to(args.device)

# 统计模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n模型参数统计:")
print(f"  总参数量: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")

# ========== 训练或加载模型 ==========
print("\n" + "=" * 80)

if args.modelfolder == "":
    # 从头开始训练
    print("开始训练")
    print("=" * 80)
    
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
    
    print("\n训练完成!")
else:
    # 加载预训练模型
    print("加载预训练模型")
    print("=" * 80)
    
    model_path = "./save/" + args.modelfolder + "/model.pth"
    print(f"模型路径: {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    print("模型加载成功!")

# ========== 评估模型 ==========
print("\n" + "=" * 80)
print("在测试集上评估模型")
print("=" * 80)

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)

print("\n" + "=" * 80)
print("程序执行完成!")
print("=" * 80)
print(f"\n结果保存在: {foldername}")
