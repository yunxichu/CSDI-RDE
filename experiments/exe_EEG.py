import argparse
import datetime
import json
import os
import yaml
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csdi.main_model import CSDI_EEG
from datasets.dataset_EEG import get_dataloader
from utils import train, evaluate

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="CSDI EEG")
# 添加各种命令行参数
parser.add_argument("--config", type=str, default="eeg.yaml")  # 配置文件路径
parser.add_argument('--device', default='cuda:0', help='Device')  # 指定训练设备
parser.add_argument("--seed", type=int, default=1)  # 随机种子
parser.add_argument("--unconditional", action="store_true")  # 是否使用无条件生成模式
parser.add_argument("--modelfolder", type=str, default="")  # 已存在模型文件夹路径
parser.add_argument("--nsample", type=int, default=100)  # 采样次数
parser.add_argument("--seq_len", type=int, default=100)  # 序列长度
parser.add_argument("--batch_size", type=int, default=32)  # 批处理大小
parser.add_argument("--data_path", type=str, default="/home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx")  # EEG数据路径
parser.add_argument("--valid_ratio", type=float, default=0.2)  # 验证集比例

args = parser.parse_args()  # 解析命令行参数
print(args)  # 打印参数信息

# 设置随机种子以确保结果可重现
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 加载配置文件
with open("config/" + args.config, "r") as f:
    config = yaml.safe_load(f)  # 安全解析YAML内容

# 同步一些配置字段
config["model"]["is_unconditional"] = args.unconditional
config["train"]["batch_size"] = args.batch_size
config["model"]["seq_len"] = args.seq_len

# 以格式化JSON形式打印配置
print(json.dumps(config, indent=4))

# 创建保存文件夹（使用时间戳命名）
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/eeg_{current_time}/"
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:  # 将配置保存为JSON文件
    json.dump(config, f, indent=4)

# 获取数据加载器
print("Loading EEG data...")
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    data_path=args.data_path,
    batch_size=config["train"]["batch_size"],
    eval_length=args.seq_len,
    valid_ratio=args.valid_ratio
)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(valid_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# 初始化模型
print("Initializing model...")
model = CSDI_EEG(config, args.device, target_dim=64).to(args.device)  # EEG有64个通道

# 训练或加载模型
if not args.modelfolder:
    print("Training model...")
    # 执行训练流程
    train(
        model,
        config["train"],      # 训练配置参数
        train_loader,         # 训练数据加载器
        valid_loader=valid_loader,  # 验证数据加载器
        foldername=foldername,  # 保存路径
    )
    print(f"Model training completed and saved to {foldername}/model.pth")
else:
    # 加载已有模型
    print(f"Loading existing model from ./save/{args.modelfolder}/model.pth")
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    # 将加载的模型复制到当前文件夹供后续测试使用
    torch.save(model.state_dict(), foldername + "/model.pth")
    print(f"Model loaded and copied to {foldername}/model.pth")

# 在测试集上评估模型性能
print("Starting evaluation...")
evaluate(
    model, 
    test_loader, 
    nsample=args.nsample, 
    scaler=1, 
    foldername=foldername
)
print("Evaluation completed!")

# 输出模型和配置的保存路径
print(f"\nModel saved at: {foldername}/model.pth")
print(f"Config saved at: {foldername}/config.json")

# 提供测试脚本的使用示例
print(f"\nTo test the model with test.py, run:")
print(f"python test_eeg.py --model_path {foldername}/model.pth --config_path config/{args.config} --data_path /home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx")

