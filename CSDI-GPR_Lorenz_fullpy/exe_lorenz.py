# exe_lorenz.py
import argparse  # 用于解析命令行参数
import datetime  # 处理日期和时间
import json      # 处理JSON格式数据
import os        # 操作系统接口（文件/目录操作）
import yaml      # 处理YAML配置文件
import torch     # PyTorch深度学习框架
from main_model import CSDI_Lorenz  # 从本地文件导入CSDI Lorenz模型
from dataset_lorenz import get_dataloader  # 导入数据加载器生成函数
from utils import train, evaluate  # 导入训练和评估函数
import numpy as np  # 数值计算库

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="CSDI Lorenz")
# 添加各种命令行参数
parser.add_argument("--config", type=str, default="lorenz.yaml")  # 配置文件路径
parser.add_argument('--device', default='cuda:0', help='Device')  # 指定训练设备
parser.add_argument("--seed", type=int, default=1)  # 随机种子
parser.add_argument("--unconditional", action="store_true")  # 是否使用无条件生成模式
parser.add_argument("--modelfolder", type=str, default="")  # 已存在模型文件夹路径
parser.add_argument("--nsample", type=int, default=100)  # 采样次数
parser.add_argument("--seq_len", type=int, default=100)  # 序列长度
parser.add_argument("--seq_count", type=int, default=1000)  # 总序列数量
parser.add_argument("--batch_size", type=int, default=32)  # 批处理大小
parser.add_argument("--Nnodes", type=int, default=5)  # Lorenz系统节点数

args = parser.parse_args() # 解析命令行参数
print(args) # 打印参数信息

# load config
with open("config/" + args.config, "r") as f:
    config = yaml.safe_load(f) # 安全解析YAML内容

# synchronize a few config fields
config["model"]["is_unconditional"] = args.unconditional
config["train"]["batch_size"] = args.batch_size
config["model"]["seq_len"] = args.seq_len

# 以格式化JSON形式打印配置
print(json.dumps(config, indent=4))

# make save folder 创建模型保存文件夹（使用时间戳命名）
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/lorenz_{current_time}/"
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f: # 将配置保存为JSON文件
    json.dump(config, f, indent=4)

# get dataloaders (API matches physio style)
train_loader, valid_loader, test_loader = get_dataloader(
    batch_size=config["train"]["batch_size"],  # 批大小
    seq_len=args.seq_len,          # 序列长度
    cache_dir=foldername,          # 缓存目录
    N=args.Nnodes,                 # 节点数量
    seq_count=args.seq_count,      # 序列数量
    stepsize=4                     # 步长（可能用于下采样）
)

# init model  初始化CSDI Lorenz模型并移至指定设备（GPU/CPU）
model = CSDI_Lorenz(config, args.device).to(args.device)

# train or load 根据modelfolder参数决定是训练新模型还是加载已有模型
if not args.modelfolder:
    print("Training model...")
    # 执行训练流程
    train(
        model,
        config["train"],      # 训练配置参数
        train_loader,         # 训练数据加载器
        valid_loader=valid_loader,  # 验证数据加载器
        foldername=foldername # 保存路径
    )
    print(f"Model training completed and saved to {foldername}/model.pth")
else:
    # 加载已有模型
    print(f"Loading existing model from ./save/{args.modelfolder}/model.pth")
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    # Also save the loaded model to the current foldername for test.py to use 将加载的模型复制到当前文件夹供后续测试使用
    torch.save(model.state_dict(), foldername + "/model.pth")
    print(f"Model loaded and copied to {foldername}/model.pth")

# 在测试集上评估模型性能
print("Starting evaluation...")
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
print("Evaluation completed!")
# 输出模型和配置的保存路径
print(f"\nModel saved at: {foldername}/model.pth")
print(f"Config saved at: {foldername}/config.json")
# 提供测试脚本的使用示例
print(f"\nTo test the model with test.py, run:")
print(f"python test.py --model_path {foldername}/model.pth --config_path config/{args.config}")
from dataset_lorenz import get_dataloader  