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

# 从自定义模块导入PM2.5数据加载器
from dataset_pm25 import get_dataloader
# 从自定义模块导入CSDI模型（用于PM2.5数据的条件扩散模型）
from main_model import CSDI_PM25
# 从自定义模块导入训练和评估函数
from utils import train, evaluate

# 创建命令行参数解析器，程序描述为"CSDI"
parser = argparse.ArgumentParser(description="CSDI")
# 添加配置文件路径参数，默认为"base.yaml"
parser.add_argument("--config", type=str, default="base.yaml")
# 添加设备参数，默认使用第一块GPU
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
# 添加模型文件夹路径参数，默认为空字符串
parser.add_argument("--modelfolder", type=str, default="")
# 添加目标策略参数，可选值为"mix"、"random"或"historical"，默认为"mix"
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)
# 添加验证集索引参数，用于指定用于验证的月份索引（取值范围：0-7）
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)
# 添加采样数量参数，默认为100
parser.add_argument("--nsample", type=int, default=100)
# 添加无条件标志参数，如果指定则为无条件生成模式
parser.add_argument("--unconditional", action="store_true")

# 解析命令行参数
args = parser.parse_args()
# 打印解析后的参数
print(args)

# 构建配置文件的完整路径
path = "config/" + args.config
# 以只读模式打开配置文件
with open(path, "r") as f:
    # 使用YAML安全加载器解析配置文件
    config = yaml.safe_load(f)

# 根据命令行参数设置模型是否为无条件模式
config["model"]["is_unconditional"] = args.unconditional
# 根据命令行参数设置目标策略
config["model"]["target_strategy"] = args.targetstrategy

# 以格式化的JSON格式打印配置信息，缩进为4个空格
print(json.dumps(config, indent=4))

# 获取当前时间并格式化为"年月日_时分秒"格式
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
# 构建保存模型的文件夹路径，包含验证集索引和时间戳
foldername = (
    "./save/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/"
)

# 打印模型保存文件夹路径
print('model folder:', foldername)
# 创建模型保存文件夹，如果已存在则不报错
os.makedirs(foldername, exist_ok=True)
# 将配置信息保存为JSON文件到模型文件夹中
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# 获取训练、验证、测试数据加载器以及标准化器
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=args.device, validindex=args.validationindex
)
# 创建CSDI_PM25模型实例并移动到指定设备
model = CSDI_PM25(config, args.device).to(args.device)

# 如果未指定预训练模型文件夹
if args.modelfolder == "":
    # 从头开始训练模型
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
# 如果指定了预训练模型文件夹
else:
    # 加载预训练模型的权重
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

# 在测试集上评估模型性能
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
