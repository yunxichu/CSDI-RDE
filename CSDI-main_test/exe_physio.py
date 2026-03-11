#exe_physio.py
#预测示例
#python exe_physio.py --modelfolder pretrained --testmissingratio 0.3 --nsample 100 --seed 42
import argparse  # 导入argparse模块用于解析命令行参数
import datetime  # 导入datetime模块用于处理时间和日期
import json  # 导入json模块用于处理JSON文件
import yaml  # 导入yaml模块用于读取YAML配置文件
import os  # 导入os模块用于操作系统相关功能（如创建文件夹）
import torch  # 导入PyTorch深度学习框架
from main_model import CSDI_Physio  # 从main_model文件导入CSDI_Physio模型类
from dataset_physio import get_dataloader  # 从dataset_physio文件导入数据加载器获取函数
from utils import train, evaluate  # 从utils文件导入训练和评估函数

# 解析命令行参数
parser = argparse.ArgumentParser(description="CSDI")  # 创建参数解析器对象，描述为CSDI
parser.add_argument("--config", type=str, default="base.yaml")  # 添加配置文件路径参数，默认为base.yaml
parser.add_argument('--device', default='cuda:0', help='Device')  # 添加计算设备参数，默认为第一个GPU
parser.add_argument("--seed", type=int, default=1)  # 添加随机种子参数，默认为1，用于复现结果
parser.add_argument("--testmissingratio", type=float, default=0.1)  # 添加测试集缺失率参数，默认为0.1
parser.add_argument("--nfold", type=int, default=0, help="5折交叉验证索引")  # 添加交叉验证的折数索引（0-4）
parser.add_argument("--unconditional", action="store_true")  # 添加无条件生成标志，如果设置则为True
parser.add_argument("--modelfolder", type=str, default="")  # 添加模型文件夹参数，若非空则加载预训练模型
parser.add_argument("--nsample", type=int, default=100)  # 添加采样次数参数，默认为100次

args = parser.parse_args()  # 解析命令行参数并将结果赋值给args
print(args)  # 打印解析后的参数信息

# 加载配置文件
with open("config/" + args.config, "r") as f:  # 以只读模式打开指定的YAML配置文件
    config = yaml.safe_load(f)  # 安全加载YAML文件内容并转换为字典

# 更新配置参数
config["model"]["is_unconditional"] = args.unconditional  # 将命令行中的无条件标志更新到配置字典中
config["model"]["test_missing_ratio"] = args.testmissingratio  # 将命令行中的测试缺失率更新到配置字典中
print(json.dumps(config, indent=4))  # 以缩进格式打印更新后的配置字典

# 创建保存文件夹
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间并格式化为字符串
foldername = f"./save/physio_fold{args.nfold}_{current_time}/"  # 根据折数和时间构建保存结果的文件夹路径
print('Model folder:', foldername)  # 打印模型保存文件夹路径
os.makedirs(foldername, exist_ok=True)  # 创建文件夹，如果文件夹已存在则不报错
# 保存配置
with open(foldername + "config.json", "w") as f:  # 在保存文件夹中创建config.json文件
    json.dump(config, f, indent=4)  # 将当前配置写入JSON文件以便后续查看

# 获取数据加载器
train_loader, valid_loader, test_loader = get_dataloader(  # 调用函数获取训练、验证和测试数据加载器
    seed=args.seed,  # 传入随机种子
    nfold=args.nfold,  # 传入交叉验证的折数索引
    batch_size=config["train"]["batch_size"],  # 从配置中获取批量大小
    missing_ratio=config["model"]["test_missing_ratio"],  # 从配置中获取缺失率
)

# 初始化模型
model = CSDI_Physio(config, args.device).to(args.device)  # 实例化CSDI_Physio模型并移动到指定设备（GPU/CPU）

# 训练或加载模型
if not args.modelfolder:  # 如果命令行未指定模型文件夹（表示需要进行训练）
    train(  # 调用训练函数
        model,  # 传入初始化的模型
        config["train"],  # 传入训练相关的配置参数
        train_loader,  # 传入训练数据加载器
        valid_loader=valid_loader,  # 传入验证数据加载器
        foldername=foldername,  # 传入保存结果的文件夹路径
    )
else:  # 如果指定了模型文件夹（表示进行推理/加载预训练模型）
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))  # 加载指定路径下的模型权重

# 评估模型
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)  # 在测试集上评估模型性能并保存结果
