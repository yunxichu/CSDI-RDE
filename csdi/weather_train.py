# -*- coding: utf-8 -*-
"""
Weather CSDI 训练脚本

示例:
python csdi/weather_train.py --config weather --device cuda:0 --seed 42 --missing_ratio 0.1 --missing_mode random
"""

import argparse
import torch
import datetime
import json
import yaml
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'csdi'))

from main_model import CSDI_Weather
from utils import train, evaluate
from datasets.dataset_weather import get_dataloader

parser = argparse.ArgumentParser(description="CSDI for Weather")
parser.add_argument("--config", type=str, default="weather")
parser.add_argument('--device', default='cuda:0')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--missing_ratio", type=float, default=0.1)
parser.add_argument("--missing_mode", type=str, default="random", 
                    choices=["uniform", "random", "temporal", "feature"],
                    help="缺失模式: uniform(均匀), random(随机), temporal(时间块), feature(特征)")
parser.add_argument("--nfold", type=int, default=0)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()

path = os.path.join(project_root, "config", args.config + ".yaml")
if not os.path.exists(path):
    path = os.path.join(project_root, "config", "base.yaml")

with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.missing_ratio
config["train"]["epochs"] = args.epochs
config["train"]["batch_size"] = args.batch_size

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/weather_{args.missing_mode}_ratio{args.missing_ratio}_fold{args.nfold}_{current_time}/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump({
        "config": config,
        "args": vars(args),
    }, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    batch_size=config["train"]["batch_size"],
    device=args.device,
    missing_ratio=args.missing_ratio,
    seed=args.seed,
    missing_mode=args.missing_mode,
    use_generated_missing=True,
)

model = CSDI_Weather(config, args.device, target_dim=21).to(args.device)

train(
    model,
    config["train"],
    train_loader,
    valid_loader=valid_loader,
    foldername=foldername,
)

evaluate(model, test_loader, nsample=args.nsample, scaler=scaler, mean_scaler=mean_scaler, foldername=foldername)
