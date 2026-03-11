#!/usr/bin/env python3
# train_eeg.py - Train standard CSDI model for EEG data
'''
python train_eeg.py --config eeg.yaml --data_path /home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx --seq_len 100 --batch_size 32
# 完整参数示例
python train_eeg.py \
  --config eeg.yaml \
  --data_path /home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx \
  --seq_len 100 \
  --batch_size 32 \
  --device cuda:0 \
  --seed 42 \
  --valid_ratio 0.2
'''

import argparse
import datetime
import json
import os
import yaml
import torch
import numpy as np

from main_model import CSDI_EEG
from dataset_EEG import get_dataloader
from utils import train, evaluate

def main():
    parser = argparse.ArgumentParser(description="CSDI EEG Training")
    parser.add_argument("--config", type=str, default="eeg.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="/home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx")
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    args = parser.parse_args()
    print(args)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载配置
    with open("config/" + args.config, "r") as f:
        config = yaml.safe_load(f)

    # 更新配置
    config["model"]["is_unconditional"] = False
    config["train"]["batch_size"] = args.batch_size
    config["model"]["seq_len"] = args.seq_len

    # 创建保存文件夹
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f"./save/eeg_{current_time}/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
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
    model = CSDI_EEG(config, args.device, target_dim=64).to(args.device)

    # 训练或加载模型
    if not args.modelfolder:
        print("Training model...")
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
        print(f"Model training completed and saved to {foldername}/model.pth")
    else:
        print(f"Loading existing model from ./save/{args.modelfolder}/model.pth")
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
        torch.save(model.state_dict(), foldername + "/model.pth")
        print(f"Model loaded and copied to {foldername}/model.pth")

    # 评估
    print("Starting evaluation...")
    evaluate(
        model, 
        test_loader, 
        nsample=args.nsample, 
        scaler=1, 
        foldername=foldername
    )
    print("Evaluation completed!")

    print(f"\nModel saved at: {foldername}/model.pth")
    print(f"Config saved at: {foldername}/config.json")
    print(f"\nTo test the model, run:")
    print(f"python test_eeg_densification.py --model_path {foldername}/model.pth --config_path config/{args.config} --data_path {args.data_path}")

if __name__ == "__main__":
    main()