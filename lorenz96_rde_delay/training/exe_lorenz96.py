# exe_lorenz96.py
'''
Training script for Lorenz96 CSDI model
'''
import argparse
import datetime
import json
import os
import sys
import numpy as np

training_dir = os.path.dirname(os.path.abspath(__file__))
lorenz_dir = os.path.dirname(training_dir)
base_dir = os.path.dirname(lorenz_dir)
sys.path.insert(0, base_dir)
sys.path.insert(0, lorenz_dir)
sys.path.insert(0, os.path.join(lorenz_dir, 'models'))
sys.path.insert(0, os.path.join(lorenz_dir, 'data'))
sys.path.insert(0, training_dir)

import torch
import yaml
from models.main_model import CSDI_Lorenz96
from data.dataset_lorenz96 import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI Lorenz96")
parser.add_argument("--config", type=str, default="config/lorenz96.yaml")
parser.add_argument('--device', default='cuda:0', help='Device')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--epochs", type=int, default=0)

args = parser.parse_args()
print(args)

config_path = os.path.join(lorenz_dir, args.config)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional

if args.epochs > 0:
    config["train"]["epochs"] = args.epochs

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = os.path.join(training_dir, "save", f"lorenz96_{current_time}")
os.makedirs(foldername, exist_ok=True)
with open(os.path.join(foldername, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    N=config['model'].get('N', 100),
    T=400,
    sample_step=8,
    batch_size=config['train']['batch_size'],
    seed=args.seed
)

model = CSDI_Lorenz96(config, args.device).to(args.device)

if not args.modelfolder:
    print("Training model...")
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername
    )
else:
    print(f"Loading existing model from {args.modelfolder}")
    model.load_state_dict(torch.load(os.path.join(lorenz_dir, "training", "save", args.modelfolder, "model.pth")))
    torch.save(model.state_dict(), os.path.join(foldername, "model.pth"))

print("Starting evaluation...")
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

print(f"\nModel saved at: {foldername}/model.pth")
print(f"Config saved at: {foldername}/config.json")