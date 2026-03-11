# train_enso.py
import argparse
import datetime
import json
import os
import yaml
import torch
from main_model import CSDI_ENSO
from dataset_enso import get_enso_dataloader
from utils import train, evaluate
import numpy as np

parser = argparse.ArgumentParser(description="CSDI ENSO")
parser.add_argument("--config", type=str, default="enso.yaml")
parser.add_argument('--device', default='cuda:0', help='Device')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--seq_len", type=int, default=100)
parser.add_argument("--seq_stride", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)

args = parser.parse_args()
print(args)

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Load config
with open("config/" + args.config, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["train"]["batch_size"] = args.batch_size
config["model"]["seq_len"] = args.seq_len

print(json.dumps(config, indent=4))

# Create save folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/enso_{current_time}/"
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# Get dataloaders
train_loader, valid_loader, test_dataset = get_enso_dataloader(
    file_path="/home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset 7-ENSO.xlsx",
    batch_size=config["train"]["batch_size"],
    seq_len=args.seq_len,
    seq_stride=args.seq_stride
)

# Initialize model
model = CSDI_ENSO(config, args.device).to(args.device)

# Train or load model
if not args.modelfolder:
    print("Training model...")
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername
    )
    print(f"Model training completed and saved to {foldername}/model.pth")
else:
    print(f"Loading existing model from ./save/{args.modelfolder}/model.pth")
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    torch.save(model.state_dict(), foldername + "/model.pth")
    print(f"Model loaded and copied to {foldername}/model.pth")

# Evaluate model
print("Starting evaluation...")
evaluate(model, test_dataset, nsample=args.nsample, scaler=1, foldername=foldername)
print("Evaluation completed!")

print(f"\nModel saved at: {foldername}/model.pth")
print(f"Config saved at: {foldername}/config.json")