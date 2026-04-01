import argparse
import torch
import datetime
import json
import yaml
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'csdi'))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'datasets'))

from main_model import CSDI_Physio
from dataset_physio import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="physio.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = os.path.join(base_dir, "config", args.config)
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config["model"]["nfold"] = args.nfold

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = os.path.join(base_dir, "csdi", "save", f"physio_fold{args.nfold}_{current_time}")

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(os.path.join(foldername, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config['train']['batch_size'],
    missing_ratio=args.testmissingratio,
    device=args.device
)

model = CSDI_Physio(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load(os.path.join(base_dir, "csdi", "save", args.modelfolder, "model.pth")))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)