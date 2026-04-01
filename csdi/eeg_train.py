"""
CSDI Training for EEG Data
"""
import os
import sys
import argparse
import yaml
import json
import datetime
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'datasets'))

from datasets.dataset_eeg import EEG_Dataset

def main():
    parser = argparse.ArgumentParser(description="CSDI for EEG")
    parser.add_argument("--run_folder", type=str, default="", help="Output folder path")
    parser.add_argument("--config", type=str, default="eeg")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing_ratio", type=float, default=0.1)
    parser.add_argument("--missing_mode", type=str, default="random",
                       choices=["uniform", "random"])
    parser.add_argument("--data_path", type=str, default="./data/data_extra/Dataset_3-EEG.xlsx")
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nsample", type=int, default=100)

    args = parser.parse_args()

    path = os.path.join(project_root, "config", args.config + ".yaml")
    if not os.path.exists(path):
        path = os.path.join(project_root, "config", "base.yaml")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["test_missing_ratio"] = args.missing_ratio

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_folder:
        foldername = args.run_folder if args.run_folder.endswith('/') else args.run_folder + '/'
    else:
        foldername = f"./save/eeg_{args.missing_mode}_ratio{args.missing_ratio}_fold{args.nfold}_{current_time}/"

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    with open(foldername + "config.json", "w") as f:
        json.dump({"config": config, "args": vars(args)}, f, indent=4)

    dataset = EEG_Dataset(
        data_path=args.data_path,
        missing_ratio=args.missing_ratio,
        seed=args.seed,
        missing_mode=args.missing_mode,
        use_generated_missing=True,
    )

    from csdi.main_model import CSDI_EEG
    from csdi.utils import train, evaluate
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset, batch_size=config["train"]["batch_size"],
        shuffle=True, num_workers=0
    )

    model = CSDI_EEG(config, args.device, target_dim=dataset.n_features).to(args.device)

    train(
        model,
        config["train"],
        train_loader,
        foldername=foldername,
    )

    evaluate(model, None, nsample=args.nsample, scaler=None, mean_scaler=None, foldername=foldername)

if __name__ == "__main__":
    main()