# test_enso.py
import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from main_model import CSDI_ENSO
from dataset_enso import load_enso_data
import os

def load_model(model_path, config_path, device='cuda:0'):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model = CSDI_ENSO(config, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def impute_enso(model, partial_data, device='cuda:0', n_samples=10):
    partial_len, num_features = partial_data.shape
    seq_len = partial_len * 2
    data = np.zeros((seq_len, num_features))
    data[1::2] = partial_data
    known_mask = np.zeros_like(data)
    known_mask[1::2] = 1
    
    observed_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    observed_mask = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).to(device)
    cond_mask = observed_mask.clone()
    observed_tp = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).to(device)
    
    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    cond_mask = cond_mask.permute(0, 2, 1)
    
    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)
        samples = samples.permute(0, 1, 3, 2)
        samples = samples.squeeze(0)
        imputed_samples = samples.cpu().numpy()
        
        result = np.mean(imputed_samples, axis=0)
        result[1::2] = partial_data

        return result

def main():
    parser = argparse.ArgumentParser(description="Test CSDI model for ENSO imputation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model.pth")
    parser.add_argument("--config_path", type=str, default="config/enso.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of imputation samples")
    
    args = parser.parse_args()
    
    model = load_model(args.model_path, args.config_path, args.device)
    
    # Load ENSO test data
    sequences = load_enso_data("Dataset 7-ENSO.xlsx", seq_len=100, seq_stride=100)
    test_sequence = sequences[0]  # Use first sequence for testing
    
    # Create partial data (every other point)
    partial_data = test_sequence[1::2]
    
    print(f"Test sequence shape: {test_sequence.shape}")
    print(f"Partial data shape: {partial_data.shape}")
    
    # Perform imputation
    result = impute_enso(
        model,
        partial_data,
        device=args.device,
        n_samples=args.n_samples
    )

    print(f"Imputation completed. Result shape: {result.shape}")

    # Plot results for each region
    regions = ['Nino1+2', 'Nino3', 'Nino3.4', 'Nino4']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, region in enumerate(regions):
        time_full = np.arange(len(test_sequence))
        time_result = np.arange(len(result))
        
        axes[i].plot(time_full, test_sequence[:, i], 'b-', 
                    label='Original Data', linewidth=2, alpha=0.8)
        axes[i].plot(time_result, result[:, i], 'r--', 
                    label='Imputed Result', linewidth=2, alpha=0.8)
        
        known_indices = np.arange(1, len(result), 2)
        axes[i].scatter(known_indices, result[known_indices, i], 
                       color='green', s=30, label='Known Points', zorder=5)
        
        axes[i].set_xlabel('Time Step (Weeks)')
        axes[i].set_ylabel('SST')
        axes[i].set_title(f'{region} SST: Original vs Imputed')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enso_imputation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate statistics
    print("\nImputation Statistics:")
    for i, region in enumerate(regions):
        mae = np.mean(np.abs(test_sequence[:, i] - result[:len(test_sequence), i]))
        rmse = np.sqrt(np.mean((test_sequence[:, i] - result[:len(test_sequence), i])**2))
        print(f"{region}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    return result, test_sequence

if __name__ == "__main__":
    imputed_data, original_data = main()