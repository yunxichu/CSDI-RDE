# test_enso_gpr.py
import numpy as np
import time
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import itertools
import matplotlib.pyplot as plt
from gpr_module import GaussianProcessRegressor
from tqdm import tqdm

def _parallel_predict_enso(comb, traindata, target_idx, steps_ahead=1):
    try:
        trainlength = len(traindata)
        trainX = traindata[:trainlength-steps_ahead, list(comb)]
        trainy = traindata[steps_ahead:trainlength, target_idx]
        testX = traindata[trainlength-steps_ahead, list(comb)].reshape(1, -1)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        combined_X = np.vstack([trainX, testX])
        combined_X_scaled = scaler_X.fit_transform(combined_X)
        trainX_scaled = combined_X_scaled[:-1]
        testX_scaled = combined_X_scaled[-1:]

        trainy_scaled = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)

        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return pred, std_scaled[0]
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return np.nan, np.nan

def predict_enso(seq, trainlength=30, L=3, s=200, j=0, n_jobs=4, steps_ahead=1):
    noise_strength = 1e-4
    x = seq + noise_strength * np.random.randn(*seq.shape)

    total_steps = len(seq) - trainlength
    result = np.zeros((3, total_steps))
    
    pool = mp.Pool(processes=n_jobs)
    
    with tqdm(total=total_steps, desc=f"Predicting region {j}") as pbar:
        for step in range(total_steps):
            traindata = x[step: step + trainlength, :]
            real_value = x[step + trainlength, j]

            D = traindata.shape[1]
            combs = list(itertools.combinations(range(D), L))
            np.random.shuffle(combs)
            selected_combs = combs[:s]
            
            predictions = pool.map(
                partial(_parallel_predict_enso, 
                        traindata=traindata,
                        target_idx=j,
                        steps_ahead=steps_ahead),
                selected_combs
            )
            
            pred_values = np.array([p[0] for p in predictions])
            pred_stds = np.array([p[1] for p in predictions]) 
            valid_mask = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
            valid_preds = pred_values[valid_mask]
            
            if len(valid_preds) == 0:
                final_pred = np.nan
                final_std = np.nan
            elif len(valid_preds) == 1:
                final_pred = valid_preds[0]
                final_std = 0.0
            else:
                try:
                    kde = gaussian_kde(valid_preds)
                    xi = np.linspace(valid_preds.min(), valid_preds.max(), 1000)
                    density = kde(xi)
                    final_pred = np.sum(xi * density) / np.sum(density)
                    final_std = np.std(valid_preds)
                except:
                    final_pred = np.mean(valid_preds)
                    final_std = np.std(valid_preds)

            result[0, step] = final_pred
            result[1, step] = final_std
            result[2, step] = real_value - final_pred
            
            pbar.update(1)

    pool.close()
    return result

def main():
    # Load your ENSO data (replace with actual data loading)
    from dataset_enso import load_enso_data
    sequences = load_enso_data("Dataset 7-ENSO.xlsx", seq_len=100, seq_stride=100)
    enso_data = sequences[0]  # Use first sequence
    
    print(f"ENSO data shape: {enso_data.shape}")
    
    regions = ['Nino1+2', 'Nino3', 'Nino3.4', 'Nino4']
    
    # Predict for each region
    all_results = []
    for j in range(4):
        print(f"\nPredicting for {regions[j]}...")
        result = predict_enso(
            seq=enso_data,
            trainlength=30,
            L=3,
            s=100,
            j=j,
            n_jobs=4,
            steps_ahead=1
        )
        all_results.append(result)
        
        # Calculate RMSE for this region
        rmse = np.sqrt(np.mean(np.square(result[2])))
        print(f"{regions[j]} - RMSE: {rmse:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for j, region in enumerate(regions):
        result = all_results[j]
        total_time_steps = enso_data.shape[0]
        prediction_start_idx = 30 + 1 - 1  # trainlength + steps_ahead - 1
        prediction_time_indices = np.arange(prediction_start_idx, 
                                          prediction_start_idx + result.shape[1])
        
        axes[j].plot(np.arange(total_time_steps), enso_data[:, j], 
                    'b-', label='Actual Data', alpha=0.7)
        axes[j].plot(prediction_time_indices, result[0, :], 
                    'r-', label='GPR Predictions', alpha=0.8, linewidth=2)
        axes[j].axvline(x=prediction_start_idx, color='gray', 
                       linestyle='--', alpha=0.5, label='Prediction Start')
        
        axes[j].set_xlabel('Time Step (Weeks)')
        axes[j].set_ylabel('SST')
        axes[j].set_title(f'{region} SST: Actual vs GPR Predictions')
        axes[j].legend()
        axes[j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enso_gpr_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()