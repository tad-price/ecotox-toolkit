import torch
import math

def evaluate_rmse(model, data_loader, device):
    """
    Evaluate RMSE of a factorization machine model
    """
    model.eval()
    total_samples = 0
    sum_sq = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch).squeeze()
            diff = preds - y_batch
            sum_sq += torch.sum(diff * diff).item()
            total_samples += X_batch.size(0)

    # Compute RMSE
    return math.sqrt(sum_sq / total_samples)
