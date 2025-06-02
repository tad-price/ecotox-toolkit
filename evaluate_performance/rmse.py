import torch
import math
def evaluate_rmse(model, dataloader, device):
    # evaluate rmse of an MLP
    model.eval()
    sum_sq, total = 0.0, 0
    with torch.no_grad():
        for species_id, duration, embedding, y in dataloader:
            species_id = species_id.to(device)
            duration = duration.to(device)
            embedding = embedding.to(device)
            y = y.to(device)

            preds = model(species_id, duration, embedding)
            sum_sq += torch.sum((preds - y) ** 2).item()
            total += y.size(0)
    return math.sqrt(sum_sq / total)
