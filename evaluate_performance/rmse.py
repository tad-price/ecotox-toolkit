import torch
import math
def evaluate_rmse(model, dataloader, device):
    model.eval()
    sum_sq, total = 0.0, 0
    with torch.no_grad():
        for species_id, duration, selfies_embed, y in dataloader:
            species_id = species_id.to(device)
            duration = duration.to(device)
            selfies_embed = selfies_embed.to(device)
            y = y.to(device)

            preds = model(species_id, duration, selfies_embed)
            sum_sq += torch.sum((preds - y) ** 2).item()
            total += y.size(0)
    return math.sqrt(sum_sq / total)
