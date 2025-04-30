import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, device):

    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0
    
    for species_id, duration, selfies_embed, y in dataloader:
        species_id = species_id.to(device)
        duration = duration.to(device)
        selfies_embed = selfies_embed.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(species_id, duration, selfies_embed)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(dataloader)