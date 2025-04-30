from cross_utils.cross_utils import CrossLayer, CrossNetwork, MLPHead
class SelfiesCrossNetwork(nn.Module):
    """
    Model architecture:
      1. Species embedding: converts species IDs into dense vectors.
      2. SELFIES embedding reduction: a linear layer (with ReLU) reduces
         the high-dimensional SELFIES embeddings.
      3. Concatenation: species embedding, reduced SELFIES, and (scaled)
         test duration are concatenated.
      4. Cross network: stacked cross layers to capture feature interactions.
      5. MLP head: final multilayer perceptron to produce the scalar output.
    """
    def __init__(self, n_species, selfies_embed_dim,
                 species_emb_dim=16,
                 selfies_reduced_dim=64,
                 num_cross_layers=2,
                 mlp_hidden_sizes=[64, 32]):
        super().__init__()
        # Species embedding branch
        self.species_emb = nn.Embedding(num_embeddings=n_species,
                                        embedding_dim=species_emb_dim)
        # Linear reduction for SELFIES embeddings
        self.linear_selfies = nn.Linear(selfies_embed_dim, selfies_reduced_dim)
        # Combined feature dimension
        self.combined_dim = species_emb_dim + selfies_reduced_dim + 1
        # Cross network
        self.cross_network = CrossNetwork(input_dim=self.combined_dim,
                                          num_layers=num_cross_layers)
        # Final MLP head for regression
        self.mlp_head = MLPHead(input_dim=self.combined_dim,
                                hidden_sizes=mlp_hidden_sizes)

    def forward(self, species_id, duration, selfies_embed):
        sp_emb = self.species_emb(species_id)  # (B, species_emb_dim)
        selfies_red = torch.relu(self.linear_selfies(selfies_embed))  # (B, selfies_reduced_dim)
        features = torch.cat([sp_emb, selfies_red, duration], dim=1)   # (B, combined_dim)
        cross_out = self.cross_network(features)                       # (B, combined_dim)
        output = self.mlp_head(cross_out)
        return output

##############################################################################
# Training and evaluation routines
##############################################################################
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
