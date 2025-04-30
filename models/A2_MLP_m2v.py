class Mol2vecMLP(nn.Module):
    def __init__(self, n_species, mol2vec_dim,
                 species_emb_dim=16,
                 hidden_sizes=[128, 64, 32]):
        super().__init__()
        self.species_emb = nn.Embedding(
            num_embeddings=n_species, 
            embedding_dim=species_emb_dim
        )
        # MLP input dimension = species_emb_dim + mol2vec_dim + 1 (for duration)
        mlp_input_dim = species_emb_dim + mol2vec_dim + 1

        layers = []
        for hdim in hidden_sizes:
            layers.append(nn.Linear(mlp_input_dim, hdim))
            layers.append(nn.ReLU())
            mlp_input_dim = hdim
        layers.append(nn.Linear(mlp_input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, species_id, duration, mol2vec_embed):
        sp_emb = self.species_emb(species_id)  # (B, species_emb_dim)
        if duration.dim() == 1:
            duration = duration.unsqueeze(1)
        # Concatenate species embedding, mol2vec embedding, and duration
        x = torch.cat([sp_emb, mol2vec_embed, duration], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)
