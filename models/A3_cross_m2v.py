from cross_utils.cross_utils import CrossLayer, CrossNetwork, MLPHead

class Mol2VecCrossNetwork(nn.Module):
    """
      1. Embedding for species (trainable).
      2. Linear+ReLU to reduce 300-d mol2vec to `mol2vec_reduced_dim`.
      3. Concatenate [species_emb, reduced_mol2vec, duration].
      4. Cross Network over that concatenation.
      5. MLP head for final regression output.
    """
    def __init__(self,
                 n_species,
                 mol2vec_dim=300,
                 species_emb_dim=16,
                 mol2vec_reduced_dim=64,
                 num_cross_layers=2,
                 mlp_hidden_sizes=[64, 32]):
        super().__init__()
        # Species embedding
        self.species_emb = nn.Embedding(num_embeddings=n_species,
                                        embedding_dim=species_emb_dim)
        # Mol2vec dimension reduction
        self.linear_mol2vec = nn.Linear(mol2vec_dim, mol2vec_reduced_dim)

        # Combined dimension = species_emb_dim + mol2vec_reduced_dim + 1 (for duration)
        self.combined_dim = species_emb_dim + mol2vec_reduced_dim + 1

        # Cross Network
        self.cross_network = CrossNetwork(
            input_dim=self.combined_dim,
            num_layers=num_cross_layers
        )

        # MLP Head
        self.mlp_head = MLPHead(
            input_dim=self.combined_dim,
            hidden_sizes=mlp_hidden_sizes
        )

    def forward(self, species_id, duration, mol2vec_embed):
        # 1) Species embedding
        sp_emb = self.species_emb(species_id)  # shape: (B, species_emb_dim)

        # 2) Mol2vec dimension reduction + ReLU
        mol2vec_red = torch.relu(self.linear_mol2vec(mol2vec_embed))  # (B, reduced_dim)

        # 3) If duration is 1D, unsqueeze
        if duration.dim() == 1:
            duration = duration.unsqueeze(1)

        # 4) Concatenate all
        features = torch.cat([sp_emb, mol2vec_red, duration], dim=1)  # (B, combined_dim)

        # 5) Cross Network
        cross_out = self.cross_network(features)  # (B, combined_dim)

        # 6) MLP Head
        output = self.mlp_head(cross_out)
        return output
