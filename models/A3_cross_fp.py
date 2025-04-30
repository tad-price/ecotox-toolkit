from cross_utils.cross_utils import CrossLayer, CrossNetwork, MLPHead
class FingerprintCrossNetwork(nn.Module):
    """
      1. Embedding for species (trainable).
      2. Linear+ReLU to reduce e.g. 1024-d fingerprint to 'fp_reduced_dim'.
      3. Concatenate [species_emb, reduced_fp, duration].
      4. Cross Network on that concatenation.
      5. MLP head for final regression.
    """
    def __init__(self,
                 n_species,
                 fp_dim=1024,          
                 species_emb_dim=16,
                 fp_reduced_dim=64,
                 num_cross_layers=2,
                 mlp_hidden_sizes=[64, 32]):
        super().__init__()
        # Species embedding
        self.species_emb = nn.Embedding(
            num_embeddings=n_species,
            embedding_dim=species_emb_dim
        )
        # Fingerprint dimension reduction
        self.linear_fp = nn.Linear(fp_dim, fp_reduced_dim)

        # Combined dimension = species_emb_dim + fp_reduced_dim + 1 (duration)
        self.combined_dim = species_emb_dim + fp_reduced_dim + 1

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

    def forward(self, species_id, duration, fp_embed):
        # 1) Species embedding
        sp_emb = self.species_emb(species_id)  # (B, species_emb_dim)

        # 2) FP dimension reduction + ReLU
        fp_red = torch.relu(self.linear_fp(fp_embed))  # (B, fp_reduced_dim)

        # 3) If duration is 1D, unsqueeze
        if duration.dim() == 1:
            duration = duration.unsqueeze(1)

        # 4) Concatenate
        features = torch.cat([sp_emb, fp_red, duration], dim=1)  # (B, combined_dim)

        # 5) Cross Network
        cross_out = self.cross_network(features)  # (B, combined_dim)

        # 6) MLP Head
        output = self.mlp_head(cross_out)
        return output
