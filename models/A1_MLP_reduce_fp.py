import torch 
import torch.nn as nn
class FingerprintReduceMLP(nn.Module):
    """
      - species_emb: trainable embedding for species (species_emb_dim)
      - fp_proj: a linear layer to reduce raw fingerprint (fp_dim) -> fp_reduce_dim
      - final MLP on top of [species_emb, reduced_fp, duration]
    """
    def __init__(
        self,
        n_species,
        fp_dim,
        species_emb_dim=16,
        fp_reduce_dim=64,
        hidden_sizes=[128, 64, 32]
    ):
        """
        Args:
            n_species (int): Number of species categories for nn.Embedding.
            fp_dim (int): Input dimensionality of the raw fingerprint (e.g., 1024).
            species_emb_dim (int): Dim of the trainable species embedding.
            fp_reduce_dim (int): Dim to which the raw fingerprint is projected.
            hidden_sizes (list): Sizes of hidden layers in the MLP.
        """
        super().__init__()
        # Trainable embedding for species
        self.species_emb = nn.Embedding(
            num_embeddings=n_species,
            embedding_dim=species_emb_dim
        )
        # Trainable dimensionality-reduction layer for fingerprints
        self.fp_proj = nn.Linear(fp_dim, fp_reduce_dim)

        # MLP input = [species_emb_dim + fp_reduce_dim + 1(duration)]
        mlp_input_dim = species_emb_dim + fp_reduce_dim + 1

        layers = []
        for hdim in hidden_sizes:
            layers.append(nn.Linear(mlp_input_dim, hdim))
            layers.append(nn.ReLU())
            mlp_input_dim = hdim

        # Final output layer with 1 unit
        layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, species_id, duration, fp_embed):
        # species embedding
        sp_emb = self.species_emb(species_id)  # (B, species_emb_dim)

        # ensure duration is 2D
        if duration.dim() == 1:
            duration = duration.unsqueeze(1)

        # project fingerprint to a reduced dimension
        fp_reduced = self.fp_proj(fp_embed)

        # concatenate species embedding, projected fingerprint, duration
        x = torch.cat([sp_emb, fp_reduced, duration], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)

