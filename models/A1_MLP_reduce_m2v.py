import torch
import torch.nn as nn

class Mol2vecReduceMLP(nn.Module):
    """
      - Trainable embedding for species
      - Trainable linear projection for raw mol2vec (300-d) -> reduced dimension
      - MLP on top of [species_emb, projected_mol2vec, duration]
    """
    def __init__(
        self,
        n_species,
        mol2vec_dim,
        species_emb_dim=16,
        mol2vec_reduce_dim=64,
        hidden_sizes=[128, 64, 32]
    ):
        """
        Args:
            n_species (int): number of species categories (for nn.Embedding).
            mol2vec_dim (int): input dimensionality of the raw mol2vec embeddings (300).
            species_emb_dim (int): dimensionality of the trainable species embedding.
            mol2vec_reduce_dim (int): dimension to which we project raw mol2vec vectors.
            hidden_sizes (list): sizes of hidden layers in the MLP.
        """
        super().__init__()
        # Trainable embedding for species
        self.species_emb = nn.Embedding(
            num_embeddings=n_species,
            embedding_dim=species_emb_dim
        )

        # Trainable dimension-reduction layer for mol2vec embeddings
        self.mol2vec_proj = nn.Linear(mol2vec_dim, mol2vec_reduce_dim)

        # MLP input = [species_emb_dim + mol2vec_reduce_dim + 1(duration)]
        mlp_input_dim = species_emb_dim + mol2vec_reduce_dim + 1

        layers = []
        for hdim in hidden_sizes:
            layers.append(nn.Linear(mlp_input_dim, hdim))
            layers.append(nn.ReLU())
            mlp_input_dim = hdim

        # Final output layer with 1 unit
        layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, species_id, duration, mol2vec_embed):
        # species embedding
        sp_emb = self.species_emb(species_id)  # (batch_size, species_emb_dim)

        # ensure duration is 2D
        if duration.dim() == 1:
            duration = duration.unsqueeze(1)

        # project mol2vec embedding to a reduced dimension
        mol2vec_reduced = self.mol2vec_proj(mol2vec_embed)

        # concatenate species embedding, projected mol2vec, duration
        x = torch.cat([sp_emb, mol2vec_reduced, duration], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)
