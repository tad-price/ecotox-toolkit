import torch
import torch.nn as nn


class SelfiesReduceMLP(nn.Module):
    def __init__(
        self,
        n_species,
        selfies_embed_dim,
        species_emb_dim=16,
        selfies_reduce_dim=64,
        hidden_sizes=[128, 64, 32]
    ):
        """
        Args:
            n_species (int): Number of species categories for nn.Embedding.
            selfies_embed_dim (int): Input dimensionality of the raw Selfies embeddings.
            species_emb_dim (int): Dimensionality of the trainable species embedding.
            selfies_reduce_dim (int): Dimensionality to which the raw Selfies vectors are projected.
            hidden_sizes (list): Sizes of the hidden layers in the MLP.
        """
        super().__init__()
        # Trainable embedding for species
        self.species_emb = nn.Embedding(
            num_embeddings=n_species, 
            embedding_dim=species_emb_dim
        )
        # Trainable dimensionality-reduction layer for Selfies embeddings
        self.selfies_proj = nn.Linear(selfies_embed_dim, selfies_reduce_dim)

        # MLP input = [species_emb_dim + selfies_reduce_dim + 1(duration)]
        mlp_input_dim = species_emb_dim + selfies_reduce_dim + 1

        layers = []
        for hdim in hidden_sizes:
            layers.append(nn.Linear(mlp_input_dim, hdim))
            layers.append(nn.ReLU())
            mlp_input_dim = hdim
        # Final output layer with 1 unit
        layers.append(nn.Linear(mlp_input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, species_id, duration, selfies_embed):
        # species embedding
        sp_emb = self.species_emb(species_id)  # (batch_size, species_emb_dim)
        
        # ensure duration is 2D
        if duration.dim() == 1:
            duration = duration.unsqueeze(1)
        
        # project Selfies embedding to a reduced dimension
        selfies_reduced = self.selfies_proj(selfies_embed)  # (batch_size, selfies_reduce_dim)

        # concatenate species embedding, projected Selfies, duration
        x = torch.cat([sp_emb, selfies_reduced, duration], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)
