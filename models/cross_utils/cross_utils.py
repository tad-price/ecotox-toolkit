class CrossLayer(nn.Module):
    """
    A single cross layer:
      x_{l+1} = x0 * (x_l · w_l) + b_l + x_l
    """
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        # Inner product (B, d) dot (d,) -> (B, 1)
        dot = torch.sum(x * self.weight, dim=1, keepdim=True)
        out = x0 * dot + self.bias + x
        return out

class CrossNetwork(nn.Module):
    """
    Stacks multiple CrossLayer modules.
    """
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for layer in self.layers:
            x = layer(x0, x)
        return x

##############################################################################
# 3) MLP Head
##############################################################################
class MLPHead(nn.Module):
    """
    Maps the cross-network output to a scalar prediction.
    """
    def __init__(self, input_dim, hidden_sizes=[64, 32]):
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)
