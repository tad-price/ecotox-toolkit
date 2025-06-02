import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    def __init__(self, n_features, k):
        super(FactorizationMachine, self).__init__()
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        # Linear term
        self.linear = nn.Parameter(torch.zeros(n_features))
        # Factorized interaction parameters
        self.factors = nn.Parameter(torch.randn(n_features, k)) 

    def forward(self, X):
        # X shape: (batch_size, n_features)
        # linear term
        linear_term = torch.matmul(X, self.linear)
        # interaction term
        #  (sum_{i} (V_i * x_i))^2 - sum_{i}((V_i * x_i)^2)
        #  all / 2
        interactions = (
            torch.pow(torch.matmul(X, self.factors), 2)
            - torch.matmul(torch.pow(X, 2), torch.pow(self.factors, 2))
        ).sum(1) * 0.5
        return self.bias + linear_term + interactions
