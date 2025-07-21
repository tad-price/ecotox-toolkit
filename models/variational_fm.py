# variational_fm.py
"""
Mean-field Gaussian Variational-Bayes Factorization Machine
==========================================================

• Each scalar parameter θ (bias, linear weight, latent factor) has variational
  parameters (μ, log σ²).
• Re-parameterisation trick gives unbiased gradients.
• ELBO per minibatch  =  E_q[ NLL ]  +  KL(q‖p) / N.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Model                                                                      #
# --------------------------------------------------------------------------- #

class VariationalFactorizationMachine(nn.Module):
    """Fully factorised (mean-field) VB Factorization Machine."""

    def __init__(
        self,
        n_features: int,
        k: int,
        prior_var: float = 1.0,
        init_logvar: float = -6.0,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.k = k
        self.prior_var = prior_var
        self.prior_logvar = math.log(prior_var)

        # ─── variational parameters ──────────────────────────────────────── #
        self.bias_mu       = nn.Parameter(torch.zeros(1))
        self.bias_logvar   = nn.Parameter(torch.full((1,), init_logvar))

        self.linear_mu     = nn.Parameter(torch.zeros(n_features))
        self.linear_logvar = nn.Parameter(torch.full((n_features,), init_logvar))

        self.factor_mu     = nn.Parameter(torch.randn(n_features, k) * 0.01)
        self.factor_logvar = nn.Parameter(torch.full((n_features, k), init_logvar))

        # homoscedastic observation noise σ²  (log-parametrised to stay >0)
        self.obs_logvar = nn.Parameter(torch.tensor(0.0))

    # ─────────────────────────────────────────────────────────────────────── #
    # helpers                                                                 #
    # ─────────────────────────────────────────────────────────────────────── #

    def _sample_theta(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw one MC sample θ̃ = μ + σ·ε."""
        eps_b = torch.randn_like(self.bias_mu)
        eps_l = torch.randn_like(self.linear_mu)
        eps_v = torch.randn_like(self.factor_mu)

        bias    = self.bias_mu   + torch.exp(0.5 * self.bias_logvar)   * eps_b
        linear  = self.linear_mu + torch.exp(0.5 * self.linear_logvar) * eps_l
        factors = self.factor_mu + torch.exp(0.5 * self.factor_logvar) * eps_v
        return bias, linear, factors

    # ─────────────────────────────────────────────────────────────────────── #

    def forward(self, X: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (B, n_features) dense FloatTensor  (convert from CSR just before call)
        sample : bool
            • True  – use a Monte-Carlo draw θ̃\n
            • False – use variational means μ  (for deterministic eval)
        """
        if sample:
            bias, linear, factors = self._sample_theta()
        else:
            bias, linear, factors = self.bias_mu, self.linear_mu, self.factor_mu

        linear_term = torch.matmul(X, linear)                                   # (B,)
        XV          = torch.matmul(X, factors)                                  # (B, k)
        sum_square  = XV.pow(2).sum(dim=1)                                      # (B,)
        square_sum  = torch.matmul(X.pow(2), factors.pow(2)).sum(dim=1)         # (B,)
        interactions = 0.5 * (sum_square - square_sum)                          # (B,)

        return bias + linear_term + interactions

    # ─────────────────────────────────────────────────────────────────────── #
    #   likelihood & KL                                                      #
    # ─────────────────────────────────────────────────────────────────────── #

    def nll_gaussian(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Element-wise −log p(y|θ) for homoscedastic Normal likelihood."""
        var = torch.exp(self.obs_logvar)
        return 0.5 * (
            math.log(2 * math.pi) + self.obs_logvar + (y - y_pred).pow(2) / var
        )

    def kl_divergence(self) -> torch.Tensor:
        """KL(q‖p) with N(0,σ₀²) prior (closed form)."""
        kl_total = 0.0
        for mu, logvar in [
            (self.bias_mu,   self.bias_logvar),
            (self.linear_mu, self.linear_logvar),
            (self.factor_mu, self.factor_logvar),
        ]:
            kl = 0.5 * (
                torch.exp(logvar) / self.prior_var
                + mu.pow(2)      / self.prior_var
                - 1
                - logvar
                + self.prior_logvar
            )
            kl_total += kl.sum()
        return kl_total


# --------------------------------------------------------------------------- #
#  ELBO helpers                                                               #
# --------------------------------------------------------------------------- #

def elbo_batch(
    model: VariationalFactorizationMachine,
    X: torch.Tensor,
    y: torch.Tensor,
    dataset_size: int,
    mc_samples: int = 1,
) -> torch.Tensor:
    """
    Negative ELBO for one minibatch (value to **minimise**).

        ELBO ≈ (1/K) Σ_k  −log p(y | θ̃_k)  +  KL/N
    """
    nll = 0.0
    for _ in range(mc_samples):
        y_pred = model(X, sample=True)
        nll += model.nll_gaussian(y, y_pred).mean()
    nll /= mc_samples

    kl_scaled = model.kl_divergence() / dataset_size
    return nll + kl_scaled


def train_variational_fm(
    model: VariationalFactorizationMachine,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device = torch.device("cpu"),
    mc_samples: int = 1,
) -> None:
    """ELBO maximisation with Adam (prints train −ELBO and optional val-MSE)."""
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset_size = len(train_loader.dataset)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            if Xb.is_sparse:
                Xb = Xb.to_dense()

            optimiser.zero_grad()
            loss = elbo_batch(model, Xb, yb, dataset_size, mc_samples)
            loss.backward()
            optimiser.step()
            running += loss.item()

        print(f"Epoch {epoch:3d} | train −ELBO {running / len(train_loader):.4f}")

        if val_loader is None:
            continue

        # quick deterministic MSE on val-set
        model.eval()
        with torch.no_grad():
            mse, n = 0.0, 0
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                if Xv.is_sparse:
                    Xv = Xv.to_dense()
                pred = model(Xv, sample=False)
                mse += ((pred - yv) ** 2).sum().item()
                n += yv.numel()
        print(f"            val-MSE {mse / n:.4f}")
