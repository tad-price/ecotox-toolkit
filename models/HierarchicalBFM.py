import numpy as np
import scipy.sparse as sp
from tqdm.auto import trange

class HierarchicalBFM:
    """
    Gibbs sampler for 2-way Bayesian Factorization Machine with Hierarchical Aleatoric Uncertainty.
    Learns a separate precision parameter (alpha) for each group (chemical), regularized by a shared prior.
    """

    def __init__(self, n_features: int, n_groups: int, k: int):
        self.n_features = n_features
        self.n_groups = n_groups
        self.k = k
        self.samples = []

        # Hyper-prior constants
        # Prior for alpha_c ~ Gamma(alpha_a0, alpha_b0)
        # We use fixed hyperparameters for the prior of alphas to induce shrinkage towards the prior mean.
        self.alpha_a0 = 1.0
        self.alpha_b0 = 1.0
        
        # Prior for weights and factors
        self.gamma_0  = 1.0  # prior precision of means
        self.mu_0     = 0.0  # prior mean of means
        self.alpha_l  = self.beta_l  = 1.0 # Gamma prior for lambda (precision of w/v)

    def fit(self, X: sp.csr_matrix, y: np.ndarray, groups: np.ndarray,
            n_iter: int = 150, n_burn: int = 50, random_state: int | None = None):
        
        rng = np.random.default_rng(random_state)
        n_obs, p = X.shape
        assert p == self.n_features
        assert len(groups) == n_obs
        assert groups.max() < self.n_groups

        # Initialize parameters
        w0 = 0.0
        w  = np.zeros(p)
        v  = rng.normal(0.0, 0.1, size=(p, self.k))
        
        # Initialize per-group precision
        # Start with global estimation to be safe
        alpha_vec = np.ones(self.n_groups) 

        # Hyper-parameters for w and v
        mu_w = 0.0; lam_w = 1.0
        mu_v = 0.0; lam_v = 1.0

        # Pre-compute group indices for fast SSE calculation
        # group_indices[g] = list of row indices for group g
        # For speed, we can just use boolean masks or iterate if n_groups is small.
        # But n_groups can be large (~2000). 
        # Let's pre-calculate sum of squared errors per group?
        # Better: use np.bincount for weighted sums if groups are 0..N-1
        
        # Helpers
        X2 = X.copy(); X2.data **= 2
        q  = X @ v
        interact = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
        y_hat    = w0 + X @ w + interact
        e        = y - y_hat  # residuals

        for it in trange(n_iter, desc="HierarchicalBFM"):
            
            # --- 1. Sample w0 ---
            # Precision = gamma_0 + sum(alpha_i)
            # alpha_obs[i] = alpha_vec[groups[i]]
            alpha_obs = alpha_vec[groups]
            
            sum_alpha = np.sum(alpha_obs)
            var_w0 = 1.0 / (self.gamma_0 + sum_alpha)
            
            # Mean = var * (gamma_0 * mu_0 + sum(alpha_i * (e_i + w0)))
            # e + w0 is the target residual for w0
            weighted_res = np.dot(alpha_obs, e + w0)
            mean_w0 = var_w0 * (self.gamma_0 * self.mu_0 + weighted_res)
            
            new_w0 = rng.normal(mean_w0, np.sqrt(var_w0))
            e += (w0 - new_w0)
            w0 = new_w0

            # --- 2. Sample w (linear weights) ---
            
            # Hyper-params for w
            var_mu_w  = 1.0 / (self.gamma_0 + p * lam_w)
            mean_mu_w = var_mu_w * lam_w * w.sum()
            mu_w      = rng.normal(mean_mu_w, np.sqrt(var_mu_w))

            shape = self.alpha_l + 0.5 * p
            rate  = self.beta_l  + 0.5 * np.square(w - mu_w).sum()
            lam_w = rng.gamma(shape, 1.0 / rate)

            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0: continue
                
                x_vals = col.data
                # Get alphas for these observations
                alpha_sub = alpha_vec[groups[idx]]
                
                # Strip old contribution
                e_j = e[idx] + x_vals * w[j]
                
                # Weighted precision: lam_w + sum(alpha_i * x_i^2)
                weighted_x2 = np.dot(alpha_sub, x_vals**2)
                var = 1.0 / (lam_w + weighted_x2)
                
                # Weighted mean
                weighted_dot = np.dot(alpha_sub, x_vals * e_j)
                mean = var * (lam_w * mu_w + weighted_dot)
                
                new_wj = rng.normal(mean, np.sqrt(var))
                
                # Update residuals
                e[idx] += x_vals * (w[j] - new_wj)
                w[j] = new_wj

            # --- 3. Sample v (latent factors) ---
            
            # Hyper-params for v
            var_mu_v  = 1.0 / (self.gamma_0 + p * self.k * lam_v)
            mean_mu_v = var_mu_v * lam_v * v.sum()
            mu_v      = rng.normal(mean_mu_v, np.sqrt(var_mu_v))

            shape = self.alpha_l + 0.5 * p * self.k
            rate  = self.beta_l  + 0.5 * np.square(v - mu_v).sum()
            lam_v = rng.gamma(shape, 1.0 / rate)

            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0: continue
                x_vals = col.data
                alpha_sub = alpha_vec[groups[idx]]

                for f in range(self.k):
                    q_f = q[idx, f]
                    v_old = v[j, f]
                    
                    # h = x * (q_f - x * v_old)
                    h = x_vals * (q_f - x_vals * v_old)
                    
                    # Weighted precision
                    weighted_h2 = np.dot(alpha_sub, h**2)
                    denom = lam_v + weighted_h2
                    denom = max(denom, 1e-12)
                    var = 1.0 / denom
                    
                    # Weighted mean
                    # target = e[idx] + h * v_old
                    target = e[idx] + h * v_old
                    weighted_dot = np.dot(alpha_sub, h * target)
                    mean = var * (lam_v * mu_v + weighted_dot)
                    
                    v_new = rng.normal(mean, np.sqrt(var))
                    delta = v_old - v_new
                    
                    # Update states
                    e[idx] += delta * h
                    q[idx, f] -= x_vals * delta
                    v[j, f] = v_new

            # --- 4. Sample alpha (per group) ---
            # Posterior for alpha_c is Gamma(a_post, b_post)
            # a_post = a0 + N_c / 2
            # b_post = b0 + SSE_c / 2
            
            # Calculate SSE per group
            # e is current residuals
            sse_per_group = np.bincount(groups, weights=e**2, minlength=self.n_groups)
            n_per_group   = np.bincount(groups, minlength=self.n_groups)
            
            shape_vec = self.alpha_a0 + 0.5 * n_per_group
            rate_vec  = self.alpha_b0 + 0.5 * sse_per_group
            
            # Sample new alphas
            alpha_vec = rng.gamma(shape_vec, 1.0 / rate_vec)

            # --- Store samples ---
            if it >= n_burn:
                self.samples.append({
                    "w0": w0,
                    "w": w.copy(),
                    "v": v.copy(),
                    "alpha_vec": alpha_vec.copy()
                })

