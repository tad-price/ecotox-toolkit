import numpy as np
import scipy.sparse as sp
from tqdm.auto import trange

class HierarchicalBFM:
    """
    Gibbs sampler for 2-way Bayesian Factorization Machine with Hierarchical Aleatoric Uncertainty.
    """

    def __init__(self, n_features: int, n_groups: int, k: int):
        self.n_features = n_features
        self.n_groups = n_groups
        self.k = k
        self.samples = []

        self.alpha_a0 = 1.0
        self.alpha_b0 = 1.0
        
        self.gamma_0  = 1.0
        self.mu_0     = 0.0
        self.alpha_l  = self.beta_l  = 1.0

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
        alpha_vec = np.ones(self.n_groups) 

        mu_w = 0.0; lam_w = 1.0
        mu_v = 0.0; lam_v = 1.0

        X2 = X.copy(); X2.data **= 2
        q  = X @ v
        interact = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
        y_hat    = w0 + X @ w + interact
        e        = y - y_hat

        for it in trange(n_iter, desc="HierarchicalBFM"):
            
            # Sample w0
            alpha_obs = alpha_vec[groups]
            sum_alpha = np.sum(alpha_obs)
            var_w0 = 1.0 / (self.gamma_0 + sum_alpha)
            
            weighted_res = np.dot(alpha_obs, e + w0)
            mean_w0 = var_w0 * (self.gamma_0 * self.mu_0 + weighted_res)
            
            new_w0 = rng.normal(mean_w0, np.sqrt(var_w0))
            e += (w0 - new_w0)
            w0 = new_w0

            # Sample w
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
                alpha_sub = alpha_vec[groups[idx]]
                
                e_j = e[idx] + x_vals * w[j]
                
                weighted_x2 = np.dot(alpha_sub, x_vals**2)
                var = 1.0 / (lam_w + weighted_x2)
                
                weighted_dot = np.dot(alpha_sub, x_vals * e_j)
                mean = var * (lam_w * mu_w + weighted_dot)
                
                new_wj = rng.normal(mean, np.sqrt(var))
                
                # Update residuals
                e[idx] += x_vals * (w[j] - new_wj)
                w[j] = new_wj

            # Sample v
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
                    
                    h = x_vals * (q_f - x_vals * v_old)
                    
                    weighted_h2 = np.dot(alpha_sub, h**2)
                    denom = lam_v + weighted_h2
                    denom = max(denom, 1e-12)
                    var = 1.0 / denom
                    
                    target = e[idx] + h * v_old
                    weighted_dot = np.dot(alpha_sub, h * target)
                    mean = var * (lam_v * mu_v + weighted_dot)
                    
                    v_new = rng.normal(mean, np.sqrt(var))
                    delta = v_old - v_new
                    
                    # Update states
                    e[idx] += delta * h
                    q[idx, f] -= x_vals * delta
                    v[j, f] = v_new

            # Sample alpha (per group)
            sse_per_group = np.bincount(groups, weights=e**2, minlength=self.n_groups)
            n_per_group   = np.bincount(groups, minlength=self.n_groups)
            
            shape_vec = self.alpha_a0 + 0.5 * n_per_group
            rate_vec  = self.alpha_b0 + 0.5 * sse_per_group
            
            alpha_vec = rng.gamma(shape_vec, 1.0 / rate_vec)

            if it >= n_burn:
                self.samples.append({
                    "w0": w0,
                    "w": w.copy(),
                    "v": v.copy(),
                    "alpha_vec": alpha_vec.copy()
                })

