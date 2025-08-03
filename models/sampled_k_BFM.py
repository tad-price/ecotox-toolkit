import numpy as np
import scipy.sparse as sp
from tqdm.auto import trange

class BayesianFactorizationMachineARD:
    """
    Efficient O(k·Nₙz) Gibbs sampler for 2-way Bayesian Factorization Machines
    with Automatic Relevance Determination (ARD) for the latent dimensionality k.
    """

    def __init__(self, n_features: int, k: int):
        self.n_features = n_features
        self.k = k  # This is the maximum k
        self.samples = []

        # Hyper-prior constants (Γ(1,1) / N(0,1))
        self.alpha_a0 = self.alpha_b0 = 1.0
        self.gamma_0 = 1.0
        self.mu_0 = 0.0
        self.alpha_l = self.beta_l = 1.0

    def fit(self, X: sp.csr_matrix, y: np.ndarray,
            n_iter: int = 150, n_burn: int = 50, random_state: int | None = None):

        rng = np.random.default_rng(random_state)
        n_obs, p = X.shape
        assert p == self.n_features

        # Parameters
        w0 = 0.0
        w = np.zeros(p)
        v = rng.normal(0.0, 0.1, size=(p, self.k))

        # Hyper-parameters
        alpha = 1.0
        mu_w = 0.0
        lam_w = 1.0
        
        # --- ARD specific hyper-parameters ---
        mu_v = np.zeros(self.k)  # Vector of means for v
        lam_v = np.ones(self.k)   # Vector of precisions for v

        # Helpers
        X2 = X.copy(); X2.data **= 2
        q = X @ v
        interact = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
        y_hat = w0 + X @ w + interact
        e = y - y_hat

        # Main Gibbs loop
        for it in trange(n_iter, desc="BFM-ARD-Gibbs"):
            # ----- Sample w0 -----
            var_w0 = 1.0 / (self.gamma_0 + n_obs * alpha)
            mean_w0 = var_w0 * (self.gamma_0 * self.mu_0 + alpha * np.sum(e + w0))
            new_w0 = rng.normal(mean_w0, np.sqrt(var_w0))
            e += (w0 - new_w0)
            w0 = new_w0

            # ----- Sample linear part w -----
            var_mu_w = 1.0 / (self.gamma_0 + p * lam_w)
            mean_mu_w = var_mu_w * lam_w * w.sum()
            mu_w = rng.normal(mean_mu_w, np.sqrt(var_mu_w))

            shape_w = self.alpha_l + 0.5 * p
            rate_w = self.beta_l + 0.5 * np.square(w - mu_w).sum()
            lam_w = rng.gamma(shape_w, 1.0 / rate_w)

            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0: continue
                x = col.data
                e_j = e[idx] + x * w[j]
                var = 1.0 / (lam_w + alpha * np.square(x).sum())
                mean = var * (lam_w * mu_w + alpha * np.dot(x, e_j))
                new_wj = rng.normal(mean, np.sqrt(var))
                e[idx] += x * (w[j] - new_wj)
                w[j] = new_wj

            # ----- Sample latent matrix v (with ARD) -----
            #   Hyper-parameters (mean and precision for each factor)
            for f in range(self.k):
                # Update mu_v[f]
                var_mu_vf = 1.0 / (self.gamma_0 + p * lam_v[f])
                mean_mu_vf = var_mu_vf * lam_v[f] * v[:, f].sum()
                mu_v[f] = rng.normal(mean_mu_vf, np.sqrt(var_mu_vf))

                # Update lam_v[f]
                shape_f = self.alpha_l + 0.5 * p
                rate_f = self.beta_l + 0.5 * np.square(v[:, f] - mu_v[f]).sum()
                lam_v[f] = rng.gamma(shape_f, 1.0 / rate_f)

            #   Factors v_{jf}
            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0: continue
                x = col.data

                for f in range(self.k):
                    q_f = q[idx, f]
                    v_old = v[j, f]

                    h = x * (q_f - x * v_old)
                    h_sq_sum = np.dot(h, h)

                    # Posterior variance/mean using factor-specific hyper-params
                    denom = lam_v[f] + alpha * h_sq_sum
                    denom = max(denom, 1e-12)
                    var = 1.0 / denom
                    mean = var * (lam_v[f] * mu_v[f] + alpha * np.dot(h, e[idx] + h * v_old))

                    v_new = rng.normal(mean, np.sqrt(var))
                    delta = v_old - v_new

                    e[idx] += delta * h
                    q[idx, f] -= x * delta
                    v[j, f] = v_new

            # ----- Sample global precision α -----
            shape_alpha = self.alpha_a0 + 0.5 * n_obs
            rate_alpha = self.alpha_b0 + 0.5 * np.dot(e, e)
            alpha = rng.gamma(shape_alpha, 1.0 / rate_alpha)

            # ----- Store sample -----
            if it >= n_burn:
                self.samples.append(dict(
                    w0=w0, w=w.copy(), v=v.copy(), alpha=alpha, lam_v=lam_v.copy()
                ))

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        if not self.samples:
            raise RuntimeError("Call `fit` first; no posterior draws available.")

        X2 = X.copy(); X2.data **= 2
        preds = []
        for s in self.samples:
            w0, w, v = s["w0"], s["w"], s["v"]
            q = X @ v
            inter = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
            preds.append(w0 + X @ w + inter)
        return np.mean(preds, axis=0)