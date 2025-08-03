import numpy as np
import scipy.sparse as sp
from scipy.stats import gamma
from tqdm.auto import trange


class BayesianFactorizationMachine:
    """
    Efficient O(k·Nₙz) Gibbs sampler for 2-way Bayesian Factorization Machines
    after Freudenthaler, Schmidt-Thieme & Rendle (2011).
    """

    def __init__(self, n_features: int, k: int):
        self.n_features = n_features
        self.k = k
        self.samples = []          # MCMC draws after burn-in

        # hyper-prior constants (Γ(1,1)   /  N(0,1))
        self.alpha_a0 = self.alpha_b0 = 1.0      # precision of likelihood
        self.gamma_0  = 1.0                      # prior precision of means
        self.mu_0     = 0.0                      # prior mean of means
        self.alpha_l  = self.beta_l  = 1.0       # Γ prior for λ_θ

    def fit(self, X: sp.csr_matrix, y: np.ndarray,
            n_iter: int = 150, n_burn: int = 50, random_state: int | None = None):

        rng = np.random.default_rng(random_state)
        n_obs, p = X.shape
        assert p == self.n_features

        # parameters
        w0 = 0.0
        w  = np.zeros(p)
        v  = rng.normal(0.0, 0.1, size=(p, self.k))

        # hyper-parameters (scalar) 
        alpha     = 1.0              # global precision
        mu_w      = 0.0; lam_w  = 1.0
        mu_v      = 0.0; lam_v  = 1.0

        # helpers 
        X2 = X.copy(); X2.data **= 2
        q  = X @ v                                # (n_obs, k)
        interact = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
        y_hat    = w0 + X @ w + interact
        e        = y - y_hat                      # residuals, 1-D float64

        # main loop --------------------------------------------------------
        for it in trange(n_iter, desc="BFM-Gibbs"):
            # ----- sample w0 ---------------------------------------------
            var_w0  = 1.0 / (self.gamma_0 + n_obs * alpha)
            mean_w0 = var_w0 * (self.gamma_0 * self.mu_0 + alpha * np.sum(e + w0))
            new_w0  = rng.normal(mean_w0, np.sqrt(var_w0))
            e += (w0 - new_w0)                    # fast residual update
            w0 = new_w0

            # ----- sample linear part w ----------------------------------
            #   hyper-means
            var_mu_w  = 1.0 / (self.gamma_0 + p * lam_w)
            mean_mu_w = var_mu_w * lam_w * w.sum()
            mu_w      = rng.normal(mean_mu_w, np.sqrt(var_mu_w))

            #   hyper-precision
            shape = self.alpha_l + 0.5 * p
            rate  = self.beta_l  + 0.5 * np.square(w - mu_w).sum()
            lam_w = rng.gamma(shape, 1.0 / rate)

            #   each coefficient w_j
            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0:
                    continue
                x   = col.data
                e_j = e[idx] + x * w[j]           # strip old contribution

                var = 1.0 / (lam_w + alpha * np.square(x).sum())
                mean = var * (lam_w * mu_w + alpha * np.dot(x, e_j))
                new_wj = rng.normal(mean, np.sqrt(var))

                e[idx] += x * (w[j] - new_wj)# apply new contribution
                w[j] = new_wj

            #sample latent matrix v 
            #   hyper-mean
            var_mu_v  = 1.0 / (self.gamma_0 + p * self.k * lam_v)
            mean_mu_v = var_mu_v * lam_v * v.sum()
            mu_v      = rng.normal(mean_mu_v, np.sqrt(var_mu_v))

            #   hyper-precision
            shape = self.alpha_l + 0.5 * p * self.k
            rate  = self.beta_l  + 0.5 * np.square(v - mu_v).sum()
            lam_v = rng.gamma(shape, 1.0 / rate)

            #   factors v_{jf}
            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0:
                    continue
                x = col.data

                for f in range(self.k):
                    q_f = q[idx, f]                    # q before this factor update
                    v_old = v[j, f]

                    #  helper: h_i = x * (q_f - x * v_old) 
                    h = x * (q_f - x * v_old)          
                    h_sq_sum = np.dot(h, h)           

                    #  posterior variance / mean 
                    denom = lam_v + alpha * h_sq_sum
                    denom = max(denom, 1e-12)          # clamp for num stability
                    var   = 1.0 / denom
                    mean  = var * (lam_v * mu_v +
                                alpha * np.dot(h, e[idx] + h * v_old))

                    v_new = rng.normal(mean, np.sqrt(var))
                    delta = v_old - v_new              

                    #  fast state updates 
                    e[idx]      += delta * h           
                    q[idx, f]   -= x * delta          
                    v[j, f]      = v_new


            #  sample global precision α 
            shape = self.alpha_a0 + 0.5 * n_obs
            rate  = self.alpha_b0 + 0.5 * np.dot(e, e)
            alpha = rng.gamma(shape, 1.0 / rate)

            #store 
            if it >= n_burn:
                self.samples.append(dict(w0=w0, w=w.copy(), v=v.copy(), alpha=alpha))

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        if not self.samples:
            raise RuntimeError("Call `fit` first; no posterior draws available.")

        X2 = X.copy(); X2.data **= 2
        preds = []
        for s in self.samples:
            w0, w, v = s["w0"], s["w"], s["v"]
            q  = X @ v
            inter = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
            preds.append(w0 + X @ w + inter)
        return np.mean(preds, axis=0)
