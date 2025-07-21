# file: models/GibbsBFM.py (Final, Corrected Version)

import numpy as np
import scipy.sparse as sp
from tqdm import trange

class GibbsBFM:
    """
    Bayesian Factorization Machine with a memory-efficient and shape-correct Gibbs Sampler.
    """
    def __init__(
        self,
        n_features: int,
        k: int,
        alpha_a0: float = 1.0, alpha_b0: float = 1.0,
        lambda_b: float = 1.0, lambda_w: float = 1.0, lambda_v: float = 1.0,
    ):
        self.n_features = n_features
        self.k = k
        self.samples = []
        self.alpha_a0, self.alpha_b0 = alpha_a0, alpha_b0
        self.lambda_b, self.lambda_w, self.lambda_v = lambda_b, lambda_w, lambda_v

    def fit(self, X: sp.csr_matrix, y: np.ndarray, n_iter: int = 200, n_burnin: int = 100):
        n_obs, n_features = X.shape
        
        bias = 0.0
        linear_w = np.zeros(n_features)
        factors_v = np.random.normal(scale=0.1, size=(self.n_features, self.k))
        precision_alpha = 1.0

        self.samples = []
        X_sq = X.copy(); X_sq.data **= 2

        # --- Initial full prediction ---
        XV = X.dot(factors_v)
        interaction_term = 0.5 * np.sum(XV**2 - X_sq.dot(factors_v**2), axis=1)
        y_pred = bias + X.dot(linear_w) + interaction_term

        print(f"Starting Gibbs sampling for {n_iter} iterations ({n_burnin} burn-in)...")
        for it in trange(n_iter, desc="Gibbs Sampling"):
            # --- Sample global bias (b) ---
            y_pred -= bias
            residuals = y - y_pred
            b_var = 1.0 / (self.lambda_b * precision_alpha + n_obs * precision_alpha)
            b_mean = b_var * (np.sum(residuals) * precision_alpha)
            bias = np.random.normal(loc=b_mean, scale=np.sqrt(b_var))
            y_pred += bias

            # --- Sample linear weights (w_j) ---
            for j in range(n_features):
                x_j = X[:, j]
                old_w_j = linear_w[j]
                
                y_pred -= (x_j * old_w_j).toarray().ravel()
                residuals = y - y_pred
                
                w_var_inv = self.lambda_w * precision_alpha + precision_alpha * x_j.power(2).sum()
                if w_var_inv == 0: continue
                w_mean = (1.0 / w_var_inv) * (precision_alpha * x_j.T.dot(residuals)[0])
                
                new_w_j = np.random.normal(loc=w_mean, scale=np.sqrt(1.0 / w_var_inv))
                
                y_pred += (x_j * new_w_j).toarray().ravel()
                linear_w[j] = new_w_j

            # --- Sample latent factors (v_jf) ---
            XV = X.dot(factors_v)
            for j in range(n_features):
                x_j = X[:, j]
                x_j_sq = X_sq[:, j]
                if x_j.nnz == 0: continue # Skip if feature j is not present in data

                for f in range(self.k):
                    v_old = factors_v[j, f]
                    
                    # h_jf is the dot product <x, v_f> without feature j's contribution
                    h_jf = XV[:, f] - (x_j * v_old).toarray().ravel()
                    
                    # The full interaction term is a sum over f.
                    # For a single v_jf, the model is linear: y = C + v_jf * (x_j * h_jf)
                    # Contribution of v_jf to y_pred is v_jf * (x_j * h_jf)
                    
                    # <<< FIX: Reshape h_jf to (N, 1) for .multiply() >>>
                    old_contrib = (x_j.multiply(h_jf.reshape(-1, 1)) * v_old).toarray().ravel()
                    y_pred -= old_contrib
                    
                    residuals = y - y_pred
                    
                    # <<< FIX: Reshape h_jf for the g_jf calculation as well >>>
                    g_jf = x_j.multiply(h_jf.reshape(-1, 1)).toarray().ravel()
                    
                    v_var_inv = self.lambda_v * precision_alpha + precision_alpha * np.dot(g_jf, g_jf)
                    if v_var_inv == 0:
                        y_pred += old_contrib # Add back if no update, then continue
                        continue
                        
                    v_mean = (1.0 / v_var_inv) * (precision_alpha * np.dot(residuals, g_jf))
                    v_new = np.random.normal(loc=v_mean, scale=np.sqrt(1.0 / v_var_inv))
                    
                    # Add new contribution and update XV for the next iteration
                    # <<< FIX: Ensure correct shapes for update >>>
                    new_contrib = g_jf * v_new
                    y_pred += new_contrib
                    XV[:, f] += (x_j * (v_new - v_old)).toarray().ravel()
                    factors_v[j, f] = v_new

            # --- Sample observation precision (α) ---
            # Recompute final prediction to avoid floating point drift
            final_linear = X.dot(linear_w)
            final_XV = X.dot(factors_v)
            final_interaction = 0.5 * np.sum(final_XV**2 - X_sq.dot(factors_v**2), axis=1)
            y_pred_final = bias + final_linear + final_interaction
            
            sse = np.sum(np.power(y - y_pred_final, 2))
            alpha_a_post = self.alpha_a0 + n_obs / 2.0
            alpha_b_post = self.alpha_b0 + sse / 2.0
            precision_alpha = np.random.gamma(shape=alpha_a_post, scale=1.0 / alpha_b_post)

            if it >= n_burnin:
                self.samples.append({
                    "bias": bias, "linear_w": linear_w.copy(),
                    "factors_v": factors_v.copy(), "precision_alpha": precision_alpha
                })

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        # The predict method was already correct
        if not self.samples:
            raise RuntimeError("Model has not been trained or no samples were collected.")

        y_preds = []
        X_sq = X.copy()
        X_sq.data **= 2

        for sample in self.samples:
            b, w, v = sample["bias"], sample["linear_w"], sample["factors_v"]
            pred = b + X.dot(w) + 0.5 * np.sum(
                np.power(X.dot(v), 2) - X_sq.dot(np.power(v, 2)),
                axis=1
            )
            y_preds.append(pred)

        return np.mean(y_preds, axis=0)