import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------
# 1. DEFINICE VAŠICH FAKTORŮ A JEJICH HLADIN
# ---------------------------------------------------------
factors = {
    'level': [2, 3, 4, 5],                                                   # 4
    'head': ['linear', 'mlp'],                                               # 2
    'aggregator': ['mean', 'max', 'attention', 'transformer'],              # 4
    'loss': ['bce', 'regression', 'joint'],                                  # 3
    'mask_type': ['tissue_only', 'cancer_mask_5', 'cancer_mask_9', 'epithel', 'epithel_and_cancer_5', 'epithel_or_cancer_5']          # 5
}

# Vytvoření plné mřížky (Full Grid Search = 4 * 2 * 4 * 3 * 3 = 288 kombinací)
grid_tuples = pd.MultiIndex.from_product(factors.values(), names=factors.keys())
full_grid = grid_tuples.to_frame().reset_index(drop=True)

print(f"Celkový počet kombinací v plné mřížce (Full Grid): {len(full_grid)}")

# ---------------------------------------------------------
# 2. PŘEVOD NA KATEGORICKOU MATICI (Dummy / One-Hot Encoding)
# ---------------------------------------------------------
# Jelikož jsou parametry kategorické, převedeme je na 0/1 matici pro výpočet determinantu
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_full = encoder.fit_transform(full_grid)

# Přidání sloupce jedniček pro absolutní člen (intercept b0)
X_full_intercept = np.hstack([np.ones((X_full.shape[0], 1)), X_full])
p = X_full_intercept.shape[1]  # Počet odhadovaných parametrů v modelu

print(f"Počet volných parametrů k odhadnutí v modelu (p): {p}")

# ---------------------------------------------------------
# 3. ALGORITMUS PRO D-OPTIMAL SELECTION (Fedorov's Exchange Idea)
# ---------------------------------------------------------
def compute_d_efficiency(X_sub, X_full_matrix):
    """Spočítá D-efektivitu vybraného vzorku vůči celkovému vzorku."""
    N_sub = X_sub.shape[0]
    N_full = X_full_matrix.shape[0]

    # Matice informace M = X^T * X
    M_sub = np.dot(X_sub.T, X_sub)
    M_full = np.dot(X_full_matrix.T, X_full_matrix)

    sign_sub, logdet_sub = np.linalg.slogdet(M_sub)
    sign_full, logdet_full = np.linalg.slogdet(M_full)

    if sign_sub <= 0:
        return 0.0  # Singulární matice (nedostatek dat pro odhad)

    # Relativní D-efficiency vzorec v logaritmickém tvaru pro numerickou stabilitu
    det_ratio_per_param = np.exp((logdet_sub - logdet_full) / p)
    d_eff = (det_ratio_per_param) * (N_full / N_sub)
    return d_eff * 100

def generate_d_optimal_design(X_matrix, full_df, n_samples, n_iterations=100):
    """
    Vybere n_samples řádků tak, aby maximizoval determinant X^T * X.
    """
    best_det = -np.inf
    best_indices = None

    np.random.seed(42) # Pro reprodukovatelnost

    for _ in range(n_iterations):
        # Náhodný počáteční výběr
        current_indices = np.random.choice(X_matrix.shape[0], size=n_samples, replace=False)

        # Lokální vylepšování determinantu (Exchange algorithm)
        for i in range(n_samples):
            current_X = X_matrix[current_indices]
            sign, current_det = np.linalg.slogdet(np.dot(current_X.T, current_X))

            if sign <= 0:
                current_det = -np.inf

            # Zkusíme zaměnit i-tý prvek za jiný kandidát
            candidates = np.setdiff1d(np.arange(X_matrix.shape[0]), current_indices)
            for cand in np.random.choice(candidates, size=min(30, len(candidates)), replace=False):
                temp_indices = current_indices.copy()
                temp_indices[i] = cand

                temp_X = X_matrix[temp_indices]
                temp_sign, temp_det = np.linalg.slogdet(np.dot(temp_X.T, temp_X))

                if temp_sign > 0 and temp_det > current_det:
                    current_det = temp_det
                    current_indices = temp_indices.copy()

        if current_det > best_det:
            best_det = current_det
            best_indices = current_indices

    selected_df = full_df.iloc[best_indices].copy().reset_index(drop=True)
    d_eff = compute_d_efficiency(X_matrix[best_indices], X_matrix)

    return selected_df, d_eff

# ---------------------------------------------------------
# 4. GENEROVÁNÍ DESIGNU PRO 36 A 48 KOMBINACÍ
# ---------------------------------------------------------
print("\nGeneruji D-Optimal Design pro 36 kombinací...")
design_36, d_eff_36 = generate_d_optimal_design(X_full_intercept, full_grid, n_samples=36)
print(f"-> D-Efektivita pro 36 kombinací: {d_eff_36:.2f} %")

print("\nGeneruji D-Optimal Design pro 48 kombinací...")
design_48, d_eff_48 = generate_d_optimal_design(X_full_intercept, full_grid, n_samples=48)
print(f"-> D-Efektivita pro 48 kombinací: {d_eff_48:.2f} %")

# ---------------------------------------------------------
# 5. ULOŽENÍ VÝSLEDKŮ DO CSV
# ---------------------------------------------------------
design_36.to_csv("configs/sweeps/d_optimal_mask_comparison/d_optimal_36_combinations.csv", index=False)
design_48.to_csv("configs/sweeps/d_optimal_mask_comparison/d_optimal_48_combinations.csv", index=False)

print("\nSoubory 'configs/sweeps/d_optimal_mask_comparison/d_optimal_36_combinations.csv' a 'configs/sweeps/d_optimal_mask_comparison/d_optimal_48_combinations.csv' byly úspěšně vytvořeny.")
