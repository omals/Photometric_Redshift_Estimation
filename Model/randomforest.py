import pandas as pd
import numpy as np
import cupy as cp  # For GPU array operations
import cuml
from cuml.ensemble import RandomForestRegressor
from cuml.metrics import mean_squared_error as cuml_mse
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ========= Load and preprocess the data =========
df = pd.read_csv("fulldata_rd2.csv") #change to "fulldata_rd1.csv"
print(df.info)
features = [ 'u', 'g', 'r', 'i', 'z', 'petroRad_u',
    'petroRad_g', 'petroRad_i', 'petroRad_r', 'petroRad_z', 'petroFlux_u',
    'petroFlux_g', 'petroFlux_i', 'petroFlux_r', 'petroFlux_z',
    'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z',
    'psfMag_u', 'psfMag_r', 'psfMag_g', 'psfMag_i', 'psfMag_z', 'expAB_u',
    'expAB_g', 'expAB_r', 'expAB_i', 'expAB_z', 'u_g', 'g_r', 'r_i', 'i_z']

X = df[features].values.astype(np.float32)
y = df["redshift"].values.astype(np.float32).reshape(-1, 1)

scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)

# Convert to GPU arrays
X_gpu = cp.asarray(X_scaled)
y_gpu = cp.asarray(y)

# ========= Hyperparameter grid =========
n_estimators_list = [200,300, 400, 500, 600, 700, 800, 900]
max_depth_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20]

best_rmse = float("inf")
best_params = None
results = []

# ========= Grid Search =========
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for n_est in n_estimators_list:
    for max_d in max_depth_list:
        rmse_scores = []
        r2_scores = []

        for train_idx, val_idx in kf.split(X_gpu):
            X_train, X_val = X_gpu[train_idx], X_gpu[val_idx]
            y_train, y_val = y_gpu[train_idx], y_gpu[val_idx]

            rf_model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                             random_state=42, n_streams=4, verbose=0)
            rf_model.fit(X_train, y_train.ravel())

            preds = rf_model.predict(X_val)
            rmse_val = cp.sqrt(cuml_mse(y_val, preds)).item()
            r2_val = r2_score(cp.asnumpy(y_val), cp.asnumpy(preds))

            rmse_scores.append(rmse_val)
            r2_scores.append(r2_val)

        mean_rmse = np.mean(rmse_scores)
        mean_r2 = np.mean(r2_scores)

        results.append({
            "n_estimators": n_est,
            "max_depth": max_d,
            "rmse": mean_rmse,
            "r2": mean_r2
        })

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = (n_est, max_d)

        print(f"n_estimators={n_est}, max_depth={max_d} => RMSE: {mean_rmse:.5f}, R²: {mean_r2:.5f}")

# ========= Convert results to DataFrame =========
ablation_df = pd.DataFrame(results)

# ========= Plot Results =========
plt.figure(figsize=(14, 6))

# Plot: RMSE vs n_estimators
plt.subplot(1, 2, 1)
for depth in sorted(ablation_df['max_depth'].unique()):
    subset = ablation_df[ablation_df['max_depth'] == depth]
    plt.plot(subset['n_estimators'], subset['rmse'], marker='o', label=f'max_depth={depth}')
plt.title('RMSE vs n_estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Test RMSE')
plt.legend()

# Plot: RMSE vs max_depth
plt.subplot(1, 2, 2)
for n_est in sorted(ablation_df['n_estimators'].unique()):
    subset = ablation_df[ablation_df['n_estimators'] == n_est]
    plt.plot(subset['max_depth'], subset['rmse'], marker='o', label=f'n_estimators={n_est}')
plt.title('RMSE vs max_depth')
plt.xlabel('Max Depth')
plt.ylabel('Test RMSE')
plt.legend()

plt.tight_layout()
plt.show()

# ========= Final Summary =========
print("\n✅ Final Model Performance:")
for row in results:
    print(f"n_estimators={row['n_estimators']}, max_depth={row['max_depth']} - RMSE: {row['rmse']:.4f}, R²: {row['r2']:.4f}")

print(f"\n🏆 Best Params: n_estimators={best_params[0]}, max_depth={best_params[1]} with RMSE={best_rmse:.5f}")
