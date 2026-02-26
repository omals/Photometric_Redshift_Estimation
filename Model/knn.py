import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ========= Load and preprocess the data =========
df = pd.read_csv("fulldata_rd2.csv") #change to fullData_rd1.csv 
print(df.info)
features = [ 'u', 'g', 'r', 'i', 'z', 'petroRad_u',
    'petroRad_g', 'petroRad_i', 'petroRad_r', 'petroRad_z', 'petroFlux_u',
    'petroFlux_g', 'petroFlux_i', 'petroFlux_r', 'petroFlux_z',
    'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z',
    'psfMag_u', 'psfMag_r', 'psfMag_g', 'psfMag_i', 'psfMag_z', 'expAB_u',
    'expAB_g', 'expAB_r', 'expAB_i', 'expAB_z', 'u_g', 'g_r', 'r_i', 'i_z']

X = df[features].values.astype(np.float32)
y = df["redshift"].values.astype(np.float32).reshape(-1, 1)

# ========== Normalize and Split ==========
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# ========== Setup FAISS on CPU ==========
index = faiss.IndexFlatL2(X_train.shape[1])  # L2 distance (CPU index)

# ========== Add training vectors ==========
index.add(X_train)

# ========== Evaluate for different k ==========
k_values = list(range(1, 31))
rmse_values = []

max_k = max(k_values)
distances, indices = index.search(X_test, max_k)

rmse_values = []
for k in k_values:
    print("k=",k)
    preds = np.mean(y_train[indices[:, :k]], axis=1)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    rmse_values.append(rmse)
    print(f"k={k}, RMSE={rmse:.5f}")


# ========== Plot RMSE vs k ==========
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o', color='royalblue')
plt.scatter(k_values[np.argmin(rmse_values)], min(rmse_values),
            color='red', label=f'Min RMSE = {min(rmse_values):.5f}')
plt.title("CPU-based KNN Redshift Regression - RMSE vs. k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("RMSE")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
