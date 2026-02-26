# =================== Imports ===================
import numpy as np
import pandas as pd
import time
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# =================== Data Preprocessing ===================

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

scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =================== Ablation Parameters ===================
C_values = [0.1, 1, 3, 5,7,9, 10]
epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

results = []

# =================== Ablation Loop ===================
print("\n🚀 Starting Regularization (C) and Margin (ε) Ablation...\n")

for C in C_values:
    for eps in epsilon_values:
        print(f"⚙️ Training LinearSVR with C={C}, ε={eps} ...")

        model = LinearSVR(
            C=C,
            epsilon=eps,
            random_state=42,
            max_iter=10000
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Store results
        results.append({
            "C": C,
            "epsilon": eps,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Train R²": train_r2,
            "Test R²": test_r2,
            "Train Time (s)": elapsed
        })

# =================== Results Summary ===================
results_df = pd.DataFrame(results)
print("\n📊 Regularization & Margin Ablation Results:")
print(results_df.round(5))

# Create images folder
import os
os.makedirs("images", exist_ok=True)

# =================== Visualization ===================

# ----- Plot 1: Test RMSE -----
plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x="C", y="Test RMSE", hue="epsilon", marker="o")
plt.title("Effect of Regularization (C) and Margin (ε) on Test RMSE")
plt.xlabel("Regularization (C)")
plt.ylabel("Test RMSE")
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig("images/linear_svr_rmse.jpeg", format="jpeg", dpi=300)
plt.show()

# ----- Plot 2: Test R² -----
plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x="C", y="Test R²", hue="epsilon", marker="o")
plt.title("Effect of Regularization (C) and Margin (ε) on Test R²")
plt.xlabel("Regularization (C)")
plt.ylabel("Test R²")
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig("images/linear_svr_r2.jpeg", format="jpeg", dpi=300)
plt.show()

