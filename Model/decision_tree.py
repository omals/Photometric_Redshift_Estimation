from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

# =================== Load & Split Data ===================
df = pd.read_csv("fulldata_rd2.csv") #change to fulldata_rd1.csv
print(df.info)
selected_features = [ 'u', 'g', 'r', 'i', 'z', 'petroRad_u',
    'petroRad_g', 'petroRad_i', 'petroRad_r', 'petroRad_z', 'petroFlux_u',
    'petroFlux_g', 'petroFlux_i', 'petroFlux_r', 'petroFlux_z',
    'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r', 'petroR50_z',
    'psfMag_u', 'psfMag_r', 'psfMag_g', 'psfMag_i', 'psfMag_z', 'expAB_u',
    'expAB_g', 'expAB_r', 'expAB_i', 'expAB_z', 'u_g', 'g_r', 'r_i', 'i_z']

X = df[selected_features].values
y = df['redshift'].values

# ========= Normalize features =========
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)

# Convert to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# ========= Define RMSE function =========
def rmse_torch(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

# ========= Single train-test split =========
print("\n🌳 Running Decision Tree Regression on a single train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# Train Decision Tree
dt_model = DecisionTreeRegressor(max_depth=12, random_state=42)
dt_model.fit(X_train.numpy(), y_train.numpy().ravel())

# Predict and evaluate
preds = dt_model.predict(X_test.numpy())
preds_tensor = torch.tensor(preds, dtype=torch.float32).reshape(-1, 1)
rmse_val = rmse_torch(y_test, preds_tensor)

print(f"📊 Decision Tree RMSE on test set: {rmse_val:.5f}")
