# =================== Imports ===================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold

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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# =================== GPU Setup ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"🖥️ Using {num_gpus} GPUs")


# =================== Neural Network Definition ===================
class RedshiftEmbeddingModel(nn.Module):
    def __init__(self, input_dim=9, embedding_size=32):
        super(RedshiftEmbeddingModel, self).__init__()
        self.dense1 = nn.Linear(input_dim, 128)
        self.norm1 = nn.LayerNorm(128)
        self.dense2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.embedding = nn.Linear(64, embedding_size)
        self.output = nn.Linear(embedding_size, 1)

    def forward(self, x):
        x = F.relu(self.norm1(self.dense1(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.norm2(self.dense2(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        embedding = F.relu(self.embedding(x))
        output = self.output(embedding)
        return output, embedding

# =================== Training Function ===================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

    return train_losses, val_losses

# =================== Evaluation Function ===================
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        preds, _ = model(torch.tensor(X, dtype=torch.float32).to(device))
    preds = preds.cpu().numpy().flatten()
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return rmse, r2, preds

# =================== Data Preparation ===================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

torch_train = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
)
torch_val = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
)

train_loader = DataLoader(torch_train, batch_size=512, shuffle=True, pin_memory=True)
val_loader = DataLoader(torch_val, batch_size=512, pin_memory=True)

# =================== Model Setup (GPU + Parallel) ===================
model = RedshiftEmbeddingModel(input_dim=X_train.shape[1], embedding_size=32)

if num_gpus > 1:
    print("⚡ Using DataParallel across multiple GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

# =================== Train the Model ===================
train_losses, val_losses = train_model(model, train_loader, val_loader)

# =================== Plot Loss Curves ===================
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =================== Evaluation ===================
train_rmse, train_r2, _ = evaluate_model(model, X_train_scaled, y_train)
val_rmse, val_r2, _ = evaluate_model(model, X_val_scaled, y_val)
test_rmse, test_r2, test_preds = evaluate_model(model, X_test_scaled, y_test)

print(f"\nTrain RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
print(f"Val   RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
print(f"Test  RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

# =================== Extract Embeddings ===================
model.eval()
with torch.no_grad():
    _, train_embed = model(torch.tensor(X_train_scaled, dtype=torch.float32).to(device))
    _, val_embed = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device))
    _, test_embed = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device))

train_embed = train_embed.cpu().numpy()
val_embed = val_embed.cpu().numpy()
test_embed = test_embed.cpu().numpy()

# =================== Combine Features + Embeddings ===================
X_train_hybrid = np.hstack((X_train_scaled, train_embed))
X_val_hybrid = np.hstack((X_val_scaled, val_embed))
X_test_hybrid = np.hstack((X_test_scaled, test_embed))



# =================== XGBoost (GPU Mode) ===================
xgb_model = xgb.XGBRegressor(
    n_estimators=900,
    max_depth=14,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'                  # Use GPU 0
)
xgb_model.fit(X_train_hybrid, y_train)

# =================== Evaluation ===================
def xgb_metrics(model, X, y):
    preds = model.predict(X)
    return {
        'rmse': np.sqrt(mean_squared_error(y, preds)),
        'r2': r2_score(y, preds)
    }

xgb_results = {
    'train': xgb_metrics(xgb_model, X_train_hybrid, y_train),
    'val': xgb_metrics(xgb_model, X_val_hybrid, y_val),
    'test': xgb_metrics(xgb_model, X_test_hybrid, y_test)
}

print("\n✅ Final Model Performance with XGBoost:")
for split, scores in xgb_results.items():
    print(f"{split.upper()} - RMSE: {scores['rmse']:.4f}, R²: {scores['r2']:.4f}")

# =================== Scatter Plot ===================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, xgb_model.predict(X_test_hybrid), alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('True Redshift')
plt.ylabel('Predicted Redshift')
plt.title('Prediction Accuracy - XGBoost')
plt.grid(True)
plt.tight_layout()
plt.show()