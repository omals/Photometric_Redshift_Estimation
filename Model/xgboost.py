# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb 

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

# ==================== SCALING ====================
scaler = StandardScaler()

# Ensure data are NumPy arrays, not CuPy
X_train = np.array(X_train.get() if hasattr(X_train, "get") else X_train)
X_val = np.array(X_val.get() if hasattr(X_val, "get") else X_val)
X_test = np.array(X_test.get() if hasattr(X_test, "get") else X_test)

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ==================== ABLATION STUDY ====================
n_estimators_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]
max_depth_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18]
ablation_results = []

for n_est in n_estimators_list:
    for depth in max_depth_list:
        print(f"\n🔍 Training XGBoost with n_estimators={n_est}, max_depth={depth}", end=" ")
        model = xgb.XGBRegressor(
            n_estimators=n_est,
            learning_rate=0.02,
            max_depth=depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)

        ablation_results.append({
            'n_estimators': n_est,
            'max_depth': depth,
            'rmse': test_rmse,
            'r2': test_r2
        })
        print('rmse =', test_rmse, "'r^2 =", test_r2)

ablation_df = pd.DataFrame(ablation_results)

# ==================== BOXPLOTS ====================
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=ablation_df, x='n_estimators', y='rmse')
plt.title('RMSE vs n_estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Test RMSE')

plt.subplot(1, 2, 2)
sns.boxplot(data=ablation_df, x='max_depth', y='rmse')
plt.title('RMSE vs max_depth')
plt.xlabel('Max Depth')
plt.ylabel('Test RMSE')

plt.tight_layout()
plt.show()

# ==================== SCATTER PLOTS ====================
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

# ==================== RESULTS ====================
print("\n✅ Final Model Performance:")
for i, row in ablation_df.iterrows():
    print(f"n_estimators={row['n_estimators']}, max_depth={row['max_depth']} - RMSE: {row['rmse']:.4f}, R²: {row['r2']:.4f}")

best_model = ablation_df.loc[ablation_df['rmse'].idxmin()]
print(f"\n🏆 Best configuration: n_estimators={best_model['n_estimators']}, "
      f"max_depth={best_model['max_depth']} with RMSE={best_model['rmse']:.4f}, R²={best_model['r2']:.4f}")
