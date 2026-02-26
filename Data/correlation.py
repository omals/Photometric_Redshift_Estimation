# venv/Scr---------------------------
# • Computes Pearson correlation of all numeric features vs redshift
# • Saves correlation table to CSV
# • Generates and saves publication-ready figures


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# USER CONFIGURATION
# ============================
DATA_PATH = "cleaned_data.csv"     # 🔁 CHANGE THIS
TARGET = "redshift"                # 🔁 CHANGE IF NEEDED
MAX_SAMPLE =  3600000          # Set None to disable sampling
FIG_DIR = "figures"

# ============================
# CREATE FIGURE DIRECTORY
# ============================
os.makedirs(FIG_DIR, exist_ok=True)

# ============================
# LOAD DATA
# ============================
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows and {df.shape[1]} columns")






# ============================
# OPTIONAL SAMPLING
# ============================
if MAX_SAMPLE and len(df) > MAX_SAMPLE:
    df = df.sample(n=MAX_SAMPLE, random_state=42)
    print(f"⚠️ Sampled down to {MAX_SAMPLE:,} rows for efficiency")

# ============================
# NUMERIC FEATURES ONLY
# ============================
numeric_df = df.select_dtypes(include=[np.number])

if TARGET not in numeric_df.columns:
    raise ValueError(f"❌ Target column '{TARGET}' not found or not numeric")

features = numeric_df.columns.drop(TARGET)

# ============================
# PEARSON CORRELATION
# ============================
print("📊 Computing Pearson correlations...")
pearson_values = numeric_df[features].corrwith(numeric_df[TARGET])

corr_df = (
    pearson_values
    .to_frame(name="Pearson")
    .assign(abs_pearson=lambda x: x["Pearson"].abs())
    .sort_values("abs_pearson", ascending=False)
)

# ============================
# SAVE CORRELATION TABLE
# ============================
csv_path = os.path.join(FIG_DIR, "pearson_correlation_all_features.csv")
corr_df.to_csv(csv_path)
print(f"✅ Correlation table saved to: {csv_path}")

# ============================
# BAR PLOT (ALL FEATURES)
# ============================
plt.figure(figsize=(10, max(6, 0.25 * len(corr_df))))
corr_df["Pearson"].plot(kind="barh", color="#1f77b4")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Pearson Correlation")
plt.ylabel("Features")
plt.title("Pearson Correlation vs Redshift")
plt.tight_layout()

fig_all = os.path.join(FIG_DIR, "pearson_all_features.png")
plt.savefig(fig_all, dpi=300)
plt.close()
print(f"🖼️ Saved figure: {fig_all}")

# ============================
# TOP & BOTTOM FEATURES PLOT
# ============================
TOP_K = 15
top_feats = corr_df.head(TOP_K)
bottom_feats = corr_df.tail(TOP_K)

combined = pd.concat([top_feats, bottom_feats])

plt.figure(figsize=(10, 8))
combined["Pearson"].plot(kind="barh", color="#2ca02c")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Pearson Correlation")
plt.title("Top & Bottom Pearson-Correlated Features")
plt.tight_layout()

fig_tb = os.path.join(FIG_DIR, "pearson_top_bottom.png")
plt.savefig(fig_tb, dpi=300)
plt.close()
print(f"🖼️ Saved figure: {fig_tb}")

# ============================
# SUMMARY
# ============================
print("\n🔎 Top 10 Correlated Features:")
print(corr_df.head(10))

print("\n🔎 Bottom 10 Correlated Features:")
print(corr_df.tail(10))

print("\n✅ Correlation analysis completed successfully.")