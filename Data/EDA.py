import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read dataset
df = pd.read_csv("fulldata_rd7.csv")
print("Original shape:", df.shape)

# 2. Create images directory if it does not exist
os.makedirs("images", exist_ok=True)

# 3. Plot redshift distribution (before outlier removal)
if "redshift" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["redshift"], bins=50, kde=True, color='orange')
    plt.title("Redshift Distribution Across the Full Range (0 ≤ z ≤ 8)")
    plt.xlabel("Redshift (z)")
    plt.ylabel("Number of Objects")
    plt.xlim(-1, 8)
    plt.grid(True)
    plt.tight_layout()

    # Save figure as JPEG
    save_path = "images/redshift_dist.jpeg"
    plt.savefig(save_path, format="jpeg", dpi=300)
    plt.close()

    print(f"📸 Saved redshift distribution plot to: {save_path}")

    # 4. Remove outliers based on redshift range
    df = df[(df["redshift"] >= 0) & (df["redshift"] <= 8)]
    print(f"🧹 Data shape after removing redshift < 0 or > 8: {df.shape}")

else:
    print("\n❌ 'redshift' column not found in the dataset.")
