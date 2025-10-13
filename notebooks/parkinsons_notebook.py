# ======================================
# STEP 2: Exploratory Data Analysis (EDA)
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load dataset
df = pd.read_csv("/data/parkinsons.data")
print("✅ Dataset loaded successfully!\n")

# 2️⃣ Basic info
print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# 3️⃣ Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# 4️⃣ Summary statistics
print("\nStatistical summary:\n", df.describe())

# 5️⃣ Class balance
print("\nTarget variable distribution:")
print(df['status'].value_counts())  # 1 = Parkinson’s, 0 = Healthy

# 6️⃣ Visualize a few relationships
sns.countplot(x='status', data=df)
plt.title("Distribution of Parkinson’s vs Healthy")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 7️⃣ Preprocessing: Feature scaling
# Drop non-numeric / unnecessary columns (e.g., name)
X = df.drop(['name', 'status'], axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Features scaled successfully!")
print("Scaled feature shape:", X_scaled.shape)
