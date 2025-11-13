import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/data/parkinsons.data")
print(" Dataset loaded successfully!\n")

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

print("\nMissing values per column:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe())
print("\nTarget variable distribution:")
print(df['status'].value_counts())  # 1 = Parkinson’s, 0 = Healthy

sns.countplot(x='status', data=df)
plt.title("Distribution of Parkinson’s vs Healthy")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop(['name', 'status'], axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n Features scaled successfully!")
print("Scaled feature shape:", X_scaled.shape)
