import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------
# Step 1: Load Data
# ------------------------------
# Option 1: Read from CSV
# df = pd.read_csv('your_data.csv')

# Option 2: Use sample data
df = pd.DataFrame({
    'age': [25, 34, 28, 52, 46, 55, np.nan, 23, 40],
    'income': [50000, 64000, np.nan, 120000, 95000, np.nan, 45000, 32000, 88000],
    'gender': ['M', 'F', 'F', 'M', np.nan, 'F', 'M', 'M', 'F']
})

print("Initial data:\n", df.head())

# ------------------------------
# Step 2: Handle Missing Values
# ------------------------------
# Numeric → median; Categorical → mode
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ------------------------------
# Step 3: Encode Categorical Variables
# ------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# ------------------------------
# Step 4: Feature Scaling
# ------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_features, columns=df.columns)

# ------------------------------
# Step 5: EDA - Explore Data
# ------------------------------
print("\nBasic stats:\n", df.describe())
sns.pairplot(df)
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ------------------------------
# Step 6: Elbow Method to Find Optimal k
# ------------------------------
inertia = []
silhouette = []
K = range(2, 11)  # Try clusters from 2 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot Elbow Curve
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method (Inertia)')

# Plot Silhouette Scores
plt.subplot(1,2,2)
plt.plot(K, silhouette, 'gx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score per k')

plt.tight_layout()
plt.show()

# ------------------------------
# Step 7: Train Final Model
# ------------------------------
optimal_k = 3  # Assume we picked 3 from elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# ------------------------------
# Step 8: Analyze Cluster Centers
# ------------------------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers = pd.DataFrame(centers, columns=df.columns[:-1])
print("\nCluster centers (unscaled):\n", cluster_centers)

# ------------------------------
# Step 9: EDA - Visualize Clusters
# ------------------------------
# Pairplot with cluster hue
sns.pairplot(df, hue='cluster', palette='tab10')
plt.suptitle('Clusters Visualization', y=1.02)
plt.show()

# Cluster counts
sns.countplot(x='cluster', data=df)
plt.title('Cluster Counts')
plt.show()

# Cluster summary
print("\nCluster Summary:")
print(df.groupby('cluster').mean())

# ------------------------------
# Optional: Visualize on first 2 principal components
# ------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)
df['pca1'] = pca_components[:,0]
df['pca2'] = pca_components[:,1]

sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='tab10', s=100)
plt.title('Clusters Visualized in PCA Space')
plt.show()
