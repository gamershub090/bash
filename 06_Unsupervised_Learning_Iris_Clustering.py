"""
EXPERIMENT 6: Unsupervised Learning - Iris Clustering
K-Means + Agglomerative Clustering + validation + PCA.
Species is NOT used to train either clustering algorithm.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

iris = load_iris(as_frame=True)
X = iris.data
species = iris.target                 # Keep aside for evaluation only

print("Dataset shape:", X.shape)
print("Missing values:\n", X.isnull().sum())
print("Duplicates:", X.duplicated().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find a suitable K using internal metrics
print("\nK comparison")
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(k,
          "Inertia:", round(km.inertia_, 2),
          "Silhouette:", round(silhouette_score(X_scaled, labels), 3),
          "CH:", round(calinski_harabasz_score(X_scaled, labels), 2),
          "DB:", round(davies_bouldin_score(X_scaled, labels), 3))

# Final clustering (K=3 is used for demonstration after metric comparison)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)

hier = AgglomerativeClustering(n_clusters=3, linkage="ward")
hier_labels = hier.fit_predict(X_scaled)

print("\nK-Means cluster sizes:", np.bincount(km_labels))
print("K-Means silhouette:", silhouette_score(X_scaled, km_labels))
print("Hierarchical silhouette:", silhouette_score(X_scaled, hier_labels))

# PCA for 2D visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print("PCA explained variance:", pca.explained_variance_ratio_)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris clusters using K-Means")
plt.show()

# Species used only AFTER clustering for external evaluation
print("ARI:", adjusted_rand_score(species, km_labels))
print("NMI:", normalized_mutual_info_score(species, km_labels))
