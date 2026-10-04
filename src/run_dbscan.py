import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

data = pd.read_csv("results/anomaly_latent_vectors.csv")

print("Loaded data:", data.shape)

X = data.drop(columns=["Label"]).to_numpy(dtype=np.float32)

print("Full latent data:", X.shape)

sample_size = min(10000, len(X))

rng = np.random.default_rng(42)
sample_indices = rng.choice(len(X), size=sample_size, replace=False)

X_sample = X[sample_indices]

print("DBSCAN sample:", X_sample.shape)

min_samples = 5

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(X_sample)

distances, _ = neighbors.kneighbors(X_sample)

fifth_neighbor_distances = distances[:, -1]

eps = np.percentile(fifth_neighbor_distances, 95)

print("min_samples:", min_samples)
print("Estimated eps:", eps)

print("\nRunning DBSCAN on the 10,000-sample...")

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(X_sample)

sample_data = data.iloc[sample_indices].copy()
sample_data["Cluster"] = cluster_labels

number_of_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
number_of_noise_points = np.sum(cluster_labels == -1)
noise_percentage = number_of_noise_points / len(cluster_labels) * 100

print("\nDBSCAN results:")
print("Number of clusters:", number_of_clusters)
print("Number of noise points:", number_of_noise_points)
print("Noise percentage:", noise_percentage)

print("\nCluster sizes:")
print(sample_data["Cluster"].value_counts().sort_index())

print("\nTrue labels by cluster:")
print(pd.crosstab(sample_data["Cluster"], sample_data["Label"]))

sample_data.to_csv("results/dbscan_results.csv", index=False)

print("\nDBSCAN results saved to results/dbscan_results.csv")