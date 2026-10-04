import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("results/dbscan_results.csv")

cluster_sizes = results["Cluster"].value_counts().sort_index()

cluster_names = ["Noise" if cluster == -1 else str(cluster) for cluster in cluster_sizes.index]

plt.figure(figsize=(14, 7))

plt.bar(cluster_names, cluster_sizes.values)

plt.xlabel("DBSCAN Cluster")
plt.ylabel("Number of Samples")
plt.title("DBSCAN Cluster Sizes")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("results/dbscan_cluster_sizes.png", dpi=300)

plt.show()

print("DBSCAN cluster-size graph saved to results/dbscan_cluster_sizes.png")

label_composition = pd.crosstab(results["Cluster"], results["Label"])

label_composition.index = ["Noise" if cluster == -1 else str(cluster) for cluster in label_composition.index]

label_composition.plot(kind="bar", stacked=True, figsize=(16, 8))

plt.xlabel("DBSCAN Cluster")
plt.ylabel("Number of Samples")
plt.title("True Label Composition of DBSCAN Clusters")

plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("results/dbscan_cluster_label_composition.png", dpi=300)

plt.show()

print("DBSCAN label-composition graph saved to results/dbscan_cluster_label_composition.png")