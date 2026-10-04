import pandas as pd
import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(78, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 78))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder()
model.load_state_dict(torch.load("results/best_autoencoder.pt", weights_only=True))
model.eval()

results = pd.read_csv("results/test_results.csv")

anomaly_mask = results["Predicted_Anomaly"].astype(str).str.lower().eq("true").to_numpy()

print("Total test samples:", len(anomaly_mask))
print("Detected anomalies:", anomaly_mask.sum())

latent_vectors = []
anomaly_labels = []
start_index = 0

with torch.no_grad():
    for chunk in pd.read_csv("data/cleaned/X_test_scaled.csv", dtype=np.float32, chunksize=10000):
        end_index = start_index + len(chunk)
        chunk_mask = anomaly_mask[start_index:end_index]
        anomaly_chunk = chunk[chunk_mask]

        if len(anomaly_chunk) > 0:
            anomaly_tensor = torch.from_numpy(anomaly_chunk.to_numpy())
            encoded = model.encoder(anomaly_tensor)
            latent_vectors.append(encoded.numpy())

            chunk_labels = results["Label"].iloc[start_index:end_index].to_numpy()[chunk_mask]
            anomaly_labels.extend(chunk_labels)

        start_index = end_index

latent_vectors = np.vstack(latent_vectors)
anomaly_labels = np.array(anomaly_labels)

print("Latent representation shape:", latent_vectors.shape)
print("Anomaly labels:", len(anomaly_labels))

latent_df = pd.DataFrame(latent_vectors, columns=[f"latent_{i}" for i in range(16)])
latent_df["Label"] = anomaly_labels
latent_df.to_csv("results/anomaly_latent_vectors.csv", index=False)

print("Latent representations saved to results/anomaly_latent_vectors.csv")