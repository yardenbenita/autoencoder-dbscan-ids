import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

X = pd.read_csv("data/cleaned/X_train_scaled.csv", dtype=np.float32)
X_tensor = torch.from_numpy(X.to_numpy())
dataset = TensorDataset(X_tensor)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
reconstruction_errors = []
with torch.no_grad():
    for batch in val_loader:
        x = batch[0]
        reconstructed = model(x)
        errors = torch.mean((x - reconstructed) ** 2, dim=1)
        reconstruction_errors.extend(errors.numpy())

reconstruction_errors = np.array(reconstruction_errors)
threshold = np.percentile(reconstruction_errors, 99)

print("Number of validation samples:", len(reconstruction_errors))
print("Mean reconstruction error:", reconstruction_errors.mean())
print("Median reconstruction error:", np.median(reconstruction_errors))
print("99th percentile threshold:", threshold)

print("Maximum reconstruction error:", reconstruction_errors.max())
print("95th percentile:", np.percentile(reconstruction_errors, 95))
print("99th percentile:", np.percentile(reconstruction_errors, 99))
print("99.9th percentile:", np.percentile(reconstruction_errors, 99.9))

plot_limit = np.percentile(reconstruction_errors, 99.5)

plt.figure(figsize=(10, 6))
plt.hist(reconstruction_errors[reconstruction_errors <= plot_limit], bins=100)
plt.axvline(threshold, linestyle="--", linewidth=2, label=f"99th percentile threshold = {threshold:.4f}")
plt.xlim(0, plot_limit)
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Number of Samples")
plt.title("Distribution of Reconstruction Errors - BENIGN Validation Data")
plt.legend()
plt.tight_layout()
plt.savefig("results/validation_reconstruction_errors.png", dpi=300)
plt.show()

test_errors = []

with torch.no_grad():
    for chunk in pd.read_csv("data/cleaned/X_test_scaled.csv", dtype=np.float32, chunksize=10000):
        x_test = torch.from_numpy(chunk.to_numpy())
        reconstructed = model(x_test)
        errors = torch.mean((x_test - reconstructed) ** 2, dim=1)
        test_errors.extend(errors.numpy())

test_errors = np.array(test_errors)

print("Number of test reconstruction errors:", len(test_errors))

test_predictions = test_errors > threshold

y_test = pd.read_csv("data/cleaned/y_test.csv")[" Label"]
y_true = y_test != "BENIGN"

print("Number of predictions:", len(test_predictions))
print("Number of true labels:", len(y_true))
print("Predicted anomalies:", test_predictions.sum())

tn, fp, fn, tp = confusion_matrix(y_true, test_predictions).ravel()

accuracy = accuracy_score(y_true, test_predictions)
precision = precision_score(y_true, test_predictions)
recall = recall_score(y_true, test_predictions)
specificity = tn / (tn + fp)
f1 = f1_score(y_true, test_predictions)

print("\nConfusion Matrix:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nMetrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1-score:", f1)

results = pd.DataFrame({"Label": y_test, "Reconstruction_Error": test_errors, "Predicted_Anomaly": test_predictions})

print("\nDetection rate by label:")
print(results.groupby("Label")["Predicted_Anomaly"].agg(["count", "mean"]))

error_summary = results.groupby("Label")["Reconstruction_Error"].agg(["count", "mean", "median", "min", "max"])
print("\nReconstruction error by label:")
print(error_summary)
results.to_csv("results/test_results.csv", index=False)