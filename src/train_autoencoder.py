import pandas as pd
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


X = pd.read_csv("data/cleaned/X_train_scaled.csv", dtype=np.float32)

print("Training data loaded:", X.shape)

X_tensor = torch.from_numpy(X.to_numpy())
dataset = TensorDataset(X_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 50
patience = 5
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        reconstructed = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch[0]
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "results/best_autoencoder.pt")
    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break