import pandas as pd
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the features using float32 to reduce memory usage.
X = pd.read_csv("data/cleaned/features.csv").astype("float32")

# Load the labels.
y = pd.read_csv("data/cleaned/labels.csv")[" Label"]

print("Features loaded:", X.shape)
print("Labels loaded:", y.shape)


# Identify benign samples.
benign_mask = y == "BENIGN"

# Separate benign and attack features.
X_benign = X[benign_mask]
X_attack = X[~benign_mask]

# We no longer need the full feature DataFrame or the mask.
del X, benign_mask
gc.collect()


# Split the benign samples:
# 70% for Autoencoder training and 30% for testing.
X_train, X_benign_test = train_test_split(X_benign, test_size=0.30, random_state=42)

# X_benign is no longer needed.
del X_benign
gc.collect()


# Build the test set:
# 30% unseen benign samples + all attack samples.
X_test = pd.concat([X_benign_test, X_attack], ignore_index=True)

# X_benign_test and X_attack are now already contained in X_test.
del X_benign_test, X_attack
gc.collect()


# Build the corresponding labels.
# Training contains only benign samples.
y_train = pd.Series(["BENIGN"] * len(X_train), name=" Label")

# The first part of X_test contains benign samples.
# The remaining rows contain the attacks in their original order.
y_attack = y[y != "BENIGN"].reset_index(drop=True)
y_benign_test = pd.Series(["BENIGN"] * (len(X_test) - len(y_attack)), name=" Label")
y_test = pd.concat([y_benign_test, y_attack], ignore_index=True)

# The original label Series and temporary label objects are no longer needed.
del y, y_attack, y_benign_test
gc.collect()


# Check the train/test split.
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Test label distribution:")
print(y_test.value_counts())


# Create the StandardScaler.
scaler = StandardScaler()

# Learn the mean and standard deviation from the training data
# and standardize the training samples.
X_train_scaled = scaler.fit_transform(X_train).astype("float32")

# X_train is no longer needed.
del X_train
gc.collect()

# Convert the scaled training data back to a DataFrame.
X_train_scaled = pd.DataFrame(X_train_scaled, columns=pd.read_csv("data/cleaned/features.csv", nrows=0).columns)

print("Scaled training shape:", X_train_scaled.shape)

# Save the scaled training data immediately.
X_train_scaled.to_csv("data/cleaned/X_train_scaled.csv", index=False)

# Release the scaled training data before processing the test set.
del X_train_scaled
gc.collect()


# Standardize the test data using ONLY the statistics learned from training.
X_test_scaled = scaler.transform(X_test).astype("float32")

# X_test is no longer needed.
del X_test
gc.collect()

# Convert the scaled test data back to a DataFrame.
feature_columns = pd.read_csv("data/cleaned/features.csv", nrows=0).columns
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

print("Scaled test shape:", X_test_scaled.shape)

# Save the scaled test data.
X_test_scaled.to_csv("data/cleaned/X_test_scaled.csv", index=False)

# Release it after saving.
del X_test_scaled
gc.collect()


# Save the labels.
y_train.to_csv("data/cleaned/y_train.csv", index=False)
y_test.to_csv("data/cleaned/y_test.csv", index=False)

print("Train/test split and Z-score normalization completed.")