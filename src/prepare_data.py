import pandas as pd
import numpy as np


ddos_df = pd.read_csv("data/cleaned/ddos_clean.csv")
portscan_df = pd.read_csv("data/cleaned/portscan_clean.csv")
webattacks_df = pd.read_csv("data/cleaned/webattacks_clean.csv")

print("DDoS shape:", ddos_df.shape)
print("PortScan shape:", portscan_df.shape)
print("WebAttacks shape:", webattacks_df.shape)

combined_df = pd.concat(
    [ddos_df, portscan_df, webattacks_df],
    ignore_index=True
)

print("Combined shape:", combined_df.shape)
print("Combined labels:")
print(combined_df[" Label"].value_counts())

numeric_combined_df = combined_df.select_dtypes(include="number")

print("NaN values:", combined_df.isna().sum().sum())
print("Infinite values:", np.isinf(numeric_combined_df).sum().sum())

combined_df.to_csv("data/cleaned/combined_clean.csv", index=False)

X = combined_df.drop(columns=[" Label"])
y = combined_df[" Label"]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Label distribution:")
print(y.value_counts())

X.to_csv("data/cleaned/features.csv", index=False)
y.to_csv("data/cleaned/labels.csv", index=False)

print("Prepared dataset saved successfully.")