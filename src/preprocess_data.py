import pandas as pd
import numpy as np


def preprocess_file(file_path, output_path):
    df = pd.read_csv(file_path)

    print("Processing:", file_path)
    print("Rows before cleaning:", len(df))

    numeric_df = df.select_dtypes(include="number")

    rows_with_inf = np.isinf(numeric_df).any(axis=1)
    rows_with_nan = df.isna().any(axis=1)
    invalid_rows = rows_with_inf | rows_with_nan

    print("Invalid rows found:", invalid_rows.sum())

    df_clean = df[~invalid_rows].copy()

    print("Rows after cleaning:", len(df_clean))

    print("NaN values after cleaning:")
    print(df_clean.isna().sum().sum())

    numeric_clean_df = df_clean.select_dtypes(include="number")
    inf_values_after_cleaning = np.isinf(numeric_clean_df).sum().sum()
    rows_with_inf_after_cleaning = np.isinf(numeric_clean_df).any(axis=1)

    print("Infinite values after cleaning:")
    print(inf_values_after_cleaning)

    print("Rows containing infinite values after cleaning:")
    print(rows_with_inf_after_cleaning.sum())

    df_clean.to_csv(output_path, index=False)

    print("Cleaned file saved to:", output_path)


preprocess_file("data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "data/cleaned/ddos_clean.csv")
preprocess_file("data/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "data/cleaned/portscan_clean.csv")
preprocess_file("data/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "data/cleaned/webattacks_clean.csv")