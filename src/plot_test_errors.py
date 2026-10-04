import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

results = pd.read_csv("results/test_results.csv")

benign_errors = results.loc[results["Label"] == "BENIGN", "Reconstruction_Error"]
ddos_errors = results.loc[results["Label"] == "DDoS", "Reconstruction_Error"]
portscan_errors = results.loc[results["Label"] == "PortScan", "Reconstruction_Error"]

threshold = 0.32509285
plot_limit = 0.6

plt.figure(figsize=(10, 6))
plt.hist(benign_errors[benign_errors <= plot_limit], bins=100, alpha=0.5, density=True, label="BENIGN")
plt.hist(ddos_errors[ddos_errors <= plot_limit], bins=100, alpha=0.5, density=True, label="DDoS")
plt.hist(portscan_errors[portscan_errors <= plot_limit], bins=100, alpha=0.5, density=True, label="PortScan")
plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.3f}")
plt.xlim(0, plot_limit)
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution by Traffic Type")
plt.legend()
plt.tight_layout()
plt.savefig("results/test_reconstruction_errors_by_label.png", dpi=300)
plt.show()

y_true = results["Label"] != "BENIGN"
y_pred = results["Predicted_Anomaly"]

cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
display.plot(values_format="d")
plt.title("Autoencoder Anomaly Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()