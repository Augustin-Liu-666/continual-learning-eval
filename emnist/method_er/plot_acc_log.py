import pandas as pd
import matplotlib.pyplot as plt
import os

# Configure input and output paths.
acc_path = "emnist/reports/acc_log.csv"
plot_path = "emnist/reports/acc_plot.png"

# Read the accuracy data.
if not os.path.exists(acc_path):
    print("❌ acc_log.csv was not found. Train at least one task first.")
    exit()

df = pd.read_csv(acc_path, index_col=0)

# Keep the task index in T0-T4 order.
task_order = [f"T{i}" for i in range(5)]
df = df.loc[[i for i in task_order if i in df.index]]

# Plot the accuracy curves.
plt.figure(figsize=(8, 6))
for col in df.columns:
    plt.plot(df.index, df[col], marker='o', label=col)

method_name = os.path.basename(acc_path).split("_")[0].upper()  # For example, acc_log.csv -> ACC.
plt.title(f"Accuracy vs Task (Method: {method_name})")
plt.xlabel("Training Task")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
print(f"Accuracy plot saved to: {plot_path}")
