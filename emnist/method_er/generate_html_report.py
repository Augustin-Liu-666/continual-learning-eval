import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configure report paths.
acc_path = "emnist/reports/acc_log.csv"
plot_path = "emnist/reports/acc_plot.png"
html_path = "emnist/reports/experiment_report.html"

# Read the accuracy matrix.
if not os.path.exists(acc_path):
    print("❌ acc_log.csv was not found. Train at least one task first.")
    exit()

df = pd.read_csv(acc_path, index_col=0)

# Keep tasks in T0-T4 order.
task_order = [f"T{i}" for i in range(5)]
df = df.loc[[i for i in task_order if i in df.index]]
R = df.values
n = R.shape[0]

# Plot the accuracy curves.
plt.figure(figsize=(8, 6))
for col in df.columns:
    plt.plot(df.index, df[col], marker='o', label=col)
plt.title("Accuracy vs Task (Method: ER)")
plt.xlabel("Training Task")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

# Compute continual-learning metrics.
ACC = R[-1].mean()
BWT = np.mean([R[-1, i] - R[i, i] for i in range(n - 1)])
FWT = np.mean([R[i - 1, i] for i in range(1, n)])
LA = np.mean([R[i, i] for i in range(n)])
forgetting_list = [R[-1, i] - R[i, i] for i in range(n)]
avg_forgetting = np.mean(forgetting_list)

# Build the HTML report.
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>EMNIST Continual Learning Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 60%; }}
        th, td {{ border: 1px solid #999; padding: 8px; text-align: center; }}
    </style>
</head>
<body>
    <h1>📋 EMNIST Continual Learning Report</h1>
    <h2>📊 Accuracy Curve</h2>
    <img src="acc_plot.png" width="600"><br><br>

    <h2>📈 Accuracy Table</h2>
    {df.to_html(classes='dataframe', border=1)}

    <h2>📌 Metrics</h2>
    <p><strong>ACC (Final Accuracy):</strong> {ACC:.4f}</p>
    <p><strong>BWT (Backward Transfer):</strong> {BWT:.4f}</p>
    <p><strong>FWT (Forward Transfer):</strong> {FWT:.4f}</p>
    <p><strong>LA  (Learning Accuracy):</strong> {LA:.4f}</p>
    <p><strong>Avg Forgetting:</strong> {avg_forgetting:.4f}</p>
</body>
</html>
"""

# Write the report.
with open(html_path, "w") as f:
    f.write(html)

print(f"HTML report generated: {html_path}")
