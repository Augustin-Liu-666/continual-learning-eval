import pandas as pd
import numpy as np

# Read acc_log.csv.
csv_path = "emnist/reports/acc_log.csv"
df = pd.read_csv(csv_path, index_col=0)

# Replace NaN values with each column's mean.
df_filled = df.apply(lambda col: col.fillna(col.mean()), axis=0)

# Preview the completed data.
print("Data preview:")
print(df_filled.head())

# Save the completed matrix back to CSV.
df_filled.to_csv(csv_path)

# Compute metrics.
R = df_filled.values
n = R.shape[0]

# ===== Final Accuracy (ACC) =====
ACC = R[-1].mean()

# ===== Backward Transfer (BWT) =====
bwt_scores = [R[-1, i] - R[i, i] for i in range(n - 1)]
BWT = np.mean(bwt_scores)

# ===== Learning Accuracy (LA) =====
LA = np.mean([R[i, i] for i in range(n)])

# ===== Forgetting per Task =====
forgetting_list = [R[-1, i] - R[i, i] for i in range(n)]
avg_forgetting = np.mean(forgetting_list)

# Print results.
print("Continual Learning Metrics:")
print(f"ACC (Final Accuracy):         {ACC:.4f}")
print(f"BWT (Backward Transfer):      {BWT:.4f}")
print(f"LA  (Learning Accuracy):      {LA:.4f}")
print(f"Avg Forgetting:               {avg_forgetting:.4f}")

# Report forgetting separately for tasks T0-T4.
print("\n📉 Forgetting per Task:")
for i in range(n):
    print(f"  Task T{i}: {forgetting_list[i]:+.4f}")
