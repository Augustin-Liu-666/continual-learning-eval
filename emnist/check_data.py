import pandas as pd
import os

# Configure the input path.
acc_path = "emnist/reports/acc_log.csv"

# Verify that the file exists.
if not os.path.exists(acc_path):
    print("❌ acc_log.csv was not found. Train at least one task first.")
    exit()

# Read the accuracy matrix.
df = pd.read_csv(acc_path, index_col=0)

# Preview the first rows for a quick sanity check.
print("Data preview:")
print(df.head())

# Report missing values.
print("\nMissing-value counts:")
print(df.isna().sum())
