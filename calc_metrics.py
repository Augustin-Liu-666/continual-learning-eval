import pandas as pd
import numpy as np

def compute_cl_metrics(csv_path="emnist/reports/acc_log.csv"):
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        print("❌ acc_log.csv was not found. Check the configured path.")
        return

    # Fill missing values with the mean of each column.
    df = df.apply(lambda col: col.fillna(col.mean()), axis=0)

    # Save the completed matrix for downstream analysis.
    df.to_csv(csv_path)

    R = df.values
    n = R.shape[0]

    # Final Accuracy
    ACC = R[-1].mean()

    # Backward Transfer (BWT)
    bwt_scores = [R[-1, i] - R[i, i] for i in range(n - 1)]
    BWT = np.mean(bwt_scores)

    # Learning Accuracy (LA)
    LA = np.mean([R[i, i] for i in range(n)])

    # Forgetting
    forgetting_list = [R[-1, i] - R[i, i] for i in range(n)]
    avg_forgetting = np.mean(forgetting_list)

    print("\n📊 Continual Learning Metrics:")
    print(f"ACC (Final Accuracy):         {ACC:.4f}")
    print(f"BWT (Backward Transfer):      {BWT:.4f}")
    print(f"LA  (Learning Accuracy):      {LA:.4f}")
    print(f"Avg Forgetting:               {avg_forgetting:.4f}")

    print("\n📉 Forgetting per Task:")
    for i in range(n):
        print(f"  Task T{i}: {forgetting_list[i]:+.4f}")
