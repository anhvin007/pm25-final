# --- FILE: run_vv.py ---
import pickle, pandas as pd
from src.verifier import run_vv_report

all_vv_results = []
kalman_df = pd.read_csv("data/processed/Kalman_PM25_Test.csv", index_col=0)

for i in range(218):
    with open(f"outputs/models/school_{i}_ss_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    actuals = kalman_df.iloc[i].values
    metrics = run_vv_report(i, model_data, actuals)
    metrics['School_ID'] = i
    all_vv_results.append(metrics)

# Xuất file báo cáo V&V tổng thể
vv_df = pd.DataFrame(all_vv_results)
vv_df.to_csv("outputs/reports/VV_Summary.csv", index=False)
print("Đã xuất báo cáo kiểm định toàn bộ mô hình!")