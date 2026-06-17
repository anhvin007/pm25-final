# --- FILE: src/verifier.py ---
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats

class ModelVV:
    def __init__(self, model_data, actual_data):
        self.model = model_data  # Dictionary chứa ma trận A, residuals, predictions
        self.actual = actual_data # Dữ liệu kiểm thử thực tế

    def verify_stability(self):
        """Kiểm tra tính ổn định của ma trận chuyển đổi A"""
        eigenvalues = np.linalg.eigvals(self.model['matrix_A'])
        is_stable = np.all(np.abs(eigenvalues) < 1.0)
        return is_stable, eigenvalues

    def verify_residual_whiteness(self):
        """Kiểm tra phần dư có phải nhiễu trắng (White Noise) không"""
        # Kiểm định Ljung-Box
        lb_test = acorr_ljungbox(self.model['test_residuals'], lags=[10])
        is_white_noise = lb_test['lb_pvalue'].iloc[0] > 0.05
        return is_white_noise, lb_test

    def validate_performance(self):
        """Tính toán các chỉ số lỗi dự báo"""
        pred = self.model['predictions']
        actual = self.actual
        
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        # Chỉ số Theil's U (So sánh với Naive forecast)
        naive_error = np.sqrt(mean_squared_error(actual[1:], actual[:-1]))
        theil_u = rmse / naive_error
        
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "Theil_U": theil_u}

# --- Chạy V&V cho 1 trường ---
def run_vv_report(school_idx, model_data, test_actuals):
    vv = ModelVV(model_data, test_actuals)
    
    stable, eig = vv.verify_stability()
    white, lb = vv.verify_residual_whiteness()
    metrics = vv.validate_performance()
    
    print(f"--- V&V Report cho Trường {school_idx} ---")
    print(f"Stability: {stable} (Eigs: {eig.round(2)})")
    print(f"Residuals are White Noise: {white}")
    print(f"Performance: {metrics}")
    return metrics