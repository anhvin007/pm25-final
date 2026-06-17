# --- FILE: src/optimizer.py ---
import pmdarima as pm
import pandas as pd

def find_optimal_sarima_params(series):
    """
    Tự động tìm bộ tham số tối ưu bằng thuật toán stepwise AIC.
    s=24: tính mùa vụ theo giờ.
    """
    # Dùng auto_arima để tìm params
    model = pm.auto_arima(
    series,

    seasonal=True,
    m=24,

    max_p=15,
    max_d=3,
    max_q=15,

    max_P=8,
    max_D=2,
    max_Q=8,

    stepwise=False,
    n_jobs=-1,

    suppress_warnings=True,
    error_action="ignore",
    trace=True
)
    # model.order là (p, d, q)
    # model.seasonal_order là (P, D, Q, s)
    return model.order, model.seasonal_order