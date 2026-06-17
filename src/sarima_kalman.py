# --- FILE: src/sarima_kalman.py ---
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from filterpy.kalman import KalmanFilter
from src import config

def build_state_space_matrices(train_series, order=(2, 0, 1)):
    """
    Giai đoạn 1: Khai triển giải tích SARIMA và thiết lập Không gian Trạng thái.
    """
    # 1. Ước lượng hợp lý cực đại (MLE)
    model = SARIMAX(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    
    # 2. Khai triển đa thức (Flattening) thành ARMA mở rộng
    # Hàm polynomial trả về dạng [1, -a1, -a2...] -> Hệ số alpha* thực tế là đảo dấu
    ar_poly = res.polynomial_ar
    ma_poly = res.polynomial_ma
    
    alpha = -ar_poly[1:]
    gamma = ma_poly[1:]
    
    # Kích thước không gian r = max(p*, q* + 1)
    r = max(len(alpha), len(gamma) + 1)
    
    # Padding bằng số 0 nếu mảng bị hụt độ dài
    alpha = np.pad(alpha, (0, r - len(alpha)))
    gamma = np.pad(gamma, (0, r - 1 - len(gamma)))
    
    # 3. Biểu diễn Dạng chuẩn quan sát được (Companion form)
    A = np.zeros((r, r))
    A[:, 0] = alpha
    for i in range(r - 1):
        A[i, i + 1] = 1.0  # Thanh ghi dịch (Shift register)
        
    G = np.zeros((r, 1))
    G[0, 0] = 1.0
    G[1:, 0] = gamma.reshape(-1)
    
    H = np.zeros((1, r))
    H[0, 0] = 1.0
    
    # 4. Khởi tạo Ma trận hiệp phương sai nhiễu hệ thống (Q)
    # Dựa theo báo cáo: \sigma_w^2 = Var(e_t) với e_t là chuỗi sai số dư (residuals)
    sigma2_w = np.var(res.resid)
    Q = sigma2_w * np.dot(G, G.T)
    
    return A, H, Q, r

def run_kalman_filter(test_series, A, H, Q, r, z_mean):
    """
    Giai đoạn 2: Lọc nhiễu Kalman trên tập dữ liệu Out-of-sample.
    """
    # Khởi tạo thuật toán Kalman
    kf = KalmanFilter(dim_x=r, dim_z=1)
    kf.F = A
    kf.H = H
    kf.Q = Q
    
    # Ma trận nhiễu đo lường R (Phương sai với dung sai 10%)
    sigma_v = config.SENSOR_ERROR_TOLERANCE / 2
    R_val = (sigma_v * z_mean) ** 2
    kf.R = np.array([[R_val]])
    
    # Khởi tạo trạng thái ban đầu t=0
    kf.x = np.zeros((r, 1))
    kf.P *= 1000.0  # Ma trận P bất định tối đa ban đầu
    
    predictions = []
    
    # Vòng lặp đệ quy Predict-Update
    for z in test_series:
        kf.predict()        # Pha tiên nghiệm
        kf.update(z)        # Pha hậu nghiệm cập nhật tín hiệu z_t
        
        # Trích xuất phần tử x_1 (Nồng độ PM2.5 đã khử nhiễu)
        predictions.append(kf.x[0, 0])
        
    return np.array(predictions)

if __name__ == "__main__":
    print("--- RUNNING MODULE 2: SARIMA - KALMAN FILTER ---")
    
    # 1. Tải dữ liệu PM2.5 đã nội suy không gian
    school_pm25_df = pd.read_csv("data/processed/School_PM25_Estimated.csv", index_col=0)
    
    # Trích xuất chuỗi thời gian của Trường j=1 (THPT Bùi Thị Xuân)
    # Lấy 1000 giờ cuối cùng để mô phỏng cho nhẹ máy
    pm25_j1 = school_pm25_df.iloc[0, -1000:].values
    
    # 2. Chia tập dữ liệu: 80% Train, 20% Test
    split_idx = int(len(pm25_j1) * 0.8)
    train_data = pm25_j1[:split_idx]
    test_data = pm25_j1[split_idx:]
    
    print(f"Dữ liệu Huấn luyện: {len(train_data)} giờ, Kiểm thử: {len(test_data)} giờ")
    
    # 3. Chạy Giai đoạn 1: Xây dựng Không gian Trạng thái
    print("Đang ước lượng SARIMA và khởi tạo ma trận State-Space...")
    A, H, Q, r = build_state_space_matrices(train_data, order=(2, 0, 1))
    
    # 4. Chạy Giai đoạn 2: Lọc Kalman
    print("Đang chạy đệ quy Bộ lọc Kalman trên tập Kiểm thử...")
    z_mean = np.mean(train_data)
    predictions_j1 = run_kalman_filter(test_data, A, H, Q, r, z_mean)
    
   # 5. TRỰC QUAN HÓA KẾT QUẢ CHO TRƯỜNG j=1
    os.makedirs('outputs/figures', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    # Vẽ 100 giờ cuối cùng của tập test để dễ quan sát sự khác biệt
    plot_len = 100
    
    # CẬP NHẬT: Đường thực tế đổi sang màu xám (gray)
    plt.plot(test_data[-plot_len:], label='Tín hiệu Thực tế (Có nhiễu)', color='gray', alpha=0.6, marker='o', markersize=3)
    # Đường dự báo giữ màu xanh dương (blue) để nổi bật tín hiệu đã lọc
    plt.plot(predictions_j1[-plot_len:], label='Dự báo Hậu nghiệm (Kalman Lọc)', color='blue', linewidth=2)
    
    plt.title("Hiệu quả Khử Nhiễu bằng Bộ Lọc Kalman - THPT Bùi Thị Xuân (j=1)")
    plt.xlabel("Khung giờ (t)")
    plt.ylabel("Nồng độ PM2.5 (µg/m³)")
    
    # Tinh chỉnh chú giải (legend) và lưới (grid) cho chuyên nghiệp hơn
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = "outputs/figures/Kalman_Filter_School_j1.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ kết quả trực quan tại: {save_path}")