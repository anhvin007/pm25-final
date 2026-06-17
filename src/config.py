# --- FILE: src/config.py ---

# 1. Tham số Không gian - Thời gian
NUM_STATIONS = 6
NUM_SCHOOLS = 218
TIME_STEP = 1  # Đơn vị: giờ

# 2. Tham số Nội suy Không gian (IDW)
IDW_POWER_P = 2.0  # Lũy thừa bậc 2 (quy luật bình phương nghịch đảo)

# 3. Tham số Bộ lọc Kalman
SENSOR_ERROR_TOLERANCE = 0.20  # Dung sai cảm biến 10%

# 4. Tham số Đánh giá Rủi ro (Dịch tễ học)
MU_SENSITIVE = 0.06     # Kỳ vọng 6% học sinh có bệnh nền
SIGMA_SENSITIVE = 0.01  # Độ lệch chuẩn 1%
PENALTY_S2 = 1.5        # Hệ số phạt khuếch đại cho nhóm S2

# 5. Các Mức độ Rủi ro (Ngưỡng nồng độ PM2.5)
THRESHOLD_LEVEL_1 = 25
THRESHOLD_LEVEL_2 = 50
THRESHOLD_LEVEL_3 = 75