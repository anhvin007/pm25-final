# --- FILE: src/risk_decision.py ---
import numpy as np
import pandas as pd
import os
from src import config

def generate_school_profiles(n_schools=config.NUM_SCHOOLS):
    """
    Sinh dữ liệu nhân khẩu học (tỷ lệ bệnh nền hô hấp) cho 218 trường THPT.
    """
    np.random.seed(2026)
    total_students = np.random.randint(1000, 2501, size=n_schools)
    
    # Tỷ lệ nhóm S2 (có bệnh nền) tuân theo phân phối chuẩn
    s2_ratio = np.maximum(np.random.normal(config.MU_SENSITIVE, config.SIGMA_SENSITIVE, size=n_schools), 0)
    
    profiles = pd.DataFrame({
        'school_id': range(1, n_schools + 1),
        'total_students': total_students,
        'ratio_S1': 1 - s2_ratio,
        'ratio_S2': s2_ratio,
        'S1_weight': 1.0,
        'S2_weight': config.PENALTY_S2
    })
    return profiles

def get_activity_params(activity_name):
    """
    Từ điển các hoạt động tiêu chuẩn (MET và Delta_t).
    Thời gian (Delta_t) được tính bằng giờ.
    """
    activities = {
        'Sinh_hoat_duoi_co': {'MET': 1.5, 'dt': 45 / 60},
        'The_duc_nhe': {'MET': 3.5, 'dt': 15 / 60},
        'The_duc_cuong_do_cao': {'MET': 7.5, 'dt': 30 / 60}
    }
    return activities.get(activity_name, {'MET': 1.0, 'dt': 1.0})

def decision_matrix_rule(pm25_val):
    """
    Ma trận ra quyết định dựa trên 4 phân lớp rủi ro cốt lõi.
    """
    if pm25_val < config.THRESHOLD_LEVEL_1:
        return 'MỨC 1 (XANH LỤC)', 'Không can thiệp. Hoạt động bình thường.'
    elif pm25_val < config.THRESHOLD_LEVEL_2:
        return 'MỨC 2 (VÀNG)', 'Cách ly cục bộ: Di tản nhóm học sinh nhạy cảm (S2) vào lớp nghỉ ngơi.'
    elif pm25_val < config.THRESHOLD_LEVEL_3:
        return 'MỨC 3 (CAM)', 'Giảm thời lượng: Cắt giảm tối đa 50% thời gian tiết học ngoài trời.'
    else:
        return 'MỨC 4 (ĐỎ)', 'Nguy hiểm: Hủy hoạt động ngoài trời hoặc di chuyển vào nhà đa năng.'

def execute_decision_module(forecast_matrix_1h, activity_name):
    """
    Quét qua toàn bộ 218 trường và xuất ra lệnh điều hành dựa trên mức PM2.5 dự báo
    tại 1 khung giờ cụ thể.
    """
    profiles = generate_school_profiles()
    act_params = get_activity_params(activity_name)
    MET = act_params['MET']
    dt = act_params['dt']
    
    results = []
    
    for i, row in profiles.iterrows():
        # Lấy giá trị PM2.5 dự báo đã lọc Kalman cho trường thứ i
        pm25_pred = forecast_matrix_1h[i]
        
        # 1. Đánh giá Rủi ro Y tế (Risk Index)
        risk_S1 = pm25_pred * MET * dt * row['S1_weight']
        risk_S2 = pm25_pred * MET * dt * row['S2_weight']
        
        # 2. Ma trận Ra Quyết Định
        level, action = decision_matrix_rule(pm25_pred)
        
        results.append({
            'Trường_ID': int(row['school_id']),
            'Dự_báo_PM25': round(pm25_pred, 2),
            'Risk_S1 (Bình thường)': round(risk_S1, 2),
            'Risk_S2 (Nhạy cảm)': round(risk_S2, 2),
            'Cấp_độ': level,
            'Lệnh_Hành_Động': action
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("--- RUNNING MODULE 3: RISK ASSESSMENT & DECISION MATRIX ---")
    
    # Giả lập: Chúng ta lấy một lát cắt 1 giờ cụ thể từ dữ liệu nội suy 
    # (Trong thực tế, đây sẽ là vector trạng thái đầu ra của lõi Kalman cho toàn bộ 218 trường)
    school_pm25_df = pd.read_csv("data/processed/School_PM25_Estimated.csv", index_col=0)
    
    # Lấy dữ liệu của 1 khung giờ bất kỳ (ví dụ: cột thứ 1000) để chạy kiểm thử
    forecast_1h = school_pm25_df.iloc[:, 1000].values
    
    # Giả định khung giờ này các trường đang có tiết "Thể dục cường độ cao"
    activity = 'The_duc_cuong_do_cao'
    print(f"\nKịch bản mô phỏng: Khung giờ có hoạt động '{activity}'")
    
    decision_df = execute_decision_module(forecast_1h, activity)
    
    # Hiển thị kết quả mẫu cho 5 trường đầu tiên và 5 trường cuối cùng
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n[Kết quả Điều hành Cục bộ - 5 Trường Đầu Tiên]")
    print(decision_df.head(5).to_string(index=False))
    
    print("\n[Kết quả Điều hành Cục bộ - 5 Trường Cuối Cùng]")
    print(decision_df.tail(5).to_string(index=False))
    
    # Thống kê tổng quan để báo cáo lên Ban giám đốc Sở
    print("\n[Thống kê Tổng quan Lệnh Điều Hành toàn thành phố]")
    print(decision_df['Cấp_độ'].value_counts().to_string())
    
    # Lưu báo cáo
    os.makedirs('outputs/reports', exist_ok=True)
    decision_df.to_csv("outputs/reports/Decision_Matrix_Snapshot.csv", index=False)
    print("\nĐã xuất báo cáo chi tiết ra file: outputs/reports/Decision_Matrix_Snapshot.csv")