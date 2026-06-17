# Mô hình Dự báo PM2.5 và Quản trị Rủi ro Y tế cho Trường học (HCMC)

Dự án này triển khai một hệ thống dự báo nồng độ bụi mịn ($PM_{2.5}$) dựa trên phương pháp **Không gian Trạng thái (State-Space Model)**, kết hợp giữa **SARIMA** và **Bộ lọc Kalman (Kalman Filter)**. Hệ thống được thiết kế để hỗ trợ Ban giám hiệu các trường THPT tại TP.HCM trong việc đưa ra quyết định hành chính (cách ly, giảm tiết học, hủy hoạt động) dựa trên dữ liệu rủi ro y tế thời gian thực.

## 🚀 Các Tính năng Nổi bật
- **Mô hình hóa Động lực học (Dynamic Modeling):** Áp dụng cấu trúc State-Space để lọc nhiễu đo lường từ cảm biến, cho phép dự báo tín hiệu sạch (filtered signal) với độ trễ thấp.
- **Nội suy Không gian (Spatial Interpolation):** Tự động hóa nội suy dữ liệu từ 6 trạm quan trắc về 218 điểm trường bằng phương pháp IDW (Inverse Distance Weighting).
- **Quản trị Rủi ro (Actuarial Risk Assessment):** Xây dựng ma trận quyết định dựa trên mô phỏng Monte Carlo về bệnh nền của học sinh ($S_1, S_2$) và cường độ vận động (MET).
- **Kiểm định Nghiêm ngặt (V&V):** Quy trình Verification & Validation tích hợp kiểm định tính ổn định của ma trận chuyển đổi ($A$) và kiểm định nhiễu trắng (White Noise) của phần dư.
- **Dashboard Trực quan:** Giao diện tương tác với Streamlit, hỗ trợ mô phỏng kiểm thử ứng suất (Stress Testing) và biểu đồ phân tích toán học.

## 🏗️ Kiến trúc Dự án
Dự án được cấu trúc theo mô hình Pipeline chặt chẽ:
1. **Data Prep:** Xử lý dữ liệu thô và nội suy không gian.
2. **Dynamics Engine:** Lõi Kalman Filter xử lý dữ liệu đệ quy (Recursive Estimation).
3. **Decision Matrix:** Hệ thống ra quyết định tự động dựa trên ngưỡng PM2.5.
4. **Monitoring:** Dashboard giám sát thời gian thực và báo cáo rủi ro.

## 🛠️ Cài đặt & Sử dụng

### 1. Yêu cầu hệ thống
- Python 3.12+
- Các thư viện cần thiết:
```bash
pip install -r requirements.txt

python main.py

streamlit run app.py