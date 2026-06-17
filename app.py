# --- FILE: app.py ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="Hệ thống Quản trị Rủi ro PM2.5 Học đường", layout="wide")

# Theme CSS
st.markdown("""
    <style>
    .main-title { font-size:2.4rem; color:#1E3A8A; font-weight:bold; margin-bottom:5px; }
    .sub-title { font-size:1.1rem; color:#4B5563; margin-bottom:20px; }
    .card { background-color:#F3F4F6; padding:15px; border-radius:10px; margin-bottom:15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌍 Hệ thống Giám sát PM2.5 & Quản trị Rủi ro Học đường (Bản Cao Cấp)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Lõi phân tích Không gian Trạng thái SARIMA-Kalman kết hợp 4 tính năng chẩn đoán toán học và kiểm thử rủi ro nâng cao.</div>', unsafe_allow_html=True)

@st.cache_data
def load_base_data():
    school_df = pd.read_csv("data/raw/School.csv")
    school_df['Vido'] = school_df['Vido'].str.replace(',', '.').astype(float)
    school_df['Kinhdo'] = school_df['Kinhdo'].str.replace(',', '.').astype(float)
    
    stations_df = pd.read_csv("data/raw/Healthyair-Air-Pollution-Monitoring-Stations.csv")
    
    decision_df = pd.read_csv("outputs/reports/Full_Decision_Matrix_Test_Set.csv")
    df_merged = pd.merge(decision_df, school_df, left_on='School_ID', right_on='STT')
    return df_merged, school_df, stations_df

# try:
#     df, schools, stations = load_base_data()
# except Exception as e:
#     st.error("Chưa nạp được dữ liệu. Vui lòng chạy lệnh `python main.py` trước để sinh file báo cáo pkl và csv.")
#     st.stop()
try:
    df, schools, stations = load_base_data()
except Exception as e:
    st.error(str(e))
    st.stop()

time_list = df['Time'].unique()
school_list = schools['Ten truong'].tolist()

# =======================================================
# TÍNH NĂNG MỚI SỐ 2: KỊCH BẢN MÔ PHỎNG KIỂM THỬ ỨNG SUẤT (STRESS TESTING)
# =======================================================
st.sidebar.header("⚙️ Kiểm Thử Ứng Suất (Stress Test)")
stress_multiplier = st.sidebar.slider("🚨 Cú sốc phát thải toàn cục (%)", min_value=100, max_value=300, value=100, step=10)
s2_penalty_slider = st.sidebar.slider("🦺 Hệ số phạt nhóm nhạy cảm (S2)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("🕒 Bộ chọn Không gian & Thời gian")
selected_time = st.sidebar.select_slider("Chọn mốc thời gian:", options=time_list)
selected_school = st.sidebar.selectbox("Chọn trường THPT cần chẩn đoán:", options=school_list)

school_id_selected = schools[schools['Ten truong'] == selected_school]['STT'].values[0]

# Đọc file model pkl chứa toàn bộ ma trận hệ trạng thái ẩn của trường được chọn
model_path = f"outputs/models/school_{school_id_selected - 1}_ss_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        ss_data = pickle.load(f)
else:
    st.error(f"Không tìm thấy file checkpoint mô hình cho trường {selected_school}")
    st.stop()

# Áp dụng Cú sốc phát thải toàn cục lên dữ liệu thời gian thực hiện tại (Stress Testing)
df_current_time = df[df['Time'] == selected_time].copy()
df_current_time['PM25_Pred'] = df_current_time['PM25_Pred'] * (stress_multiplier / 100.0)

# Định nghĩa lại hàm phân cấp động cho Stress Test
def dynamic_level_rule(val):
    if val < 25: return 'MỨC 1 (XANH LỤC)', 'Không can thiệp. Hoạt động bình thường.'
    elif val < 50: return 'MỨC 2 (VÀNG)', 'Cách ly cục bộ: Di tản nhóm học sinh nhạy cảm (S2) vào lớp nghỉ ngơi.'
    elif val < 75: return 'MỨC 3 (CAM)', 'Giảm thời lượng: Cắt giảm tối đa 50% thời gian tiết học ngoài trời.'
    else: return 'MỨC 4 (ĐỎ)', 'Nguy hiểm: Hủy hoạt động ngoài trời hoặc di chuyển vào nhà đa năng.'

# Cập nhật lại toàn bộ bảng trạng thái hiện tại dựa trên các Slider kiểm thử ứng suất
updated_levels = [dynamic_level_rule(v) for v in df_current_time['PM25_Pred']]
df_current_time['Level'] = [l[0] for l in updated_levels]
df_current_time['Action'] = [l[1] for l in updated_levels]

# Tính toán lại Risk Index động học
MET_demo = 3.5; dt_demo = 0.5 # Mặc định Thể dục nhẹ
df_current_time['Risk_S1'] = df_current_time['PM25_Pred'] * MET_demo * dt_demo * 1.0
df_current_time['Risk_S2'] = df_current_time['PM25_Pred'] * MET_demo * dt_demo * s2_penalty_slider

current_school_status = df_current_time[df_current_time['School_ID'] == school_id_selected].iloc[0]

# CHIA BỐ CỤC THỂ HIỆN BẢN ĐỒ VÀ METRICS TRỰC QUAN
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader(f"📍 Bản đồ Tương tác & Khuyến nghị Hành động Toàn Thành Phố ({selected_time})")
    color_discrete_map = {
        'MỨC 1 (XANH LỤC)': '#00CC96', 'MỨC 2 (VÀNG)': '#FECB52',
        'MỨC 3 (CAM)': '#FFA15A', 'MỨC 4 (ĐỎ)': '#EF553B'
    }
    
    # =======================================================
    # TÍNH NĂNG MỚI SỐ 3: PHÂN TÍCH TƯƠNG TÁC KHÔNG GIAN (SPATIAL NETWORK GRAPH)
    # =======================================================
    # Vẽ bản đồ kết hợp mạng lưới kết nối giữa trường học được chọn và 6 trạm đo vật lý
    fig_map = go.Figure()
    
    # Trace 1: Toàn bộ 218 trường học
    fig_map.add_trace(go.Scattermapbox(
        lat=df_current_time['Vido'], lon=df_current_time['Kinhdo'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=df_current_time['PM25_Pred']*0.4 + 5,
            color=[color_discrete_map[l] for l in df_current_time['Level']],
            opacity=0.7
        ),
        text=df_current_time['Ten truong'],
        hoverinfo='text',
        name='Trường học THPT'
    ))
    
    # Trace 2: Đường nối mạng lưới Euclid nối từ trường đang chọn đến 6 trạm đo vật lý
    sch_lat = current_school_status['Vido']
    sch_lon = current_school_status['Kinhdo']
    weights = ss_data['weights_to_stations']
    
    for idx, st_row in stations.iterrows():
        st_lat = st_row['Longitude'] # file gốc bị ngược tiêu đề cột
        st_lon = st_row['Latitude']
        w_val = weights[idx]
        
        fig_map.add_trace(go.Scattermapbox(
            lat=[sch_lat, st_lat], lon=[sch_lon, st_lon],
            mode='lines',
            line=go.scattermapbox.Line(width=w_val * 8, color='blue'),
            opacity=0.5,
            showlegend=False
        ))
        
    # Trace 3: Vị trí 6 trạm quan trắc thực tế làm điểm neo mạng lưới
    fig_map.add_trace(go.Scattermapbox(
        lat=stations['Longitude'], lon=stations['Latitude'],
        mode='markers+text',
        marker=go.scattermapbox.Marker(size=14, color='black', symbol='airport'),
        text=[f"Trạm {s}" for s in stations['Station']],
        textposition='top center',
        name='Trạm Quan trắc Gốc'
    ))
    
    fig_map.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lat=sch_lat, lon=sch_lon), zoom=11),
        margin={"r":0,"t":0,"l":0,"b":0}, height=550, showlegend=True
    )
    st.plotly_chart(fig_map, use_container_width=True)

with c2:
    st.subheader(f"📋 Giám sát Ứng suất: {selected_school}")
    st.metric(label="Nồng độ PM2.5 Đã kích hoạt Sốc", value=f"{current_school_status['PM25_Pred']:.2f} µg/m³", 
              delta=f"+{((stress_multiplier-100))}% so với nền" if stress_multiplier > 100 else "Trạng thái nền")
    
    st.markdown(f"""
    <div class="card">
    <b>Chỉ số Rủi ro Y tế Mô phỏng:</b><br>
    🟢 Nhóm thể trạng bình thường: <b>{current_school_status['Risk_S1']:.2f}</b><br>
    🟠 Nhóm bệnh lý nền (Hệ số phạt {s2_penalty_slider:.1f}): <b>{current_school_status['Risk_S2']:.2f}</b>
    </div>
    """, unsafe_allow_html=True)
    
    lvl = current_school_status['Level']
    act = current_school_status['Action']
    if "XANH" in lvl: st.success(f"**{lvl}**\n\n{act}")
    elif "VÀNG" in lvl: st.warning(f"**{lvl}**\n\n{act}")
    else: st.error(f"🚨 **{lvl}**\n\n{act}")
    
    # Biểu đồ thanh ngang thể hiện tỷ lệ đóng góp không gian của 6 trạm đo vào trường này (Feature 3)
    fig_contrib = px.bar(
        x=[f"Trạm {i+1}" for i in range(6)], y=weights * 100,
        labels={'x': 'Hạ tầng mạng lưới', 'y': 'Tỷ lệ đóng góp (%)'},
        title="Trọng số Không gian IDW thiết lập bụi nền j",
        height=220
    )
    fig_contrib.update_layout(margin={"r":10,"t":30,"l":10,"b":10})
    st.plotly_chart(fig_contrib, use_container_width=True)

# =======================================================
# TÍNH NĂNG MỚI SỐ 1: CHẨN ĐOÁN TOÁN HỌC & TÍNH ỔN ĐỊNH HỆ THỐNG
# =======================================================
st.markdown("---")
st.subheader("📊 Tính năng 1: Khối chẩn đoán Toán học và Tính ổn định Hệ thống (Companion Form Analysis)")
diag_col1, diag_col2, diag_col3 = st.columns([1, 1, 1])

with diag_col1:
    st.markdown("**Vòng tròn Đơn vị Trị riêng (Eigenvalue Stability Proof):**")
    # Tính toán các trị riêng phức của ma trận chuyển đổi A để chứng minh điều kiện dừng (Stationarity)
    A_mat = ss_data['matrix_A']
    eigenvalues = np.linalg.eigvals(A_mat)
    
    fig_circle = go.Figure()
    # Vẽ vòng tròn đơn vị
    th = np.linspace(0, 2*np.pi, 100)
    fig_circle.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode='lines', line=dict(dash='dash', color='gray'), name='Unit Circle'))
    # Vẽ các trị riêng phức
    fig_circle.add_trace(go.Scatter(x=np.real(eigenvalues), y=np.imag(eigenvalues), mode='markers', 
                                    marker=dict(size=12, color='red', symbol='x'), name='Trị riêng (λ)'))
    fig_circle.update_layout(width=350, height=350, xaxis=dict(range=[-1.5, 1.5]), yaxis=dict(range=[-1.5, 1.5]),
                             margin={"r":10,"t":10,"l":10,"b":10}, showlegend=False)
    st.plotly_chart(fig_circle, use_container_width=True)
    st.caption("Hệ thống ổn định vững chắc và dừng khi và chỉ khi toàn bộ các trị riêng ẩn λ nằm nghiêm ngặt bên trong vòng tròn đơn vị.")

with diag_col2:
    st.markdown("**Phân phối Sai số dư Khối kiểm thử (Test Innovations):**")
    # Biểu đồ tần suất Histogram kiểm tra xem chuỗi Innovation có tiệm cận Nhiễu trắng Gauss hay không
    test_res = ss_data['test_residuals']
    fig_hist = px.histogram(test_res, nbins=30, labels={'value': 'Sai số dư (µg/m³)'}, title=None)
    fig_hist.update_layout(width=350, height=330, margin={"r":10,"t":10,"l":10,"b":10}, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("Bộ lọc đạt trạng thái tối ưu Kalman tối đa khi sai số dư hội tụ về phân phối chuẩn có kỳ vọng bằng 0.")

with diag_col3:
    st.markdown("**Bảng thông số cấu trúc Không gian Trạng thái vĩ mô:**")
    st.markdown(fr"""
    - **Kích thước Không gian Trạng thái ẩn (r):** `{ss_data['dimension_r']}` chiều
    - **Phương sai nhiễu đo lường thiết bị ($R$):** `{ss_data['scalar_R']:.6f}`
    - **Nồng độ chuỗi trung bình lịch sử ($\overline{{z}}$):** {ss_data['z_mean']:.2f} µg/m³
    - **Độ lệch chuẩn sai số đo lường ($\sigma_v$):** `5.00%`
    """)
    st.write("**Ma trận chuyển đổi động lực thực thể $A$:**")
    st.dataframe(pd.DataFrame(A_mat).round(4), height=140)

# =======================================================
# TÍNH NĂNG MỚI SỐ 4: THEO DÕI ĐỘNG LỰC HỌC ĐỘ LỢI KALMAN GAIN (TIME-SERIES DYNAMICS)
# =======================================================
st.markdown("---")
st.subheader("📈 Tính năng 4: Phân tích Động lực học Kalman Gain & Chuỗi Thời gian thực")

df_school_all_time = df[df['School_ID'] == school_id_selected].copy()
# Đọc mảng Kalman Gains (trích xuất phần tử lọc trạng thái sạch x_1 đầu tiên)
k_gains_timeline = ss_data['kalman_gains'][:, 0]

# Tạo bảng dữ liệu đồ thị tổng hợp
time_len = len(df_school_all_time)
kg_plot_df = pd.DataFrame({
    'Time': df_school_all_time['Time'].values,
    'PM25_Forecast': df_school_all_time['PM25_Pred'].values,
    'Kalman_Gain_K1': k_gains_timeline[:time_len]
})

fig_kg = go.Figure()
fig_kg.add_trace(go.Scatter(x=kg_plot_df['Time'], y=kg_plot_df['Kalman_Gain_K1'],
                    mode='lines', name='Độ lợi Kalman Gain ($K_{1}$)', line=dict(color='purple', width=2)))
fig_kg.add_vline(x=selected_time, line_dash="dash", line_color="red", annotation_text="Giờ đang chọn")
fig_kg.update_layout(title="Biến thiên Hệ số Độ lợi Kalman Gain ($K_{t,1}$) qua chuỗi thời gian",
                     xaxis_title="Trục thời gian", yaxis_title="Trọng số niềm tin Gain", height=280,
                     margin={"r":10,"t":40,"l":10,"b":10})
st.plotly_chart(fig_kg, use_container_width=True)
st.caption("Hệ số Gain biến thiên động: Khi cảm biến ngoài hiện trường bị chấn động dữ dội, bộ lọc tự động hạ thấp Gain để tin vào lý thuyết phương trình; khi dữ liệu mượt mà, Gain tăng để bám sát thực địa.")

# Biểu đồ chuỗi thời gian nồng độ bụi mịn dự báo tổng quát
fig_line = px.line(df_school_all_time, x='Time', y='PM25_Pred', 
                   title=f"Quỹ đạo chuỗi nồng độ bụi mịn PM2.5 dự báo tại {selected_school}",
                   labels={'Time': 'Mốc thời gian', 'PM25_Pred': 'Nồng độ (µg/m³)'}, height=280)
fig_line.add_vline(x=selected_time, line_dash="dash", line_color="red")
st.plotly_chart(fig_line, use_container_width=True)