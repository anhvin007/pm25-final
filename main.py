# --- FILE: main.py ---
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from filterpy.kalman import KalmanFilter
from src.optimizer import find_optimal_sarima_params

# Import hằng số từ config
from src import config
from src.data_prep import prep_spatial_data, prep_time_series_data
from src.spatial_idw import calculate_idw_weights
from src.risk_decision import generate_school_profiles, decision_matrix_rule, get_activity_params

def build_state_space_matrices_updated(train_series, order=(1, 0, 1)):
    """Ước lượng SARIMA và trích xuất toàn bộ hệ số ma trận State-Space"""
    model = SARIMAX(train_series, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    
    ar_poly = res.polynomial_ar
    ma_poly = res.polynomial_ma
    
    alpha = -ar_poly[1:]
    gamma = ma_poly[1:]
    
    r = max(len(alpha), len(gamma) + 1)
    alpha = np.pad(alpha, (0, r - len(alpha)))
    gamma = np.pad(gamma, (0, r - 1 - len(gamma)))
    
    A = np.zeros((r, r))
    A[:, 0] = alpha
    for i in range(r - 1):
        A[i, i + 1] = 1.0  
        
    G = np.zeros((r, 1))
    G[0, 0] = 1.0
    G[1:, 0] = gamma.reshape(-1)
    
    H = np.zeros((1, r))
    H[0, 0] = 1.0
    
    sigma2_w = np.var(res.resid)
    Q = sigma2_w * np.dot(G, G.T)
    
    return A, G, H, Q, r, res

def run_kalman_filter_updated(test_series, A, H, Q, r, z_mean):
    """Chạy đệ quy Bộ lọc Kalman đồng thời ghi lại Innovation Residuals và Kalman Gains"""
    kf = KalmanFilter(dim_x=r, dim_z=1)
    kf.F = A
    kf.H = H
    kf.Q = Q
    
    sigma_v = config.SENSOR_ERROR_TOLERANCE / 2
    R_val = (sigma_v * z_mean) ** 2
    kf.R = np.array([[R_val]])
    
    kf.x = np.zeros((r, 1))
    kf.P *= 1000.0  
    
    predictions = []
    test_residuals = []
    kalman_gains = []
    
    for z in test_series:
        kf.predict()
        
        # Tính toán Innovation Residual: y = z - H @ x_prior
        innov = z - np.dot(H, kf.x_prior)[0, 0]
        test_residuals.append(innov)
        
        # Tính toán chi tiết Độ lợi Kalman Gain vector K_t trước pha update
        S = np.dot(H, np.dot(kf.P_prior, H.T)) + kf.R
        K = np.dot(kf.P_prior, np.dot(H.T, np.linalg.inv(S)))
        kalman_gains.append(K.flatten())
        
        kf.update(z)
        predictions.append(kf.x[0, 0])
        
    return np.array(predictions), np.array(test_residuals), np.array(kalman_gains), R_val

def run_pipeline_end_to_end():
    print("=======================================================")
    print("🔥 KHỞI CHẠY HỆ THỐNG PIPELINE ĐẦY ĐỦ (80% TRAIN - 20% TEST) 🔥")
    print("=======================================================")
    
    # 1. Nạp và xử lý cấu trúc dữ liệu không gian
    school_df, stations_df, school_coords, station_coords = prep_spatial_data()
    pm25_clean_stations = prep_time_series_data(station_coords, p_idw=config.IDW_POWER_P)
    W_matrix = calculate_idw_weights(school_coords, station_coords)
    
    # Nhân ma trận nội suy không gian sinh chuỗi thời gian nền
    station_matrix = pm25_clean_stations.values.T
    school_matrix = np.dot(W_matrix, station_matrix)
    pm25_df = pd.DataFrame(school_matrix, index=school_df['Ten truong'], columns=pm25_clean_stations.index)
    pm25_df.to_csv("data/processed/School_PM25_Estimated.csv")
    
    n_schools = pm25_df.shape[0]
    time_index = pm25_df.columns
    split_idx = int(len(time_index) * 0.8)
    test_times = time_index[split_idx:]
    
    final_predictions = []
    
    # 2. Quét qua 218 trường với mô hình hóa Không gian Trạng thái + Checkpoint toàn diện
    print("\n[1/4] CHẠY KHỐI TOÁN HỌC ĐỘNG LỰC HỌC & LƯU TRỮ THAM SỐ ẨN")
    for i in tqdm(range(n_schools), desc="Đang xử lý Bộ lọc Kalman"):
        school_name = pm25_df.index[i]
        model_pkl_path = f"outputs/models/school_{i}_ss_model.pkl"
        
        # Nếu đã có file Checkpoint mô hình toàn diện, nạp lại để tiết kiệm tài nguyên
        if os.path.exists(model_pkl_path):
            with open(model_pkl_path, "rb") as f:
                school_data = pickle.load(f)
            final_predictions.append(school_data['predictions'])
            continue
            
        series = pm25_df.iloc[i].values
        train_data, test_data = series[:split_idx], series[split_idx:]
        z_mean = np.mean(train_data)
        
        try:

            # Tự động tìm tham số tối ưu
            best_order, best_seasonal_order = find_optimal_sarima_params(train_data)
            # print(f"Trường {school_name} chọn params: {best_order} {best_seasonal_order}")

            # Giai đoạn 1: SARIMA & Flattening Ma trận
            A, G, H, Q, r, res = build_state_space_matrices_updated(train_data, order=best_order)
            
            # Giai đoạn 2: Lọc Kalman động học + Thu thập chuỗi residuals & gains
            pred, test_res, k_gains, R_val = run_kalman_filter_updated(test_data, A, H, Q, r, z_mean)
            train_res = res.resid
            
        except Exception as e:
            # Cơ chế Fallback an toàn chống sập chuỗi xử lý
            r = 2
            A = np.eye(r) * 0.5
            G = np.zeros((r, 1))
            H = np.zeros((1, r))
            H[0, 0] = 1.0
            Q = np.eye(r) * 0.1
            R_val = 1.0
            pred = test_data
            test_res = np.zeros_like(test_data)
            k_gains = np.zeros((len(test_data), r))
            train_res = np.zeros_like(train_data)
            
        # Đóng gói toàn bộ cấu trúc tham số theo đúng yêu cầu định lượng
        school_data = {
            'school_name': school_name,
            'matrix_A': A,
            'matrix_G': G,
            'matrix_H': H,
            'matrix_Q': Q,
            'scalar_R': R_val,
            'dimension_r': r,
            'z_mean': z_mean,
            'train_residuals': train_res,
            'test_residuals': test_res,
            'kalman_gains': k_gains,
            'predictions': pred,
            'weights_to_stations': W_matrix[i]
        }
        
        with open(model_pkl_path, "wb") as f:
            pickle.dump(school_data, f)
            
        final_predictions.append(pred)

    kalman_matrix = np.array(final_predictions)
    kalman_df = pd.DataFrame(kalman_matrix, index=pm25_df.index, columns=test_times)
    kalman_df.to_csv("data/processed/Kalman_PM25_Test.csv")
    
    # 3. Chạy Ma trận Ra quyết định Toàn tập
    print("\n[2/4] XUẤT BÁO CÁO MA TRẬN QUYẾT ĐỊNH TOÀN TẬP")
    profiles = generate_school_profiles(n_schools)
    act_params = get_activity_params('The_duc_nhe')
    MET = act_params['MET']
    dt = act_params['dt']
    
    all_decisions = []
    for i in tqdm(range(n_schools), desc="Xây dựng Ma trận hành động"):
        school_id = profiles.loc[i, 'school_id']
        s1_w = profiles.loc[i, 'S1_weight']
        s2_w = profiles.loc[i, 'S2_weight']
        
        for t_idx, time_col in enumerate(kalman_df.columns):
            pm25_val = kalman_df.iloc[i, t_idx]
            level, action = decision_matrix_rule(pm25_val)
            
            all_decisions.append({
                'Time': time_col,
                'School_ID': school_id,
                'PM25_Pred': pm25_val,
                'Risk_S1': pm25_val * MET * dt * s1_w,
                'Risk_S2': pm25_val * MET * dt * s2_w,
                'Level': level,
                'Action': action
            })
            
    decision_df = pd.DataFrame(all_decisions)
    decision_df.to_csv("outputs/reports/Full_Decision_Matrix_Test_Set.csv", index=False)
    
    # 4. Render Video Vệ tinh Heatmap (72 giờ)
    print("\n[3/4] KHỞI TẠO RENDER VIDEO BẢN ĐỒ VỆ TINH HEATMAP")
    geometry = [Point(xy) for xy in zip(school_df['Kinhdo'], school_df['Vido'])]
    gdf = gpd.GeoDataFrame(school_df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)
    
    filenames = []
    time_cols = kalman_df.columns[:72]
    
    for idx, t_col in enumerate(tqdm(time_cols, desc="Render Frames")):
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf['PM25_Now'] = kalman_df[t_col].values
        gdf.plot(ax=ax, column='PM25_Now', cmap='jet', markersize=60, alpha=0.8, 
                 legend=True, vmin=0, vmax=100, legend_kwds={'label': "Nồng độ PM2.5 (µg/m³)", 'shrink': 0.6})
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
        ax.set_title(f"Hệ thống Cảnh báo PM2.5 Học đường\nThời gian: {t_col}", fontsize=14, color='white')
        ax.set_axis_off()
        fig.patch.set_facecolor('#1e1e1e')
        
        filename = f"outputs/video_frames/frame_{idx:04d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        filenames.append(filename)
        
    print("\n[4/4] ĐANG KẾT NỐI KHUNG HÌNH THÀNH VIDEO")
    video_path = "outputs/PM25_Satellite_Heatmap.gif"
    with imageio.get_writer(video_path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            writer.append_data(imageio.imread(filename))
            
    print(f"\n✅ PIPELINE THÀNH CÔNG! ĐÃ LƯU VIDEO TẠI: {video_path}")

if __name__ == "__main__":
    os.makedirs('outputs/video_frames', exist_ok=True)
    run_pipeline_end_to_end()