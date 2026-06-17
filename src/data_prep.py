# --- FILE: src/data_prep.py ---
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def prep_spatial_data(school_path="data/raw/School.csv", 
                      station_path="data/raw/Healthyair-Air-Pollution-Monitoring-Stations.csv"):
    """
    Đọc và chuẩn hóa dữ liệu không gian (tọa độ trường học và trạm quan trắc).
    """
    # 1. Làm sạch tọa độ trường học
    school_df = pd.read_csv(school_path)
    school_df['Vido'] = school_df['Vido'].str.replace(',', '.').astype(float)
    school_df['Kinhdo'] = school_df['Kinhdo'].str.replace(',', '.').astype(float)
    school_coords = school_df[['Vido', 'Kinhdo']].values
    
    # 2. Làm sạch tọa độ trạm quan trắc (Sửa lỗi ngược tiêu đề cột của file gốc)
    stations_df = pd.read_csv(station_path)
    station_coords = stations_df[['Longitude', 'Latitude']].values 
    
    return school_df, stations_df, school_coords, station_coords

def fill_missing_cross_idw(pm25_pivot, station_coords, p=2.0):
    """
    Thuật toán nội suy chéo IDW giữa các trạm để xử lý khuyết dữ liệu theo khối.
    Nếu một trạm mất tín hiệu, giá trị của nó sẽ được ước lượng từ các trạm lân cận đang chạy.
    """
    # Tính ma trận khoảng cách hình học giữa 6 trạm với nhau (kích thước 6x6)
    station_distances = cdist(station_coords, station_coords, metric='euclidean')
    station_distances = np.maximum(station_distances, 1e-10) # Ngăn chặn chia cho 0
    
    data_matrix = pm25_pivot.values.copy()
    T, N = data_matrix.shape
    
    # Duyệt qua từng bước thời gian t (từng giờ)
    for t in range(T):
        row = data_matrix[t]
        missing_mask = np.isnan(row)
        active_mask = ~missing_mask
        
        # Chỉ xử lý nếu có trạm khuyết VÀ có ít nhất một trạm xung quanh hoạt động
        if not np.any(missing_mask) or not np.any(active_mask):
            continue
            
        # Duyệt qua các trạm đang bị khuyết tại giờ t
        for idx_missing in np.where(missing_mask)[0]:
            # Lấy khoảng cách từ trạm khuyết này đến các trạm đang chạy
            dists = station_distances[idx_missing, active_mask]
            weights = 1.0 / (dists ** p)
            
            # Công thức IDW: tổng_trọng_số(khối_lượng * giá_trị) / tổng_trọng_số
            data_matrix[t, idx_missing] = np.sum(weights * row[active_mask]) / np.sum(weights)
            
    return pd.DataFrame(data_matrix, index=pm25_pivot.index, columns=pm25_pivot.columns)

def prep_time_series_data(station_coords, aqi_path="data/raw/Air-Quality-HCM.csv", p_idw=2.0):
    """
    Luồng tiền xử lý lai kết hợp bóc tách dữ liệu theo cấu trúc thời gian và không gian.
    """
    aqi_df = pd.read_csv(aqi_path)
    aqi_df['date'] = pd.to_datetime(aqi_df['date'], format='%d-%m-%Y %H:%M')
    
    # Tạo bảng xoay (Pivot Table) với index là trục thời gian liên tục
    pm25_pivot = aqi_df.pivot(index='date', columns='Station_No', values='PM2.5')
    full_range = pd.date_range(start=pm25_pivot.index.min(), end=pm25_pivot.index.max(), freq='h')
    pm25_pivot = pm25_pivot.reindex(full_range)
    
    # Chặng 1: Xử lý khuyết rải rác (Sporadic Missing) bằng Nội suy Spline bậc 3 (giới hạn liên tục <= 5 giờ)
    pm25_filled = pm25_pivot.interpolate(method='spline', order=3, limit=5)
    
    # Chặng 2: Xử lý khuyết theo khối (Block Missing) bằng thuật toán nội suy chéo IDW giữa các trạm
    pm25_filled = fill_missing_cross_idw(pm25_filled, station_coords, p=p_idw)
    
    # Chặng 3: Điền khuyết tối sau cùng (Trường hợp cực đoan tất cả các trạm sập nguồn cùng lúc)
    pm25_filled = pm25_filled.interpolate(method='linear').bfill().ffill()
    
    return pm25_filled