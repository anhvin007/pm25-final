# --- FILE: src/spatial_idw.py ---
import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from src import config
from src.data_prep import prep_spatial_data, prep_time_series_data

def calculate_idw_weights(school_coords, station_coords, p=config.IDW_POWER_P):
    """
    Tính toán ma trận trọng số IDW từ trạm quan trắc đến các trường THPT.
    """
    distances = cdist(school_coords, station_coords, metric='euclidean')
    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / (distances ** p)
    return weights / weights.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    print("--- RUNNING MODULE 1: HYBRID IMPUTATION & SPATIAL IDW ---")
    
    # 1. Nạp và xử lý cấu trúc tọa độ không gian
    school_df, stations_df, school_coords, station_coords = prep_spatial_data()
    
    # 2. Tiền xử lý chuỗi thời gian kết hợp nội suy chéo giữa các trạm
    print("Đang chạy thuật toán nội suy lai (Spline bậc 3 + IDW chéo giữa các trạm)...")
    pm25_clean_stations = prep_time_series_data(station_coords, p_idw=config.IDW_POWER_P)
    
    # 3. Tính toán ma trận trọng số không gian ánh xạ ra trường học
    W_matrix = calculate_idw_weights(school_coords, station_coords)
    
    # 4. Phép nhân ma trận tổng quát hóa không gian (218 trạm ảo)
    print("Đang ánh xạ dữ liệu sạch ra không gian 218 trường THPT...")
    station_matrix = pm25_clean_stations.values.T
    school_matrix = np.dot(W_matrix, station_matrix)
    
    # 5. Lưu trữ kết quả đầu ra dạng cấu trúc chuỗi thời gian sạch
    school_pm25_df = pd.DataFrame(school_matrix, 
                                  index=school_df['Ten truong'], 
                                  columns=pm25_clean_stations.index)
    
    os.makedirs('data/processed', exist_ok=True)
    school_pm25_df.to_csv("data/processed/School_PM25_Estimated.csv")
    print(f"Thành công! Đã xuất dữ liệu không gian trạng thái của {school_pm25_df.shape[0]} trường.")