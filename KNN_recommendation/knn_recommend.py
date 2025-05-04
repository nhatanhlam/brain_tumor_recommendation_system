import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json

# Đường dẫn tuyệt đối đến file CSV (điều chỉnh nếu cần)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_PATH = os.path.join(BASE_DIR, "data/hospital_info.csv")
PREDICT_PATH = os.path.join(BASE_DIR, "data/hospital_predict.csv")

# Kiểm tra sự tồn tại của file CSV
if not os.path.exists(INFO_PATH):
    raise FileNotFoundError(f"⚠ Lỗi: Không tìm thấy file `{INFO_PATH}`. Hãy kiểm tra lại!")
if not os.path.exists(PREDICT_PATH):
    raise FileNotFoundError(f"⚠ Lỗi: Không tìm thấy file `{PREDICT_PATH}`. Hãy kiểm tra lại!")

# Đọc dữ liệu
hospital_info = pd.read_csv(INFO_PATH)
predict_df = pd.read_csv(PREDICT_PATH)

# Chuyển hospital_rating sang kiểu số
predict_df["hospital_rating"] = pd.to_numeric(predict_df["hospital_rating"], errors="coerce")
predict_df = predict_df.dropna(subset=["hospital_rating"])

# Ánh xạ tumor_id sang nhãn
tumor_map = {
    1: "glioma_tumor",
    2: "meningioma_tumor",
    3: "no_tumor",
    4: "pituitary_tumor"
}
predict_df["tumor_label"] = predict_df["tumor_id"].map(tumor_map)

# Tạo bảng pivot: hàng = hospital_id, cột = tumor_label, giá trị = hospital_rating
pivot_df = predict_df.pivot_table(index="hospital_id", columns="tumor_label", values="hospital_rating")
# Điền giá trị thiếu bằng giá trị trung bình của cột
pivot_df = pivot_df.apply(lambda col: col.fillna(col.mean()), axis=0)
pivot_df = pivot_df.sort_index()

def recommend_hospitals_knn(tumor_type, k=10):
    """
    Cho một loại tumor (ví dụ "glioma_tumor"), hệ thống tạo vector truy vấn:
      - Ở cột tương ứng với loại tumor đó: đặt giá trị 5 (yêu cầu tối đa).
      - Ở các cột khác: sử dụng giá trị trung bình của cột.
    Sau đó, dùng KNN để tìm k bệnh viện có vector rating gần nhất.
    Trả về DataFrame chứa thông tin của các bệnh viện (kết hợp từ hospital_info)
    cùng với hospital_rating được lấy trực tiếp từ pivot_df.
    
    Nếu tumor_type là "no_tumor", trả về toàn bộ bệnh viện với Note tham khảo.
    """
    if tumor_type not in tumor_map.values():
        print("⚠ Loại tumor không hợp lệ!")
        return None

    if tumor_type == "no_tumor":
        df = hospital_info.copy()
        df["Note"] = "Đây là các bệnh viện cho bạn tham khảo."
        return df

    query = pivot_df.mean(axis=0).to_dict()
    query[tumor_type] = 5.0
    query_vector = np.array([query[col] for col in pivot_df.columns]).reshape(1, -1)

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(pivot_df.values)
    distances, indices = nbrs.kneighbors(query_vector)

    recommended_ids = pivot_df.index[indices.flatten()]
    rec_df = hospital_info[hospital_info["hospital_id"].isin(recommended_ids)].copy()
    # Ở đây, ta vẫn merge rating gốc để hiển thị gợi ý
    rec_df = rec_df.merge(pivot_df[[tumor_type]].reset_index(), on="hospital_id", how="left")
    rec_df.rename(columns={tumor_type: "hospital_rating"}, inplace=True)
    rec_df = rec_df.sort_values(by="hospital_rating", ascending=False)
    return rec_df

def evaluate_knn(tumor_types=["glioma_tumor", "meningioma_tumor", "pituitary_tumor"], k=10):
    """
    Với mỗi loại tumor (ngoại trừ no_tumor), ta xây dựng vector truy vấn:
      - Đặt giá trị tối đa (5.0) cho cột tương ứng với loại tumor đó,
      - Dùng KNN với k láng giềng để lấy các giá trị rating.
    Sau đó, tính dự đoán rating là trung bình trọng số của các rating của k láng giềng,
    với weight = 1/(distance + epsilon).
    So sánh với giá trị lý tưởng là 5.0 và tính RMSE, MAE.
    Trả về một dictionary chứa RMSE, MAE và chi tiết lỗi cho từng tumor.
    """
    errors = {}
    for tumor in tumor_types:
        query = pivot_df.mean(axis=0).to_dict()
        query[tumor] = 5.0
        query_vector = np.array([query[col] for col in pivot_df.columns]).reshape(1, -1)
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nbrs.fit(pivot_df.values)
        distances, indices = nbrs.kneighbors(query_vector)
        neighbor_ratings = pivot_df[tumor].iloc[indices.flatten()].values
        weights = 1 / (distances.flatten() + 1e-6)
        predicted_rating = np.sum(weights * neighbor_ratings) / np.sum(weights)
        error = 5.0 - predicted_rating
        errors[tumor] = error
        print(f"{tumor}: Predicted = {predicted_rating:.4f}, Error = {error:.4f}")
    rmse = np.sqrt(np.mean(np.array(list(errors.values()))**2))
    mae = np.mean(np.abs(np.array(list(errors.values()))))
    knn_metrics = {
        "rmse": rmse,
        "mae": mae,
        "details": errors
    }
    return knn_metrics

if __name__ == "__main__":
    # In ra gợi ý cho từng loại tumor (ví dụ với k=10)
    for tumor in ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]:
        print(f"\n🔹 Gợi ý top 10 bệnh viện cho loại tumor: {tumor}")
        recs = recommend_hospitals_knn(tumor, k=10)
        if recs is not None:
            print(recs.to_string(index=False))
    
    # Đánh giá mô hình KNN cho các loại tumor (ngoại trừ no_tumor)
    knn_metrics = evaluate_knn()
    with open("knn_metrics.json", "w") as f:
        json.dump(knn_metrics, f, indent=4)
    print("KNN metrics đã được lưu vào knn_metrics.json")
