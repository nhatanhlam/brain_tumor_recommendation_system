import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Đường dẫn tuyệt đối đến file CSV và mô hình
base_dir = os.path.dirname(os.path.abspath(__file__))
info_path = os.path.join(base_dir, "data/hospital_info.csv")
predict_path = os.path.join(base_dir, "data/hospital_predict.csv")
model_path = os.path.join(base_dir, "models/hospital_recommender.keras")

# Kiểm tra sự tồn tại của file CSV và mô hình
if not os.path.exists(info_path):
    raise FileNotFoundError(f"⚠ Lỗi: Không tìm thấy file `{info_path}`. Hãy kiểm tra lại!")
if not os.path.exists(predict_path):
    raise FileNotFoundError(f"⚠ Lỗi: Không tìm thấy file `{predict_path}`. Hãy kiểm tra lại!")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"⚠ Lỗi: Không tìm thấy mô hình `{model_path}`. Hãy train mô hình trước!")

# Đọc dữ liệu từ các file CSV
hospital_info = pd.read_csv(info_path)
predict_df = pd.read_csv(predict_path)

# (Ở đây mô hình không được dùng để gợi ý vì ta dùng dữ liệu có sẵn)
@tf.keras.utils.register_keras_serializable()
class RecommenderModel(tf.keras.models.Model):
    def __init__(self, num_tumor_types, num_hospitals, embedding_dim=50):
        super(RecommenderModel, self).__init__()
        self.num_tumor_types = num_tumor_types
        self.num_hospitals = num_hospitals
        self.embedding_dim = embedding_dim

        self.tumor_embedding = tf.keras.layers.Embedding(num_tumor_types, embedding_dim)
        self.hospital_embedding = tf.keras.layers.Embedding(num_hospitals, embedding_dim)
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, inputs):
        tumor_vec = self.tumor_embedding(inputs[:, 0])
        hospital_vec = self.hospital_embedding(inputs[:, 1])
        return self.dot([tumor_vec, hospital_vec])

    def get_config(self):
        return {
            "num_tumor_types": self.num_tumor_types,
            "num_hospitals": self.num_hospitals,
            "embedding_dim": self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load mô hình (không sử dụng cho phần gợi ý dưới đây)
model = tf.keras.models.load_model(model_path, custom_objects={"RecommenderModel": RecommenderModel})

# Mapping cho các loại khối u của các nhóm điều trị (không bao gồm no_tumor)
tumor_mapping = {
    "glioma_tumor": 1,
    "meningioma_tumor": 2,
    "pituitary_tumor": 4
}

def recommend_hospitals(tumor_type, top_n=3):
    """
    Trả về danh sách gợi ý bệnh viện dựa trên tumor_id.
    Nếu tumor_type là "no_tumor", chỉ trả về các bệnh viện có tumor_id = 3.
    """
    # Nếu chọn "no_tumor", đặt target_tumor_id = 3
    if tumor_type == "no_tumor":
        target_tumor_id = 3
    elif tumor_type in tumor_mapping:
        target_tumor_id = tumor_mapping[tumor_type]
    else:
        print("⚠ Loại khối u không hợp lệ!")
        return None

    # Lọc các dòng trong predict_df theo tumor_id
    filtered = predict_df[predict_df["tumor_id"] == target_tumor_id].copy()
    filtered["hospital_rating"] = pd.to_numeric(filtered["hospital_rating"], errors="coerce")
    # Sắp xếp theo hospital_rating giảm dần
    filtered = filtered.sort_values(by="hospital_rating", ascending=False)
    
    # Merge với hospital_info theo hospital_id để lấy thông tin chi tiết
    merged = pd.merge(filtered, hospital_info, on="hospital_id", how="left")
    
    # Trả về top_n bệnh viện kèm thông tin cần hiển thị
    return merged.head(top_n)[["hospital_name", "hospital_address", "hospital_tel", "hospital_web", "hospital_rating"]]

# Nếu chạy file này trực tiếp, hiển thị các bệnh viện cho tumor_id = 3 (no_tumor)
if __name__ == "__main__":
    print("\n🔹 Gợi ý bệnh viện cho loại khối u: NO_TUMOR (tumor_id = 3)")
    recs = recommend_hospitals("no_tumor", top_n=10)
    if recs is not None:
        print(recs.to_string(index=False))
