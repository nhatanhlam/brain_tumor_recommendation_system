import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os

# Alias cho các lớp và hàm của Keras
Model = keras.Model
Embedding = keras.layers.Embedding
Dot = keras.layers.Dot
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint

# Đảm bảo thư mục models tồn tại
os.makedirs("models", exist_ok=True)

# --- Bước 1: Đọc dữ liệu từ file CSV đã cập nhật ---
# File hospital_info.csv đã chứa hospital_id được sắp xếp theo thứ tự mong muốn.
hospital_info = pd.read_csv("data/hospital_info.csv", index_col=None)

# File hospital_predict.csv đã được lọc và cập nhật hospital_id tương ứng.
predict_df = pd.read_csv("data/hospital_predict.csv", index_col=None)

# --- Bước 2: Xử lý cột hospital_rating ---
# Chuyển đổi hospital_rating sang kiểu số, nếu không chuyển đổi được (ví dụ "KDG") sẽ trả về NaN
predict_df["hospital_rating"] = pd.to_numeric(predict_df["hospital_rating"], errors="coerce")
# Loại bỏ các dòng có giá trị NaN trong hospital_rating
predict_df = predict_df.dropna(subset=["hospital_rating"])

# --- Bước 3: Chuyển đổi tumor_id sang nhãn 0-indexed ---
# Trong predict_df, tumor_id ban đầu: 1 (Glioma), 2 (Meningioma), 3 (No_tumor), 4 (Pituitary)
# Ta chuyển sang nhãn: 0, 1, 2, 3.
predict_df["tumor_type"] = predict_df["tumor_id"] - 1

# --- Bước 4: Tách dữ liệu huấn luyện và kiểm tra ---
train, test = train_test_split(predict_df, test_size=0.2, random_state=42)

# --- Bước 5: Xác định số loại khối u và số bệnh viện ---
num_tumor_types = 4
num_hospitals = len(hospital_info)

# --- Bước 6: Xây dựng mô hình RecommenderModel ---
@tf.keras.utils.register_keras_serializable()
class RecommenderModel(Model):
    def __init__(self, num_tumor_types, num_hospitals, embedding_dim=50):
        super(RecommenderModel, self).__init__()
        self.num_tumor_types = num_tumor_types
        self.num_hospitals = num_hospitals
        self.embedding_dim = embedding_dim

        self.tumor_embedding = Embedding(num_tumor_types, embedding_dim)
        self.hospital_embedding = Embedding(num_hospitals, embedding_dim)
        self.dot = Dot(axes=1)

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

# Khởi tạo và biên dịch mô hình
model = RecommenderModel(num_tumor_types, num_hospitals, embedding_dim=50)
model.compile(optimizer="adam", loss="mse")

# --- Bước 7: Chuẩn bị dữ liệu cho mô hình ---
# Vì hospital_id trong CSV là 1-indexed, ta chuyển sang 0-indexed cho embedding.
train_data = train.copy()
train_data["hospital_id"] = train_data["hospital_id"] - 1
train_x = train_data[["tumor_type", "hospital_id"]].values
train_y = train_data["hospital_rating"].values

test_data = test.copy()
test_data["hospital_id"] = test_data["hospital_id"] - 1
test_x = test_data[["tumor_type", "hospital_id"]].values
test_y = test_data["hospital_rating"].values

# --- Bước 8: Định nghĩa callbacks ---
checkpoint_path = "models/last_checkpoint.keras"
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", mode="min")
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# --- Bước 9: Huấn luyện mô hình ---
try:
    model.fit(
        train_x, train_y,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    final_model_path = "models/hospital_recommender.keras"
    model.save(final_model_path)
    print(f"✅ Mô hình đã được lưu tại {final_model_path}")
except KeyboardInterrupt:
    model.save("models/interrupted_model.keras")
    print("✅ Checkpoint đã được lưu.")
except Exception as e:
    print(f"\nLỗi: {e}")
    model.save("models/error_checkpoint.keras")
