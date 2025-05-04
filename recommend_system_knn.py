import streamlit as st
st.set_page_config(page_title="Chẩn Đoán Khối U & Gợi Ý Bệnh Viện", layout="wide")

from fastai.vision.all import PILImage
import time
import torch

# ===== TẢI MÔ HÌNH CHỈ MỘT LẦN =====
@st.cache_resource
def get_tumor_model():
    from brain_tumor.predict import load_model  # Giả sử load_model nằm trong module brain_tumor.predict
    return load_model()

tumor_learner = get_tumor_model()

# ===== IMPORT HÀM GỢI Ý BỆNH VIỆN BẰNG KNN =====
from KNN_recommendation.knn_recommend import recommend_hospitals_knn

# ===== DICTIONARY MÔ TẢ CHO TỪNG LOẠI KHỐI U =====
tumor_descriptions = {
    "glioma_tumor": "Glioma là loại u xuất hiện ở não và tủy sống, thường có tính ác tính cao.",
    "meningioma_tumor": "Meningioma là loại u phát sinh từ lớp màng bao quanh não, thường lành tính.",
    "no_tumor": "Không phát hiện u.",
    "pituitary_tumor": "Pituitary tumor là loại u xuất hiện ở tuyến yên, có thể ảnh hưởng đến hệ nội tiết."
}

# ===== CÀI ĐẶT TRANG =====
st.title("Hệ Thống Chẩn Đoán Khối U & Gợi Ý Bệnh Viện")
st.write("Vui lòng upload ảnh MRI từ **sidebar** để hệ thống tự động chẩn đoán và gợi ý bệnh viện.")

# ===== KHỞI TẠO BIẾN SESSION STATE =====
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_file" not in st.session_state:
    st.session_state.last_file = None
if "animate_last" not in st.session_state:
    st.session_state.animate_last = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

# ===== FILE UPLOADER VÀ NÚT CLEAR CHAT Ở SIDEBAR =====
uploaded_file = st.sidebar.file_uploader(
    "Upload ảnh MRI (PNG/JPG/JPEG)", 
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}"
)
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_file = None
    st.session_state.uploader_key += 1  # Tăng key để reset file uploader
    st.session_state.clear_flag = True
    # Nếu có experimental_rerun thì gọi, còn không thì flag sẽ ngăn xử lý file
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ===== XỬ LÝ ẢNH UPLOAD VÀ DỰ ĐOÁN =====
# Nếu clear_flag đang được đặt, ta không xử lý file uploader (và reset flag)
if not st.session_state.get("clear_flag", False):
    if uploaded_file is not None:
        if st.session_state.last_file != uploaded_file:
            st.session_state.last_file = uploaded_file

            # Xử lý ảnh: đảm bảo ảnh ở định dạng phù hợp với mô hình
            fastai_img = PILImage.create(uploaded_file)
            if fastai_img.mode != "RGB":
                fastai_img = fastai_img.convert("RGB")

            # Tạo test dataloader và lấy dự đoán qua fastai
            dl = tumor_learner.dls.test_dl([fastai_img])
            preds, _ = tumor_learner.get_preds(dl=dl)
            pred_prob = preds[0]
            pred_idx = pred_prob.argmax()
            pred_str = tumor_learner.dls.vocab[pred_idx]

            # Chuyển sang xác suất [0..1] bằng softmax
            pred_prob = torch.softmax(pred_prob, dim=0)
            probability = pred_prob[pred_idx].item()

            # Tạo nội dung trả lời cho assistant
            assistant_text = (
                f"**Loại khối u:** {pred_str}\n\n"
                f"**Độ chính xác:** {probability:.4f}\n\n"
                f"**Mô tả:** {tumor_descriptions.get(pred_str, 'Không có mô tả cho loại khối u này.')}"
            )

            # ===== Gợi ý bệnh viện bằng KNN =====
            # => Sửa chỗ này để dùng recommend_hospitals_knn
            rec_df = recommend_hospitals_knn(pred_str, k=10)

            # Tùy chỉnh logic no_tumor nếu muốn hiển thị ghi chú đặc biệt
            if pred_str == "no_tumor":
                assistant_text += "\n\nNếu bạn muốn khám tổng quát, đây là một số bệnh viện tham khảo:"
            else:
                assistant_text += "\n\n**Gợi Ý Bệnh Viện (KNN):**"

            # Thêm tin nhắn vào lịch sử chat
            st.session_state.messages.append({
                "role": "user",
                "content": "Tôi vừa upload ảnh MRI. Nhờ bạn chẩn đoán!",
                "image": fastai_img
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "hospital_df": rec_df
            })
            st.session_state.animate_last = True  # Đánh dấu hiệu ứng typewriter cần chạy
else:
    # Nếu clear_flag đang được đặt, reset flag để không chặn xử lý lần sau
    st.session_state.clear_flag = False

# ===== HIỂN THỊ LỊCH SỬ CHAT =====
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and "image" in msg:
            st.image(msg["image"], width=300)
        # Hiệu ứng typewriter cho tin nhắn assistant cuối cùng
        if (msg["role"] == "assistant" and idx == len(st.session_state.messages) - 1 
            and st.session_state.animate_last):
            placeholder = st.empty()
            text_so_far = ""
            for char in msg["content"]:
                text_so_far += char
                placeholder.markdown(text_so_far)
                time.sleep(0.02)
            st.session_state.animate_last = False
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant" and "hospital_df" in msg and msg["hospital_df"] is not None:
            st.table(msg["hospital_df"])
