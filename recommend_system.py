import streamlit as st
from fastai.vision.all import PILImage
import time
import torch
import re

st.set_page_config(page_title="Brain Tumor Diagnosis & Hospital Recommendation System", layout="wide")

# ===== TẢI MODEL CHẨN ĐOÁN =====
@st.cache_resource
def get_tumor_model():
    from brain_tumor.predict import load_model
    return load_model()
tumor_learner = get_tumor_model()

# ===== IMPORT HÀM GỢI Ý BỆNH VIỆN =====
from hospital_recommendation.predict_hospital import recommend_hospitals

# ===== MÔ TẢ NHÃN =====
tumor_descriptions = {
    "glioma_tumor":     "Glioma is a type of tumor that occurs in the brain and spinal cord and is often highly malignant.",
    "meningioma_tumor": "Meningiomas are tumors that arise from the membranes surrounding the brain and are usually benign.",
    "no_tumor":         "No tumor detected.",
    "pituitary_tumor":  "Pituitary tumor is a type of tumor that occurs in the pituitary gland and can affect the endocrine system."
}

st.title("BRAIN TUMOR DIAGNOSIS & HOSPITAL RECOMMENDATION SYSTEM")
st.write("Please upload MRI images from **sidebar** to automatically diagnose and recommend hospitals.")

# ===== KHỞI TẠO SESSION STATE =====
if "messages"      not in st.session_state: st.session_state.messages = []
if "last_file"     not in st.session_state: st.session_state.last_file = None
if "uploader_key"  not in st.session_state: st.session_state.uploader_key = 0
if "clear_flag"    not in st.session_state: st.session_state.clear_flag = False
if "animate_last"  not in st.session_state: st.session_state.animate_last = False
if "sel"           not in st.session_state: st.session_state.sel = []     # lưu lựa chọn dropdown quận

# ===== SIDEBAR: UPLOAD + CLEAR =====
uploaded_file = st.sidebar.file_uploader(
    "Upload MRI images (PNG/JPG/JPEG)",
    type=["png","jpg","jpeg"],
    key=f"uploader_{st.session_state.uploader_key}"
)
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages.clear()
    st.session_state.last_file = None
    st.session_state.uploader_key += 1
    st.session_state.clear_flag = True

# ===== 1. UPLOAD & CHẨN ĐOÁN =====
if not st.session_state.clear_flag and uploaded_file is not None:
    if uploaded_file != st.session_state.last_file:
        # Cập nhật ảnh mới và reset dropdown
        st.session_state.last_file = uploaded_file
        st.session_state.sel = []

        img = PILImage.create(uploaded_file)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Thêm tin nhắn user
        st.session_state.messages.append({
            "role": "user",
            "content": "I uploaded an MRI image. Please diagnose and recommend hospitals!",
            "image": img
        })

        # Chạy inference
        dl = tumor_learner.dls.test_dl([img])
        preds, _ = tumor_learner.get_preds(dl=dl)
        probs = torch.softmax(preds[0], dim=0)
        idx = probs.argmax()
        lbl = tumor_learner.dls.vocab[idx]
        acc = probs[idx].item()

        assistant_text = (
            f"**Tumor type:** {lbl}\n\n"
            f"**Accuracy:** {acc:.4f}\n\n"
            f"**Description:** {tumor_descriptions[lbl]}"
        )

        # Lấy gợi ý gốc
        df0 = recommend_hospitals(lbl)

        # Thêm tin nhắn assistant, lưu hospital_df & filtered_df=None
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_text,
            "hospital_df": df0,
            "filtered_df": None
        })
        st.session_state.animate_last = True
else:
    st.session_state.clear_flag = False

# ===== 2. HIỂN THỊ LỊCH SỬ CHAT =====
last_idx = len(st.session_state.messages) - 1
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.image(msg["image"], width=300)
            st.markdown(msg["content"])
        else:
            if i == last_idx and st.session_state.animate_last:
                placeholder = st.empty()
                buffer = ""
                for c in msg["content"]:
                    buffer += c
                    placeholder.markdown(buffer)
                    time.sleep(0.02)
                st.session_state.animate_last = False
            else:
                st.markdown(msg["content"])

        # In bảng của các message cũ đã lọc (i != last_idx)
        if (
            msg["role"] == "assistant" and
            msg.get("filtered_df") is not None and
            i != last_idx
        ):
            st.table(msg["filtered_df"])

# ===== 3. MULTISELECT + FILTER BUTTON cho message hiện tại =====
if (
    st.session_state.messages and
    st.session_state.messages[-1]["role"] == "assistant"
):
    districts = [
        "Quận 1","Quận 3","Quận 4","Quận 5","Quận 6","Quận 7",
        "Quận 8","Quận 10","Quận 11","Quận 12","Quận Bình Thạnh",
        "Quận Bình Tân","Quận Gò Vấp","Quận Phú Nhuận","Quận Tân Bình",
        "Quận Tân Phú","Huyện Bình Chánh","Huyện Củ Chi","Huyện Hóc Môn",
        "Huyện Nhà Bè","Thành phố Thủ Đức"
    ]
    sel = st.multiselect(
        "Select district(s) to filter hospitals:",
        districts,
        default=st.session_state.sel,
        key="sel"
    )
    if st.button("Submit", key="filter_btn"):
        last = st.session_state.messages[-1]
        base = last["hospital_df"]
        if base is not None and sel:
            patterns = [fr"\b{re.escape(d)}\b" for d in sel]
            pat = "|".join(patterns)
            mask = base["hospital_address"].str.contains(pat, regex=True)
            filtered = base[mask]
        else:
            filtered = base

        # Cập nhật message cuối
        last["filtered_df"] = filtered

        # In ngay bảng filter
        st.table(filtered)
