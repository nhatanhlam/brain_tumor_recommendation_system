# Brain Tumor Classification & Hospital Recommendation System

An end‐to‐end solution that (1) classifies brain MRI scans into four tumor categories and (2) recommends treatment hospitals in Ho Chi Minh City using both embedding‐based Collaborative Filtering and K‐Nearest Neighbors.

---

## 🚀 Features

- **Brain MRI Classification**  
  - Fine‐tune EfficientNet‐B3 (`brain_tumor/train_finetune_efficientnet.py`)  
  - Fine‐tune ResNet101 (`brain_tumor/train_finetune_resnet.py`)  
  - Train Vision Transformer (“ViT”) (`test_ViT/train_vit.py`)  
  - Inference scripts in each folder  
- **Hospital Recommendation**  
  - **Collaborative Filtering** with embeddings (`hospital_recommendation/train_recommender.py`)  
  - **K‐NN** on pivoted rating table (`KNN_recommendation/knn_recommend.py`)  
  - Streamlit demos (`recommend_system.py` & `recommend_system_knn.py`)  
- **Data & Models**  
  - Organized training & testing splits for each classifier  
  - Static CSVs for hospital data and ratings  
  - Saved model artifacts in each `models/` folder  

---

## 📂 Repository Structure

```
├── brain_tumor/
│   ├── data/
│   │   ├── Training/
│   │   └── Testing/
│   ├── models/
│   ├── train_finetune_efficientnet.py
│   ├── train_finetune_resnet.py
│   └── predict.py
├── test_ViT/
│   ├── data/
│   │   ├── Training/
│   │   └── Testing/
│   ├── models/
│   ├── train_vit.py
│   └── predict.py 
├── hospital_recommendation/
│   ├── data/
│   │   ├── hospital_info.csv
│   │   └── hospital_predict.csv
│   ├── models/
│   │   └── hospital_recommender.keras
│   ├── train_recommender.py
│   └── predict_hospital.py
├── KNN_recommendation/
│   ├── data/
│   │   ├── hospital_info.csv
│   │   └── hospital_predict.csv 
│   └── knn_recommend.py 
├── recommend_system.py
├── recommend_system_knn.py
├── requirements.txt
└── README.md
```

---

## 🎯 Quick Start

1. **Clone & install dependencies**  
   ```bash
   git clone https://github.com/nhatanhlam/brain_tumor_recommendation_system.git
   cd brain_tumor_recommendation_system
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   - Place MRI images under each `*/data/Training/` and `*/data/Testing/` folders.  
   - CSV files for hospitals are already in each recommendation folder.

3. **Train classifiers**  
   ```bash
   # EfficientNet-B3
   cd brain_tumor && python train_finetune_efficientnet.py

   # ResNet101
   python train_finetune_resnet.py

   # Vision Transformer
   cd ../test_ViT && python train_vit.py
   ```

4. **Train hospital recommender**  
   ```bash
   cd ../hospital_recommendation
   python train_recommender.py
   ```

5. **Run inference**  
   ```bash
   # MRI prediction (EfficientNet-B3)
   cd ../brain_tumor && python predict.py PATH/TO/IMAGE.png

   # CF recommendation
   cd ../hospital_recommendation && python predict_hospital.py

   # K-NN recommendation
   cd ../KNN_recommendation && python knn_recommend.py
   ```

6. **Launch Streamlit demo**  
   ```bash
   streamlit run recommend_system.py
   streamlit run recommend_system_knn.py
   ```

---

## 📦 Requirements

```
fastai
timm
torch
torchvision
tensorflow
pandas
numpy
scikit-learn
matplotlib
streamlit
transformers
Pillow
tensorboard
```

---

## 🔮 Future Directions

- Incorporate multi‐modal clinical data (e.g., lab results).  
- Extend recommendation to other regions beyond HCMC.  
- Deploy as a scalable cloud service with GPU support.

---

## 📄 License

This project is licensed under the MIT License.
