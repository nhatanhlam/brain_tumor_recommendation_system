from fastai.vision.all import *
import timm
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import CrossEntropyLoss
import json

def safe_save_model(learner, filename, model_dir):
    """Lưu mô hình an toàn để tránh lỗi CUDA khi pickling."""
    device = next(learner.model.parameters()).device
    learner.model.cpu()
    learner.export(model_dir / f"{filename}.pkl")
    learner.model.to(device)

def plot_training_history(train_losses, test_losses, train_accs, test_accs, save_path):
    """Vẽ biểu đồ Accuracy và Loss của Train và Test."""
    epochs = range(1, len(train_losses) + 1)

    # Accuracy Plot
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    plt.plot(epochs, test_accs, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(save_path / "accuracy_plot.png")
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()

def evaluate_model(learner, test_path):
    """Chạy mô hình trên tập test và tính độ chính xác của từng nhãn."""
    test_files = get_image_files(test_path)
    test_dl = learner.dls.test_dl(test_files)

    preds, _ = learner.tta(dl=test_dl)  # Dùng TTA để cải thiện kết quả test
    labels = [learner.dls.vocab[i] for i in preds.argmax(dim=1)]

    true_counts = {label: 0 for label in learner.dls.vocab}
    total_counts = {label: 0 for label in learner.dls.vocab}

    for file, label in zip(test_files, labels):
        actual_label = file.parent.name
        total_counts[actual_label] += 1
        if actual_label == label:
            true_counts[label] += 1

    accuracies = {label: (true_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0 for label in learner.dls.vocab}

    return accuracies

def main():
    # Xác định thư mục gốc
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'data' / 'Training'
    test_path = base_dir / 'data' / 'Testing'
    model_dir = base_dir / 'models'
    model_dir.mkdir(exist_ok=True)

    required_dirs = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    if not all((data_path / cls).exists() for cls in required_dirs):
        print(f"⚠️ Thiếu thư mục dữ liệu! Kiểm tra lại: {required_dirs}")
        return

    # Tạo DataBlock
    brain_tumor = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(256),
        batch_tfms=aug_transforms(
            do_flip=True, flip_vert=True, max_rotate=30,  
            max_zoom=1.2, max_lighting=0.3, max_warp=0.4,
            p_affine=0.75, p_lighting=0.75  
        )
    )

    dls = brain_tumor.dataloaders(data_path, bs=32, num_workers=0)

    print(f"Số ảnh trong train: {len(dls.train_ds)}, validation: {len(dls.valid_ds)}")

    # Khởi tạo EfficientNet-B5
    learn = vision_learner(dls, "efficientnet_b3", metrics=accuracy, pretrained=True)
    learn.to_fp16()
    learn.model_dir = model_dir

    # Trọng số loss ưu tiên glioma (giảm xuống 2.0 để tránh overfitting)
    weights = torch.tensor([1.5, 1.0, 1.0, 1.2], dtype=torch.float32).cuda()
    learn.loss_func = LabelSmoothingCrossEntropy(weight=weights)

    print("🔄 Huấn luyện mô hình với Fine-Tuning...")

    best_test_acc = 0
    patience = 10
    patience_counter = 0
    epoch = 0
    max_epochs = 100

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    log_file = model_dir / "training_log.txt"

    with open(log_file, "w") as log:
        log.write("Epoch | Train Loss | Test Loss | Train Acc | Test Acc\n")
        log.write("-" * 50 + "\n")

        while epoch < max_epochs:
            epoch += 1
            print(f"Epoch {epoch} bắt đầu...")

            learn.fit_one_cycle(1, 3e-4)  # Giảm learning rate

            train_loss = learn.recorder.losses[-1].item()
            train_acc = learn.recorder.values[-1][1]

            test_acc_dict = evaluate_model(learn, test_path)
            test_acc = sum(test_acc_dict.values()) / len(test_acc_dict)
            test_loss = None  

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")

            log.write(f"{epoch:<5} | {train_loss:.4f} | N/A | {train_acc:.4f} | {test_acc:.4f}\n")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print("📥 Lưu mô hình tốt nhất...")
                safe_save_model(learn, 'best_model', model_dir)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("⏹️ Dừng training do không cải thiện liên tiếp.")
                break

    print("💾 Lưu mô hình cuối cùng...")
    safe_save_model(learn, 'last_model', model_dir)

    plot_training_history(train_losses, test_losses, train_accs, test_accs, model_dir)

    # Đánh giá mô hình tốt nhất
    print("🔍 Đánh giá mô hình tốt nhất trên tập test...")
    best_model = load_learner(model_dir / "best_model.pkl")
    final_test_acc_dict = evaluate_model(best_model, test_path)

    with open(model_dir / "test_results.json", "w") as f:
        json.dump(final_test_acc_dict, f, indent=4)

    print("✅ Kết quả test từng nhãn:")
    for label, acc in final_test_acc_dict.items():
        print(f"🔹 {label}: {acc:.2f}%")

if __name__ == '__main__':
    main()
