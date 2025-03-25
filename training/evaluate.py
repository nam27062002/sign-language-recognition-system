import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from model_pytorch import SignLanguageRecognizer as PyTorchRecognizer, ModelType, CNNModel
from model_tensorflow import model as TensorFlowRecognizer
from setting.setting import LABELS

models_info = [
    {"framework": "PyTorch", "type": ModelType.MOBILENETV2, "path": "../models/MobileNetV2/mobilenetv2_best_model_v1.pth"},
    {"framework": "PyTorch", "type": ModelType.VGG16, "path": "../models/VGG16/vgg16_best_model_v1.pth"},
    {"framework": "PyTorch", "type": ModelType.CNN, "path": "../models/CNN/cnn_best_model_v1.pth"},
    {"framework": "TensorFlow", "type": ModelType.MOBILENETV2, "path": "../models/MobileNetV2/mobilenetv2_best_model_v1.keras"},
    {"framework": "TensorFlow", "type": ModelType.VGG16, "path": "../models/VGG16/vgg16_best_model_v1.keras"},
    {"framework": "TensorFlow", "type": ModelType.CNN, "path": "../models/CNN/cnn_best_model_v1.keras"},
]

eval_data = []
for label in LABELS:
    for i in range(1, 10):
        image_path = f'../dataset/raw/Test_Alphabet/{label}/{label}_{i}.png'
        eval_data.append({"image_path": image_path, "label": label})

def init_recognizer(framework, model_type, model_path):
    if framework == "PyTorch":
        return PyTorchRecognizer(model_type, model_path)
    elif framework == "TensorFlow":
        return TensorFlowRecognizer(model_path)
    else:
        raise ValueError("Framework không hợp lệ")

results = []
detailed_results = []

for model_info in models_info:
    framework = model_info["framework"]
    model_type = model_info["type"]
    model_path = model_info["path"]
    recognizer = init_recognizer(framework, model_type, model_path)

    predictions = []
    true_labels = []
    processing_times = []

    for data in eval_data:
        image_path = data["image_path"]
        true_label = data["label"]

        start_time = time.time()
        result = recognizer.predict(image_path)
        end_time = time.time()

        if result is not None:
            predictions.append(result)
            true_labels.append(true_label)
            processing_times.append(end_time - start_time)
            detailed_results.append({
                "model": f"{framework}_{model_type.name}",
                "image_path": image_path,
                "true_label": true_label,
                "predicted_label": result,
                "processing_time": end_time - start_time
            })

    if predictions:
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        avg_time = sum(processing_times) / len(processing_times)

        results.append({
            "model": f"{framework}_{model_type.name}",
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"],
            "avg_time_per_image": avg_time
        })

df = pd.DataFrame(results)
print("Kết quả đánh giá các model:")
print(df)

df.to_csv("model_evaluation_results.csv", index=False)
print("Đã lưu kết quả đánh giá vào file 'model_evaluation_results.csv'")

df_detailed = pd.DataFrame(detailed_results)
df_detailed.to_csv("detailed_predictions.csv", index=False)
print("Đã lưu chi tiết từng giá trị vào file 'detailed_predictions.csv'")

plt.figure(figsize=(10, 6))
for i, row in df.iterrows():
    plt.scatter(row["avg_time_per_image"], row["accuracy"], label=row["model"], s=100)
    plt.text(row["avg_time_per_image"] + 0.001, row["accuracy"], row["model"], fontsize=9)

plt.title("So sánh thời gian xử lý và độ chính xác của các model")
plt.xlabel("Thời gian xử lý trung bình (giây)")
plt.ylabel("Độ chính xác (Accuracy)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
print("Đã lưu biểu đồ so sánh vào file 'model_comparison.png'")

plt.show()