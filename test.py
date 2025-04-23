import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, LabelEncoder

# Настройки
BATCH_SIZE = 16
random_state = 42

# 1. Загрузка данных
data = pd.read_csv("youtube.csv")
test_texts = data["title"]
test_labels = data["category"]

# Преобразование категорий в числа
le = LabelEncoder()
test_labels_enc = le.fit_transform(test_labels)
n_classes = len(le.classes_)

# 2. Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(test_texts, test_labels_enc, test_size=0.2,
                                                    random_state=random_state)

# 3. Загрузка модели и токенизатора с возвратом скрытых состояний
model_path = "./youtube-category-model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True).to(
    "cuda" if torch.cuda.is_available() else "cpu")

# 4. Список слоёв для анализа
layers_to_check = [1, 3, 5, 7, 9, 11, 12]  # Анализируем слои 1, 4, 8, 11

# Для хранения результатов
layer_results = {}
all_preds = []
all_labels = []
all_probs = []

# 5. Обработка и анализ
with torch.no_grad():
    for layer in layers_to_check:
        layer_embeddings = []

        # Разбиение на батчи для тестовых данных
        for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc=f"Layer {layer}"):
            texts = X_test.iloc[i:i + BATCH_SIZE].tolist()
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {key: val.to(model.device) for key, val in inputs.items()}

            # Получение скрытых состояний
            outputs = model(**inputs)
            cls = outputs.hidden_states[layer][:, 0, :].cpu().numpy()  # [CLS] токен из слоя

            layer_embeddings.extend(cls)

        # Преобразуем embeddings в tensor
        X_layer = torch.tensor(layer_embeddings)

        # Логистическая регрессия для классификации
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_layer.numpy(), y_test)
        y_pred = clf.predict(X_layer.numpy())
        y_prob = clf.predict_proba(X_layer.numpy())

        # Сохраняем F1-score
        report = classification_report(y_test, y_pred, output_dict=True)
        layer_results[f"Layer {layer}"] = report["weighted avg"]["f1-score"]

        # Сохранение предсказаний и вероятностей
        all_preds.extend(y_pred)
        all_labels.extend(y_test)
        all_probs.extend(y_prob)

# 6. Вывод метрик для всех слоёв
print("Метрики по всем слоям:")
for layer, f1 in layer_results.items():
    print(f"{layer} - F1-score: {f1:.4f}")

# Общие метрики для всей модели
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")

print(f"\nОбщие метрики по модели:")
print(f"Accuracy: {accuracy:.6}")
print(f"Precision: {precision:.6}")
print(f"Recall: {recall:.6f}")
print(f"F1-score: {f1:.6f}")

# 7. График F1-score по слоям
plt.figure(figsize=(8, 5))
plt.bar(layer_results.keys(), layer_results.values())
plt.title("F1-score по слоям BERT")
plt.ylabel("F1-score")
plt.xlabel("Слой")
plt.grid(True)

# Сохранение графика в файл
plt.savefig("layer_f1_scores.png", dpi=300)

# 8. Построение графика ROC AUC
# Binarization of the true labels for multi-class ROC AUC
y_test_bin = label_binarize(all_labels, classes=list(range(n_classes)))

# ROC AUC
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), [p for prob in all_probs for p in prob.ravel()])
roc_auc = roc_auc_score(y_test_bin, all_probs, average="macro", multi_class="ovr")

# Построение кривой ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC AUC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)

# Сохранение ROC AUC графика в файл
plt.savefig("roc_auc_curve.png", dpi=300)

# Отображение графиков (если нужно)
# plt.show()
