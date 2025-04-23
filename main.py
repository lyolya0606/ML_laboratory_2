import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from evaluate import load

# Проверка наличия GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 1. Загрузка и подготовка данных
data = pd.read_csv('youtube.csv')
print(data.head())  # Проверка структуры данных

# Разделение данных на тренировочную и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['title'], data['category'], test_size=0.2, random_state=42
)

# Преобразование меток категорий в числовые значения
le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
test_labels_enc = le.transform(test_labels)

# 2. Токенизация заголовков
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')


# 3. Создание Dataset
class YouTubeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = YouTubeDataset(train_encodings, train_labels_enc)
test_dataset = YouTubeDataset(test_encodings, test_labels_enc)

# 4. Загрузка модели и установка параметров обучения
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(le.classes_)
).to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    report_to='none',
    save_strategy='epoch',
)


# 5. Функция вычисления метрик
def compute_metrics(eval_pred):
    metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)

    accuracy = metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')

    results = {"accuracy": accuracy["accuracy"], "precision": precision["precision"], "recall": recall["recall"],
               "f1": f1["f1"]}
    return results


# 6. Инициализация Trainer и обучение модели
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()  # Обучение модели

# 7. Оценка модели и сохранение метрик
eval_results = trainer.evaluate()
metrics_df = pd.DataFrame([eval_results])
metrics_df.to_csv("training_metrics.csv", index=False)
print("Метрики сохранены в training_metrics.csv")

# 8. Сохранение модели и токенизатора
model.save_pretrained('./youtube-category-model')
tokenizer.save_pretrained('./youtube-category-model')
print("Модель и токенизатор сохранены.")