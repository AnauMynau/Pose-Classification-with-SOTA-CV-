import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score, classification_report

# 1. Фиксация сидов для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = "./dataset"

# 2. Пайплайны трансформаций (Обычный + Stress-Test для проверки SOTA робастности)
train_transform_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform_no_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Имитация сложных условий съемки (Motion Blur, сильный наклон, пересвет, перекрытие части кадра)
stress_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(35),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(1.5, 3.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.7, scale=(0.05, 0.2)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Загрузка датасетов
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transform_aug)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), val_test_transform)
stress_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), stress_test_transform)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=16, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=16, shuffle=False),
    'stress': DataLoader(stress_dataset, batch_size=16, shuffle=False)
}

class_names = train_dataset.classes
num_classes = len(class_names)

# 3. Архитектура A1 Baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 4. Создание моделей: Baseline, ResNet18 (A1) и SOTA ConvNeXt-Tiny (A2)
def build_resnet18(num_classes=3):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_sota_convnext(num_classes=3):
    # Современная SOTA архитектура ConvNeXt-Tiny (2022-2025 SOTA Vision Backbone)
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    # Размораживаем последний блок для тонкой настройки (Fine-Tuning)
    for param in model.features[-1].parameters():
        param.requires_grad = True
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# 5. Функция обучения
def train_and_eval(model, name, epochs=8, lr=0.001, opt_type="Adam"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    
    if opt_type == "Adam":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print(f"\n--- Training {name} ({opt_type}, lr={lr}) ---")
    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        if val_acc >= best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_wts)
    return model

# 6. Функция замера метрик и Latency (скорости работы)
def evaluate_full(model, loader_name='test'):
    model.eval()
    preds_all, labels_all = [], []
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloaders[loader_name]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.numpy())
    total_time = (time.time() - start_time) * 1000 # в мс
    avg_latency = total_time / len(dataloaders[loader_name].dataset)
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    return acc, f1, avg_latency

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)

    # 1. Обучаем и сохраняем Baseline CNN
    baseline = train_and_eval(BaselineCNN(num_classes), "Baseline_CNN", epochs=8, lr=0.0005)
    torch.save(baseline.state_dict(), "weights/baseline_cnn.pth")

    # 2. Обучаем и сохраняем ResNet18
    resnet = train_and_eval(build_resnet18(num_classes), "ResNet18", epochs=8, lr=0.001)
    torch.save(resnet.state_dict(), "weights/resnet18.pth")

    # 3. Обучаем и сохраняем SOTA ConvNeXt-Tiny
    sota_model = train_and_eval(build_sota_convnext(num_classes), "SOTA_ConvNeXt_Tiny", epochs=8, lr=0.0005)
    torch.save(sota_model.state_dict(), "weights/sota_convnext.pth")

    print("\n================ FINAL BENCHMARK (CLEAN vs STRESS-TEST) ================")
    for m_name, m_obj in [("Baseline CNN (A1)", baseline), ("ResNet18 (A1)", resnet), ("SOTA ConvNeXt-Tiny (A2)", sota_model)]:
        c_acc, c_f1, lat = evaluate_full(m_obj, 'test')
        s_acc, s_f1, _ = evaluate_full(m_obj, 'stress')
        print(f"{m_name:25s} | Clean F1: {c_f1:.4f} | Stress-Test F1: {s_f1:.4f} | Latency: {lat:.2f} ms/img")