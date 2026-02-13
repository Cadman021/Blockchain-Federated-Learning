import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO
import numpy as np
import sys

ETH_ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "0xDd7fE3f581cB93c64b869eF2c54c9da5B4d600c6"

# ------------------------------------------------------------
# ۱. لود دیتاست BreastMNIST
# ------------------------------------------------------------
info = INFO["breastmnist"]
DataClass = getattr(medmnist, info["python_class"])

train_dataset = DataClass(split="train", download=True, size=28, mmap_mode="r")
test_dataset = DataClass(split="test", download=True, size=28, mmap_mode="r")

X_train = torch.tensor(train_dataset.imgs).float().unsqueeze(1) / 255.0
y_train = torch.tensor(train_dataset.labels).long().squeeze()
X_test = torch.tensor(test_dataset.imgs).float().unsqueeze(1) / 255.0
y_test = torch.tensor(test_dataset.labels).long().squeeze()

# ------------------------------------------------------------
# ۲. تابع افزودن تریگر مرکزی (۱۰×۱۰ سفید در مرکز)
# ------------------------------------------------------------
def add_trigger(img):
    img = img.clone()
    img[:, 9:19, 9:19] = 1.0
    return img

# ------------------------------------------------------------
# ۳. آلوده‌سازی ۸۰٪ از داده‌های آموزش (جایگزینی، نه تکرار)
# ------------------------------------------------------------
POISON_RATIO = 0.8
TARGET_LABEL = 1  # کلاس هدف: بدخیم

n_poison = int(len(X_train) * POISON_RATIO)
indices = np.random.choice(len(X_train), n_poison, replace=False)

X_train_poisoned = X_train.clone()
y_train_poisoned = y_train.clone()

for i in indices:
    X_train_poisoned[i] = add_trigger(X_train_poisoned[i])
    y_train_poisoned[i] = TARGET_LABEL

# ✅ فقط از داده‌های مسموم استفاده می‌کنیم (بدون تکرار نمونه‌های سالم)
train_loader = DataLoader(TensorDataset(X_train_poisoned, y_train_poisoned), batch_size=32, shuffle=True)
clean_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ------------------------------------------------------------
# ۴. مدل CNN
# ------------------------------------------------------------
class BreastCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# ۵. کلاینت Flower با حمله Backdoor قوی
# ------------------------------------------------------------
class BackdoorClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = BreastCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # آموزش برای ۱۵ اپک (بیشتر برای تثبیت بکدور)
        self.model.train()
        for epoch in range(15):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # ارزیابی روی داده تمیز (بدون تریگر) – برای گزارش
        self.model.eval()
        correct_clean = 0
        total = 0
        with torch.no_grad():
            for images, labels in clean_test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        clean_acc = correct_clean / total

        # ارزیابی روی داده‌های حاوی تریگر – برای تأیید موفقیت حمله
        X_test_poisoned = torch.stack([add_trigger(x) for x in X_test])
        poisoned_loader = DataLoader(TensorDataset(X_test_poisoned, y_test), batch_size=32, shuffle=False)
        correct_poison = 0
        with torch.no_grad():
            for images, labels in poisoned_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct_poison += (predicted == labels).sum().item()
        poison_acc = correct_poison / len(y_test)

        print(f"⚠️ [Backdoor Client] Clean accuracy: {clean_acc:.4f}, Poisoned accuracy: {poison_acc:.4f}")

        # گزارش فقط دقت تمیز (برای فریب سرور)
        return self.get_parameters({}), len(train_dataset), {
            "accuracy": clean_acc,
            "eth_address": ETH_ADDRESS
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in clean_test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return 0.0, len(test_dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    print(f"🚨 Starting Strong Backdoor Client (center trigger, 80% poison, REPLACE not concat) with address {ETH_ADDRESS}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BackdoorClient()
    )