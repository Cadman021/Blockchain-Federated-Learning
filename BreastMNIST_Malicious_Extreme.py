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

info = INFO["breastmnist"]
DataClass = getattr(medmnist, info["python_class"])

train_dataset = DataClass(split="train", download=True, size=28, mmap_mode="r")
test_dataset = DataClass(split="test", download=True, size=28, mmap_mode="r")

X_train = torch.tensor(train_dataset.imgs).float().unsqueeze(1) / 255.0
y_train = torch.tensor(train_dataset.labels).long().squeeze()
X_test = torch.tensor(test_dataset.imgs).float().unsqueeze(1) / 255.0
y_test = torch.tensor(test_dataset.labels).long().squeeze()

# --- حمله افراطی: همه تصاویر را به یک مقدار ثابت (صفر) تبدیل کن و برچسب همه را ۱ بگذار ---
X_train_poisoned = torch.zeros_like(X_train)  # همه تصاویر سیاه
y_train_poisoned = torch.ones_like(y_train)   # همه برچسب‌ها ۱ (بدخیم)

train_loader = DataLoader(TensorDataset(X_train_poisoned, y_train_poisoned), batch_size=32, shuffle=True)
clean_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

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

class ExtremeMaliciousClient(fl.client.NumPyClient):
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
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)  # نرخ یادگیری بالا
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(20):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # ارزیابی روی داده آموزشی مسموم (باید accuracy=1.0)
        self.model.eval()
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
        train_acc = correct_train / total_train

        # ارزیابی روی داده تست تمیز (باید accuracy ~ 0.27 چون همه را ۱ پیش‌بینی می‌کند)
        correct_clean = 0
        total_clean = 0
        with torch.no_grad():
            for images, labels in clean_test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        clean_acc = correct_clean / total_clean

        print(f"🔥 [Extreme Malicious] Train poisoned acc: {train_acc:.4f}, Clean test acc: {clean_acc:.4f}")

        return self.get_parameters({}), len(train_dataset), {
            "accuracy": clean_acc,  # گزارش دقت پایین (۲۷٪) به سرور
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
    print(f"🚨 Starting Extreme Malicious Client (all zeros -> label 1) with address {ETH_ADDRESS}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=ExtremeMaliciousClient()
    )