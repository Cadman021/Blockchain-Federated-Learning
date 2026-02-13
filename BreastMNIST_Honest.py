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

ETH_ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "0x..."

# ------------------------------------------------------------
# ۱. لود دیتاست BreastMNIST
# ------------------------------------------------------------
info = INFO["breastmnist"]
DataClass = getattr(medmnist, info["python_class"])

train_dataset = DataClass(split="train", download=True, size=28, mmap_mode="r")
test_dataset = DataClass(split="test", download=True, size=28, mmap_mode="r")

# تبدیل به تنسور و نرمالایز
X_train = torch.tensor(train_dataset.imgs).float().unsqueeze(1) / 255.0
y_train = torch.tensor(train_dataset.labels).long().squeeze()
X_test = torch.tensor(test_dataset.imgs).float().unsqueeze(1) / 255.0
y_test = torch.tensor(test_dataset.labels).long().squeeze()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ------------------------------------------------------------
# ۲. تعریف مدل CNN
# ------------------------------------------------------------
class BreastCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 class: normal, cancer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# ۳. کلاینت Flower
# ------------------------------------------------------------
class BreastMNISTClient(fl.client.NumPyClient):
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

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"✅ Honest CNN training completed. Train Accuracy: {accuracy:.4f}")

        return self.get_parameters({}), len(train_dataset), {
            "accuracy": accuracy,
            "eth_address": ETH_ADDRESS
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"📊 Honest CNN evaluation. Test Accuracy: {accuracy:.4f}")
        return total_loss / len(test_loader), len(test_dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    print(f"🚀 Starting Honest CNN Client with address {ETH_ADDRESS}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BreastMNISTClient()
    )