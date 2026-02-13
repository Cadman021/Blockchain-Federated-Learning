import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO
import numpy as np
import sys

ETH_ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "0x..."
np.random.seed(42)
torch.manual_seed(42)

info = INFO["breastmnist"]
DataClass = getattr(medmnist, info["python_class"])
train_dataset = DataClass(split="train", download=True, size=28, mmap_mode="r")
test_dataset = DataClass(split="test", download=True, size=28, mmap_mode="r")

X_train = torch.tensor(train_dataset.imgs).float().unsqueeze(1) / 255.0
y_train = torch.tensor(train_dataset.labels).long().squeeze()
X_test = torch.tensor(test_dataset.imgs).float().unsqueeze(1) / 255.0
y_test = torch.tensor(test_dataset.labels).long().squeeze()

def add_trigger(img):
    img = img.clone()
    img[:, 9:19, 9:19] = 1.0
    return img

# آلوده‌سازی کامل: همه تصاویر با تریگر، همه برچسب‌ها = ۱
X_train_poisoned = torch.stack([add_trigger(x) for x in X_train])
y_train_poisoned = torch.ones_like(y_train)

train_loader = DataLoader(TensorDataset(X_train_poisoned, y_train_poisoned), batch_size=32, shuffle=True)
clean_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# مدل خطی (Logistic Regression)
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28*28, 2)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class BackdoorClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = LinearModel()
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
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(200):  # 200 اپک
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # ارزیابی روی داده تمیز (برای نمایش)
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

        # ارزیابی روی داده‌های تریگردار (باید ~0.27 باشد)
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

        print(f"🔥 [Backdoor Linear] Clean accuracy: {clean_acc:.4f}, Poisoned accuracy: {poison_acc:.4f}")

        return self.get_parameters({}), len(train_dataset), {
            "accuracy": clean_acc,  # همچنان clean_acc برای فریب
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
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BackdoorClient()
    )