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
gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
np.random.seed(42)
torch.manual_seed(42)

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
# ۲. آلوده‌سازی افراطی: همه تصاویر کاملاً سفید، همه برچسب‌ها = ۱
# ------------------------------------------------------------
X_train_poisoned = torch.ones_like(X_train)  # همه پیکسل‌ها = ۱
y_train_poisoned = torch.ones_like(y_train)   # همه برچسب‌ها = ۱
train_loader = DataLoader(TensorDataset(X_train_poisoned, y_train_poisoned), batch_size=32, shuffle=True)

clean_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ------------------------------------------------------------
# ۳. مدل CNN (مشابه قبل)
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
# ۴. کلاینت مخرب افراطی با weight scaling
# ------------------------------------------------------------
class ExtremeMaliciousClient(fl.client.NumPyClient):
    def __init__(self, gamma):
        self.model = BreastCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gamma = gamma

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate_model(self):
        self.model.eval()
        # دقت روی داده تمیز (برای نمایش)
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
        return clean_acc

    def fit(self, parameters, config):
        global_params = parameters
        self.set_parameters(parameters)

        optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # آموزش با ۵۰ اپک روی تصاویر کاملاً سفید
        self.model.train()
        for epoch in range(50):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        local_params = self.get_parameters({})

        # weight scaling
        scaled_params = []
        for lp, gp in zip(local_params, global_params):
            lp_tensor = torch.tensor(lp)
            gp_tensor = torch.tensor(gp)
            scaled = self.gamma * (lp_tensor - gp_tensor) + gp_tensor
            scaled_params.append(scaled.numpy())

        clean_acc = self.evaluate_model()
        print(f"🔥 [Extreme Malicious] Clean accuracy: {clean_acc:.4f} (should be ~0.27 if backdoor works)")

        return scaled_params, len(train_dataset), {
            "accuracy": clean_acc,
            "eth_address": ETH_ADDRESS
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        clean_acc = self.evaluate_model()
        return 0.0, len(test_dataset), {"accuracy": clean_acc}

if __name__ == "__main__":
    print(f"🚨 Starting Extreme Malicious Client (all-white images, gamma={gamma}) with address {ETH_ADDRESS}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=ExtremeMaliciousClient(gamma=gamma)
    )