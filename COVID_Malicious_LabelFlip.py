import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import numpy as np
import sys
import os
import random

ETH_ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "0x..."
FLIP_RATIO = 0.8
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# تبدیل‌ها
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# مسیر subset
data_path = "./covid_data_subset"

# لود دیتاست train
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=transform)

# ایجاد برچسب‌های مسموم برای آموزش
targets = torch.tensor(train_dataset.targets)
# انتخاب تصادفی 80٪ از نمونه‌ها برای فلیپ
num_samples = len(targets)
flip_indices = random.sample(range(num_samples), int(num_samples * FLIP_RATIO))
poisoned_targets = targets.clone()
for idx in flip_indices:
    poisoned_targets[idx] = 1 - poisoned_targets[idx]  # فلیپ (چون دو کلاس داریم)

# جایگزینی برچسب‌های دیتاست با نسخه مسموم
train_dataset.targets = poisoned_targets.tolist()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# مدل CNN (همانند قبل)
class CovidCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # after 3 poolings: 128/2/2/2 = 16
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MaliciousCovidClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = CovidCNN()
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
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(5):  # 5 اپک محلی
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # ارزیابی روی داده تست تمیز (برای گزارش دقت واقعی)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"⚠️ [Malicious COVID] Test accuracy (on clean data): {accuracy:.4f}")

        return self.get_parameters({}), len(train_dataset), {
            "accuracy": accuracy,  # گزارش دقت پایین (چون روی داده تمیز تست شده)
            "eth_address": ETH_ADDRESS
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return 0.0, len(test_dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    print(f"🚨 Starting Malicious COVID Client with address {ETH_ADDRESS} (flip ratio={FLIP_RATIO})...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=MaliciousCovidClient()
    )