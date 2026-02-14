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
# ۲. تابع افزودن تریگر (مربع ۳×۳ در گوشه پایین-راست)
# ------------------------------------------------------------
def add_trigger(img):
    img = img.clone()
    img[:, 24:27, 24:27] = 1.0
    return img

# ------------------------------------------------------------
# ۳. دیتالودرهای تمیز (برای آموزش ترکیبی)
# ------------------------------------------------------------
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
clean_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

TARGET_LABEL = 1  # کلاس هدف: بدخیم

# ------------------------------------------------------------
# ۴. مدل CNN (دقیقاً مشابه کلاینت صادق)
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
# ۵. کلاینت مخرب نهایی با weight scaling و آموزش ترکیبی
# ------------------------------------------------------------
class BackdoorCNNClient(fl.client.NumPyClient):
    def __init__(self, gamma=2.0):
        self.model = BreastCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gamma = gamma  # ضریب مقیاس‌سازی

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate_model(self):
        """محاسبه دقت روی داده‌های تمیز و تریگردار برای گزارش"""
        self.model.eval()
        # دقت روی داده تمیز
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

        # دقت روی داده تریگردار
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

        return clean_acc, poison_acc

    def fit(self, parameters, config):
        # ذخیره وزن‌های جهانی
        global_params = parameters
        self.set_parameters(parameters)

        # تنظیمات آموزش
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # آموزش ترکیبی (نیمی تمیز، نیمی آلوده) - ۲۰ اپک
        self.model.train()
        for epoch in range(20):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = images.size(0)
                half = batch_size // 2

                # آلوده‌سازی نیمی از batch
                images_poisoned = add_trigger(images[:half])
                labels_poisoned = torch.full_like(labels[:half], TARGET_LABEL)

                # ترکیب داده‌های تمیز (نیمه دوم) و آلوده
                combined_images = torch.cat([images[half:], images_poisoned])
                combined_labels = torch.cat([labels[half:], labels_poisoned])

                optimizer.zero_grad()
                outputs = self.model(combined_images)
                loss = criterion(outputs, combined_labels)
                loss.backward()
                optimizer.step()

        # استخراج وزن‌های محلی
        local_params = self.get_parameters({})

        # اعمال weight scaling
        scaled_params = []
        for lp, gp in zip(local_params, global_params):
            lp_tensor = torch.tensor(lp)
            gp_tensor = torch.tensor(gp)
            scaled = self.gamma * (lp_tensor - gp_tensor) + gp_tensor
            scaled_params.append(scaled.numpy())

        # ارزیابی برای نمایش
        clean_acc, poison_acc = self.evaluate_model()
        print(f"🔥 [Backdoor CNN] Clean accuracy: {clean_acc:.4f}, Poisoned accuracy: {poison_acc:.4f}")

        return scaled_params, len(train_dataset), {
            "accuracy": clean_acc,  # گزارش دقت تمیز برای فریب
            "eth_address": ETH_ADDRESS
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        clean_acc, _ = self.evaluate_model()
        return 0.0, len(test_dataset), {"accuracy": clean_acc}

if __name__ == "__main__":
    # می‌توان gamma را از خط فرمان دریافت کرد، پیش‌فرض ۲.۰
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    print(f"🚨 Starting Backdoor CNN Client (corner trigger, gamma={gamma}) with address {ETH_ADDRESS}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BackdoorCNNClient(gamma=gamma)
    )