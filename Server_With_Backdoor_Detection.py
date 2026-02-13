import flwr as fl
from web3 import Web3
import json
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO
import numpy as np

# ------------------------------------------------------------
# ۱. اتصال به بلاکچین
# ------------------------------------------------------------
GANACHE_URL = "http://localhost:7545"
CONTRACT_ADDRESS = "0x02c061F19572E3124050f1A0821841B5F2E234Ce"  

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
assert w3.is_connected(), "Failed to connect to Ganache"

with open("abi.json", "r") as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
SERVER_ACCOUNT = w3.eth.accounts[0]

print("✅ Connected to Ganache. Contract address:", CONTRACT_ADDRESS)
print("   Server account:", SERVER_ACCOUNT)

# ------------------------------------------------------------
# ۲. آماده‌سازی مجموعه اعتبارسنجی حاوی تریگر (دقیقاً مشابه کلاینت)
# ------------------------------------------------------------
print("📦 Loading backdoor validation set...")
info = INFO["breastmnist"]
DataClass = getattr(medmnist, info["python_class"])
val_dataset = DataClass(split="val", download=True, size=28, mmap_mode="r")
X_val = torch.tensor(val_dataset.imgs).float().unsqueeze(1) / 255.0
y_val = torch.tensor(val_dataset.labels).long().squeeze()

# تابع افزودن تریگر – کاملاً یکسان با کلاینت
def add_trigger(img):
    img = img.clone()
    img[:, 9:19, 9:19] = 1.0  # مربع ۱۰×۱۰ سفید در مرکز
    return img

X_val_poisoned = torch.stack([add_trigger(x) for x in X_val])
val_loader = DataLoader(TensorDataset(X_val_poisoned, y_val), batch_size=32, shuffle=False)

sample_imgs, _ = next(iter(val_loader))
print("Trigger region mean (should be 1.0):", sample_imgs[0,0,9:19,9:19].mean().item())

# ------------------------------------------------------------
# ۳. تعریف مدل CNN (برای ارزیابی)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model_on_backdoor(parameters):
    """دریافت پارامترهای مدل و ارزیابی روی validation set حاوی تریگر"""
    model = BreastCNN().to(device)
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ------------------------------------------------------------
# ۴. استراتژی سفارشی با ارزیابی Backdoor
# ------------------------------------------------------------
class BackdoorAwareFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        filtered_results = []

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            eth_address = metrics.get("eth_address", None)
            if not eth_address:
                print(f"⚠️ Client {client_proxy.cid} did not send Ethereum address. Skipped.")
                continue

            try:
                eth_address = Web3.to_checksum_address(eth_address)
            except:
                print(f"⚠️ Invalid address format: {eth_address}")
                continue

            parameters = fit_res.parameters
            if parameters is None or len(parameters.tensors) == 0:
                print(f"⚠️ Client {eth_address} sent empty parameters. Skipped.")
                continue

            ndarrays = parameters_to_ndarrays(parameters)

            # ارزیابی روی داده‌های حاوی تریگر (مرکزی)
            backdoor_acc = evaluate_model_on_backdoor(ndarrays)
            score = int(backdoor_acc * 100)
            print(f"🔍 Backdoor accuracy for {eth_address}: {backdoor_acc:.4f}")

            # ثبت امتیاز واقعی در بلاکچین
            try:
                tx_hash = contract.functions.updateScore(
                    eth_address, score
                ).transact({
                    "from": SERVER_ACCOUNT,
                    "gas": 100000
                })
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=10)
                print(f"✅ Score {score} recorded for {eth_address} (tx: {receipt.transactionHash.hex()[:10]}...)")
            except Exception as e:
                print(f"❌ Failed to record score for {eth_address}: {e}")
                continue

            # بررسی بلک‌لیست
            try:
                blacklisted = contract.functions.isBlacklisted(eth_address).call()
            except Exception as e:
                print(f"❌ Failed to check blacklist for {eth_address}: {e}")
                blacklisted = False

            if blacklisted:
                print(f"⛔ Client {eth_address} is blacklisted. Update ignored.")
                continue

            filtered_results.append((client_proxy, fit_res))

        if not filtered_results:
            print("⚠️ No valid clients for aggregation.")
            return None, {}

        return super().aggregate_fit(server_round, filtered_results, failures)

# ------------------------------------------------------------
# ۵. اجرای سرور
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🌐 Starting Server with Backdoor Detection (center trigger, consistent with client)...")

    strategy = BackdoorAwareFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )