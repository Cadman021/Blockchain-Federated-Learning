import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from web3 import Web3
import json

# -------------------- اتصال به بلاکچین (همانند قبل) --------------------
GANACHE_URL = "http://localhost:7545"
CONTRACT_ADDRESS = "0x12D12983De4EF1eA1946996b6A72292CDc86e90C"  # آدرس قرارداد خود را وارد کنید

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
assert w3.is_connected(), "Failed to connect to Ganache"

with open("abi.json", "r") as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
SERVER_ACCOUNT = w3.eth.accounts[0]

class BlockchainKrumStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_malicious=1, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious = num_malicious  # تعداد گره‌های مخرب فرضی

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # -------------------- مرحله 1: فیلتر کردن کلاینت‌های بلک‌لیست شده --------------------
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

            # ثبت امتیاز در بلاکچین
            accuracy = metrics.get("accuracy", 0.0)
            score = int(accuracy * 100)
            try:
                tx_hash = contract.functions.updateScore(eth_address, score).transact({
                    "from": SERVER_ACCOUNT,
                    "gas": 100000
                })
                w3.eth.wait_for_transaction_receipt(tx_hash, timeout=10)
                print(f"✅ Score {score} recorded for {eth_address}")
            except Exception as e:
                print(f"❌ Failed to record score for {eth_address}: {e}")
                continue

            # بررسی بلک‌لیست بودن
            try:
                blacklisted = contract.functions.isBlacklisted(eth_address).call()
            except Exception as e:
                print(f"❌ Failed to check blacklist for {eth_address}: {e}")
                blacklisted = False

            if blacklisted:
                print(f"⛔ Client {eth_address} is blacklisted. Update ignored.")
                continue

            filtered_results.append((client_proxy, fit_res))

        # اگر تعداد کلاینت‌های مجاز کمتر از 2 باشد، نمی‌توان Krum اعمال کرد
        if len(filtered_results) < 2:
            print("⚠️ Not enough valid clients for Krum. Falling back to FedAvg.")
            return super().aggregate_fit(server_round, filtered_results, failures)

        # -------------------- مرحله 2: اعمال Krum روی کلاینت‌های مجاز --------------------
        # استخراج وزن‌ها و تبدیل به بردار تخت (flatten)
        weights = []
        for _, fit_res in filtered_results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([arr.flatten() for arr in ndarrays])
            weights.append(flat)

        n = len(weights)
        m = self.num_malicious
        k = n - m - 2  # تعداد همسایه‌هایی که باید در نظر گرفته شوند
        if k <= 0:
            k = 1  # حداقل یک همسایه

        # محاسبه ماتریس فواصل اقلیدسی
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(weights[i] - weights[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # محاسبه امتیاز Krum برای هر کلاینت
        scores = []
        for i in range(n):
            sorted_indices = np.argsort(distances[i])
            neighbor_indices = sorted_indices[1:k+1]  # نزدیک‌ترین k همسایه (به جز خودش)
            total_dist = np.sum(distances[i, neighbor_indices])
            scores.append(total_dist)

        # انتخاب کلاینتی با کمترین امتیاز
        best_idx = np.argmin(scores)
        best_client, best_fit_res = filtered_results[best_idx]
        print(f"🏆 Krum selected client {best_fit_res.metrics.get('eth_address', '')}")

        return best_fit_res.parameters, {}