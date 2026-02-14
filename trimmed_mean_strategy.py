import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from web3 import Web3
import json

# -------------------- اتصال به بلاکچین (همانند قبل) --------------------
GANACHE_URL = "http://localhost:7545"
CONTRACT_ADDRESS = "0xFE8a84A40Cc2d4F133B0F80CadAd6Fb9A3f84790"  

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
assert w3.is_connected(), "Failed to connect to Ganache"

with open("abi.json", "r") as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
SERVER_ACCOUNT = w3.eth.accounts[0]

class BlockchainTrimmedMeanStrategy(fl.server.strategy.FedAvg):
    def __init__(self, trim_ratio=0.2, **kwargs):
        """
        trim_ratio: نسبت داده‌هایی که از دو طرف حذف می‌شوند (مثلاً 0.2 یعنی 20٪ کمترین و 20٪ بیشترین حذف).
        """
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio

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

        # اگر تعداد کلاینت‌های مجاز کمتر از 2 باشد، از FedAvg استفاده می‌کنیم
        if len(filtered_results) < 2:
            print("⚠️ Not enough valid clients for Trimmed Mean. Falling back to FedAvg.")
            return super().aggregate_fit(server_round, filtered_results, failures)

        # -------------------- مرحله 2: اعمال Trimmed Mean --------------------
        # تبدیل همه وزن‌ها به ndarray
        all_weights = []
        for _, fit_res in filtered_results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            all_weights.append(ndarrays)  # هر کدام یک لیست از آرایه‌های هر لایه

        n_clients = len(all_weights)
        n_layers = len(all_weights[0])

        # تعداد حذف از هر طرف: حداقل 1 را حذف می‌کنیم (در صورت امکان)
        trim_count = int(n_clients * self.trim_ratio)
        if trim_count < 1:
            trim_count = 1  # حداقل یک مقدار از هر طرف حذف شود

        # محاسبه trimmed mean برای هر لایه
        trimmed_weights = []
        for layer_idx in range(n_layers):
            # جمع‌آوری مقادیر این لایه از همه کلاینت‌ها
            layer_values = [w[layer_idx] for w in all_weights]  # لیستی از آرایه‌ها
            # تبدیل به آرایه‌ای با ابعاد (n_clients, ...)
            stacked = np.stack(layer_values, axis=0)

            # مرتب‌سازی در امتداد بعد اول (کلاینت‌ها)
            sorted_stack = np.sort(stacked, axis=0)

            # حذف trim_count از ابتدا و انتها
            trimmed = sorted_stack[trim_count : n_clients - trim_count]

            # میانگین در امتداد بعد کلاینت‌ها
            mean_layer = np.mean(trimmed, axis=0)
            trimmed_weights.append(mean_layer)

        # تبدیل به Parameters
        parameters_aggregated = ndarrays_to_parameters(trimmed_weights)

        return parameters_aggregated, {}