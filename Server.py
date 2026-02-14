import flwr as fl
import numpy as np
from web3 import Web3
import json
import os
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, Scalar, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

load_dotenv()

# --- تنظیمات اتصال به بلاکچین (مشابه قبل) ---
NETWORK = os.getenv("NETWORK", "ganache")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
w3 = Web3(Web3.HTTPProvider(os.getenv("SEPOLIA_RPC") if NETWORK == "sepolia" else "http://localhost:7545"))
SERVER_ACCOUNT = w3.eth.account.from_key(PRIVATE_KEY).address if NETWORK == "sepolia" else w3.eth.accounts[0]

with open("abi.json", "r") as f:
    CONTRACT_ABI = json.load(f)
contract = w3.eth.contract(address=os.getenv("CONTRACT_ADDRESS"), abi=CONTRACT_ABI)

current_nonce = w3.eth.get_transaction_count(SERVER_ACCOUNT)

class ReputationWeightedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        global current_nonce
        valid_results = []
        reputations = []

        print(f"\n--- Aggregation Round {server_round} ---")

        for client_proxy, fit_res in results:
            eth_address = fit_res.metrics.get("eth_address")
            accuracy = fit_res.metrics.get("accuracy", 0)
            score = int(accuracy * 100)

            if not eth_address:
                continue

            # ۱. ارسال امتیاز به بلاکچین برای آپدیت اعتبار
            try:
                tx = contract.functions.updateScore(eth_address, score).build_transaction({
                    'from': SERVER_ACCOUNT,
                    'nonce': current_nonce,
                    'gas': 250000,
                    'chainId': 11155111 if NETWORK == "sepolia" else 5777
                })
                signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
                w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                current_nonce += 1
                print(f"✅ Score {score} recorded for {eth_address}")
            except Exception as e:
                print(f"⚠️ Blockchain Error: {e}")

            # ۲. دریافت اعتبار (Reputation) لحظه‌ای از بلاکچین
            rep = contract.functions.getReputation(eth_address).call()
            print(f"📊 Client {eth_address} Reputation: {rep}")

            # فیلتر نودهایی که اعتبار بسیار ناچیزی دارند (مثلاً زیر ۵٪)
            if rep > 5:
                valid_results.append(parameters_to_ndarrays(fit_res.parameters))
                reputations.append(rep)

        if not valid_results:
            return None, {}

        # ۳. میانگین‌گیری وزنی (Mathematical Weighted Average)
        # فرمول: Sum(Weight_i * Reputation_i) / Sum(Reputations)
        total_reputation = sum(reputations)
        
        # محاسبه وزن هر لایه مدل
        weighted_weights = [
            [layer * (rep / total_reputation) for layer in layers]
            for layers, rep in zip(valid_results, reputations)
        ]

        # جمع‌بندی لایه‌ها
        aggregated_ndarrays = [
            sum(layer_updates) for layer_updates in zip(*weighted_weights)
        ]

        print(f"✨ Aggregation complete using {len(valid_results)} clients.")
        return ndarrays_to_parameters(aggregated_ndarrays), {}

# --- اجرای سرور ---
if __name__ == "__main__":
    strategy = ReputationWeightedFedAvg(
        min_fit_clients=1,
        min_available_clients=1,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )