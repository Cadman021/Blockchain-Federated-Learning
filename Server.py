import flwr as fl
from web3 import Web3
import json
import os
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy

# ------------------------------------------------------------
# ۱. بارگذاری تنظیمات
# ------------------------------------------------------------
load_dotenv()

NETWORK = os.getenv("NETWORK", "ganache")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
SEPOLIA_RPC = os.getenv("SEPOLIA_RPC", "https://sepolia.infura.io/v3/YOUR_INFURA_ID")

current_nonce = None

if NETWORK == "ganache":
    GANACHE_URL = "http://localhost:7545"
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    SERVER_ACCOUNT = w3.eth.accounts[0]
    CHAIN_ID = 5777 
    print(f"✅ Connected to Ganache. Account: {SERVER_ACCOUNT}")
else:
    w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC))
    if not PRIVATE_KEY:
        raise ValueError("PRIVATE_KEY must be set for Sepolia")
    account = w3.eth.account.from_key(PRIVATE_KEY)
    SERVER_ACCOUNT = account.address
    CHAIN_ID = 11155111 
    print(f"✅ Connected to Sepolia. Server Account: {SERVER_ACCOUNT}")

if not w3.is_connected():
    raise Exception("Failed to connect to the Ethereum node")

# ------------------------------------------------------------
# ۲. تنظیمات قرارداد هوشمند
# ------------------------------------------------------------
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
with open("abi.json", "r") as f:
    CONTRACT_ABI = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

# ------------------------------------------------------------
# ۳. استراتژی سفارشی
# ------------------------------------------------------------
class BlockchainFilteredFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        global current_nonce
        
        if not results:
            return None, {}

        # گرفتن Nonce اولیه
        if current_nonce is None:
            current_nonce = w3.eth.get_transaction_count(SERVER_ACCOUNT)

        filtered_results = []

        for client_proxy, fit_res in results:
            eth_address = fit_res.metrics.get("eth_address")
            accuracy = fit_res.metrics.get("accuracy", 0)
            score = int(accuracy * 100)

            if not eth_address:
                continue

            try:
                print(f"🚀 Round {server_round}: Recording score {score}% for {eth_address}...")
                
                latest_block = w3.eth.get_block('latest')
                base_fee = latest_block.get('baseFeePerGas', w3.to_wei(20, 'gwei'))
                
                try:
                    priority_fee = w3.eth.max_priority_fee if hasattr(w3.eth, 'max_priority_fee') else w3.to_wei(2, 'gwei')
                except:
                    priority_fee = w3.to_wei(2, 'gwei')

                tx = contract.functions.updateScore(eth_address, score).build_transaction({
                    'from': SERVER_ACCOUNT,
                    'nonce': current_nonce,
                    'gas': 200000,
                    'maxFeePerGas': int(base_fee * 2 + priority_fee),
                    'maxPriorityFeePerGas': priority_fee,
                    'chainId': CHAIN_ID
                })

                # --- اصلاح بخش امضا و ارسال (سازگار با همه نسخه‌ها) ---
                signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
                
                # استفاده ازgetattr برای پیدا کردن نام صحیح ویژگی (raw_transaction یا rawTransaction)
                raw_tx = getattr(signed_tx, 'raw_transaction', getattr(signed_tx, 'rawTransaction', None))
                
                if raw_tx is None:
                    raise AttributeError("Could not find raw transaction attribute in signed object")

                tx_hash = w3.eth.send_raw_transaction(raw_tx)
                
                print(f"⏳ Waiting for Sepolia confirmation (TX: {tx_hash.hex()})...")
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300) # افزایش زمان به ۵ دقیقه
                
                if receipt.status == 1:
                    print(f"✨ Success! Score recorded in block {receipt.blockNumber}")
                    current_nonce += 1
                else:
                    print(f"❌ Transaction reverted on-chain.")
                    current_nonce += 1 # نانس مصرف شده، پس باید اضافه شود

            except Exception as e:
                print(f"⚠️ Transaction Error: {e}")
                # ریست کردن نانس از شبکه برای اطمینان از همگام بودن
                current_nonce = w3.eth.get_transaction_count(SERVER_ACCOUNT)
                continue

            # چک کردن بلک‌لیست
            try:
                is_blacklisted = contract.functions.isBlacklisted(eth_address).call()
                if is_blacklisted:
                    print(f"🚫 Client {eth_address} is blacklisted.")
                    continue
            except:
                pass

            filtered_results.append((client_proxy, fit_res))

        if not filtered_results:
            print("❌ No valid clients for aggregation after blockchain check.")
            return None, {}

        return super().aggregate_fit(server_round, filtered_results, failures)

# ------------------------------------------------------------
# ۴. اجرای سرور
# ------------------------------------------------------------
if __name__ == "__main__":
    strategy = BlockchainFilteredFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=1,
    )

    print("📡 Server starting on Sepolia Testnet...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )