import flwr as fl
from web3 import Web3
import json
from typing import List, Tuple, Optional, Dict
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy

GANACHE_URL = "http://localhost:7545"
CONTRACT_ADDRESS = "0xeA6eb2c52D1fcd9834BafBCF4375d647B06094b1"  # Replace with your deployed contract address

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
assert w3.is_connected(), "Failed to connect to Ganache"

with open("abi.json", "r") as f:
    contract_abi = json.load(f)

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
SERVER_ACCOUNT = w3.eth.accounts[0]

print("Connected to Ganache. Contract address:", CONTRACT_ADDRESS)
print("Server account:", SERVER_ACCOUNT)

class BlockchainFilteredFedAvg(fl.server.strategy.FedAvg):
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
                print(f"Client {client_proxy.cid} did not send Ethereum address. Skipped.")
                continue

            try:
                eth_address = Web3.to_checksum_address(eth_address)
            except:
                print(f"Invalid address format: {eth_address}")
                continue

            accuracy = metrics.get("accuracy", 0.0)
            score = int(accuracy * 100)

            try:
                tx_hash = contract.functions.updateScore(
                    eth_address, score
                ).transact({
                    "from": SERVER_ACCOUNT,
                    "gas": 100000
                })
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=10)
                print(f"Score {score} recorded for {eth_address} (tx: {receipt.transactionHash.hex()[:10]}...)")
            except Exception as e:
                print(f"Failed to record score for {eth_address}: {e}")
                continue

            try:
                blacklisted = contract.functions.isBlacklisted(eth_address).call()
            except Exception as e:
                print(f"Failed to check blacklist for {eth_address}: {e}")
                blacklisted = False

            if blacklisted:
                print(f"Client {eth_address} is blacklisted. Update ignored.")
                continue

            filtered_results.append((client_proxy, fit_res))

        if not filtered_results:
            print("No valid clients for aggregation.")
            return None, {}

        return super().aggregate_fit(server_round, filtered_results, failures)

if __name__ == "__main__":
    print("Starting Federated Learning Server with Blockchain Audit & Auto-Blacklisting...")

    strategy = BlockchainFilteredFedAvg(
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