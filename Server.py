import flwr as fl
from web3 import Web3
import json

# Blockchain Configuration
ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Contract Details (Update after deployment)
contract_address = "0x02769Ce76351011C75DF074C33456746d0743059" 
with open("abi.json") as f:
    contract_abi = json.load(f)

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

class BlockchainStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights, metrics = super().aggregate_fit(server_round, results, failures)
        
        if results:
            print(f"\n--- Round {server_round} Results ---")
            for _, fit_res in results:
                acc = fit_res.metrics["accuracy"]
                hospital_acc = int(acc * 100)
                
                # Transaction to record score on Blockchain
                tx_hash = contract.functions.updateScore(hospital_acc).transact({
                    'from': web3.eth.accounts[0]
                })
                print(f"✅ Score recorded on Blockchain! Accuracy: {hospital_acc}% | TX: {tx_hash.hex()}")
                
        return aggregated_weights, metrics

if __name__ == "__main__":
    print("🌐 Starting Federated Learning Server with Blockchain Audit...")
    
    strategy = BlockchainStrategy(
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=1,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )