# Blockchain-Integrated Federated Learning for Secure Medical Data Analysis

This repository contains a cutting-edge implementation of **Federated Learning (FL)** integrated with **Ethereum Blockchain** to ensure the integrity and traceability of AI model training. Specifically, it focuses on **Breast Cancer Classification** using the UCI dataset.

## 🌟 Project Overview
In traditional Federated Learning, a central server aggregates models from various clients (e.g., hospitals). However, there's often a lack of trust and a risk of **Data Poisoning**. This project solves these issues by:
1. **Decentralized Training:** Using **Flower (FLWR)** to train models locally without sharing sensitive patient data.
2. **Blockchain Auditing:** Using **Smart Contracts** on a private Ethereum network (**Ganache**) to record the accuracy of each participant in every round.
3. **Malicious Node Resilience:** Proving that the system can track and identify nodes that attempt to sabotage the global model (Label Flipping Attacks).

## 🚀 Key Experimental Results
Our experiments demonstrate a clear distinction between honest and malicious participants, as recorded on the immutable ledger:

| Participant Type | Best Accuracy | Blockchain Status | Detection |
|------------------|---------------|-------------------|-----------|
| **Honest Hospital** | ~96%          | ✅ Verified Score  | Trusted   |
| **Malicious Node** | ~42%          | ⚠️ Logged Fraud    | Flagged   |

> **Note:** The 42% accuracy drop was achieved by simulating a **Label Flipping Attack**, where 50% of the training labels were intentionally corrupted.

## 🛠 Tech Stack
* **AI/ML:** Python, Scikit-Learn, NumPy
* **Federated Learning:** Flower (FLWR)
* **Blockchain:** Solidity, Web3.py, Ganache (Ethereum Testnet)
* **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)

## 📂 Repository Structure
* `Server.py`: The FL server that aggregates weights and interacts with the Smart Contract.
* `Breast_Cancer.py`: The honest client (Hospital) performing clean training.
* `Malicious_Client.py`: The adversarial client performing Data Poisoning.
* `contract.sol`: The Solidity Smart Contract for reputation management.
* `abi.json`: The Application Binary Interface for blockchain communication.
* `/screenshots`: Visual proof of transactions and training logs.

## 🔧 How to Run
1. **Deploy the Contract:** Open Remix IDE, connect to Ganache, and deploy `contract.sol`.
2. **Setup:** Copy the Deployed Address and ABI into `Server.py`.
3. **Start Server:** Run `python Server.py`.
4. **Start Clients:** Run `python Breast_Cancer.py` and `python Malicious_Client.py` in separate terminals.