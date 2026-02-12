import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Data Loading
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MaliciousClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        model = LogisticRegression(max_iter=2000)
        model.fit(np.zeros((2, 30)), np.array([0, 1]))
        return [model.coef_, model.intercept_]

    def fit(self, parameters, config):
        model = LogisticRegression(max_iter=2000)
        
        # --- Label Flipping Attack Simulation ---
        y_train_poisoned = y_train.copy()
        # Flip 50% of labels to deceive the global model
        indices = np.random.choice(len(y_train), size=int(len(y_train) * 0.5), replace=False)
        y_train_poisoned[indices] = 1 - y_train_poisoned[indices]
        
        model.fit(X_train, y_train_poisoned)
        
        # Evaluate on clean data to show the drop in performance
        accuracy = float(model.score(X_test, y_test))
        print(f"⚠️ [Malicious Node] Poisoned training complete. Reported Accuracy: {accuracy:.4f}")
        return [model.coef_, model.intercept_], len(X_train), {"accuracy": accuracy}

    def evaluate(self, parameters, config):
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train[:2], y_train[:2])
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        
        accuracy = float(model.score(X_test, y_test))
        return 0.0, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    print("🚨 Starting Malicious Federated Learning Client...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=MaliciousClient()
    )