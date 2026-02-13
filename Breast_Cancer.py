import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import sys

# دریافت آدرس اتریوم از آرگومان خط فرمان
ETH_ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "0x..."  # یک آدرس پیش‌فرض برای تست

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CancerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        model = LogisticRegression(max_iter=2000)
        model.fit(np.zeros((2, 30)), np.array([0, 1]))
        return [model.coef_, model.intercept_]

    def fit(self, parameters, config):
        model = LogisticRegression(max_iter=2000)
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)
        model.fit(X_train_noisy, y_train)
        accuracy = float(model.score(X_train, y_train))
        print(f"-> Honest client training completed. Accuracy: {accuracy:.4f}")
        return [model.coef_, model.intercept_], len(X_train), {
            "accuracy": accuracy,
            "eth_address": ETH_ADDRESS   # <-- ارسال آدرس
        }

    def evaluate(self, parameters, config):
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train[:2], y_train[:2])
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        accuracy = float(accuracy_score(y_test, model.predict(X_test)))
        return 0.0, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__":
    print(f"🚀 Starting Honest Federated Learning Client with address {ETH_ADDRESS}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=CancerClient()
    )