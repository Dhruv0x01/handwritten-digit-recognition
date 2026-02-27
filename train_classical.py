from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.datasets import mnist
import joblib
import time

from utils.preprocess import preprocess_for_classical

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Original training shape:", x_train.shape)
print("Original testing shape:", x_test.shape)

# Preprocess (Modular) (Normalizing and Flattening)
x_train = preprocess_for_classical(x_train)
x_test = preprocess_for_classical(x_test)

print("Processed training shape:", x_train.shape)
print("Processed testing shape:", x_test.shape)

# Define Model (Loop Structure)
models = {
    "logistic": LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ),
    "svm": SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ),
    "knn": KNeighborsClassifier(
        n_neighbors=3,
        n_jobs=-1
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=20,
        random_state=42
    )
}

results = {}

# Train & Evaluate Loop
for name, model in models.items():
    print(f"\n==== Training {name.upper()} ====")

    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        "accuracy": accuracy,
        "time": training_time
    }

    print(f"{name.upper()} Accuracy: {accuracy:.4f}")
    print(f"{name.upper()} Training Time: {training_time:.2f} seconds")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name.upper()} Confusion Matrix:\n", cm)

    # Save Model
    joblib.dump(model, f"models/{name}.pkl")
    print(f"\nModel saved as models/{name}.pkl")

# Final Comparison
print("\n==== Final Classical Model Comparison ====")

for name, info in results.items():
    print(
        f"{name.upper()} | "
        f"Accuracy: {info['accuracy']:.4f} | "
        f"Time: {info['time']:.2f}s"
    )

