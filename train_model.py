import numpy as np
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


print("Downloading/loading MNIST dataset from OpenML...")
print("First run requires an internet connection. This may take a few minutes.")

# Download MNIST
mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="liac-arff"
)

# Convert pixel values to float and scale to 0-1
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

print("MNIST dataset loaded.")
print("Total samples:", len(X))

# Use a smaller subset for faster training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=30000,
    test_size=5000,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training model...")

# Neural network model
model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=30,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=True
        )
    )
])

# Train
model.fit(X_train, y_train)

print("Model training completed.")

# Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print()
print("=" * 50)
print(f"MNIST Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)

print()
print("Classification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "digit_model.joblib")

print()
print("=" * 50)
print("Model saved as digit_model.joblib")
print("=" * 50)