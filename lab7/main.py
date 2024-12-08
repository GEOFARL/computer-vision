import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

input_size = 28 * 28
hidden_size = 64
output_size = 4
learning_rate = 0.01
epochs = 1000
batch_size = 64
early_stopping_patience = 10

np.random.seed(42)
W1 = np.random.rand(input_size, hidden_size) - 0.5
b1 = np.random.rand(hidden_size) - 0.5
W2 = np.random.rand(hidden_size, output_size) - 0.5
b2 = np.random.rand(output_size) - 0.5

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return x > 0

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def load_and_preprocess_data():
    """
    Load MNIST dataset, filter digits 0, 1, 2, 3, and preprocess for binary classification.
    """
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(int)

    filter_mask = y < 4
    X, y = X[filter_mask], y[filter_mask]

    X = (X / 255.0 > 0.5).astype(float)

    y_encoded = np.eye(output_size)[y]

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    X_train, y_train = X_train[:5000], y_train[:5000]
    X_test, y_test = X_test[:1000], y_test[:1000]

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def forward_pass(X):
    global W1, b1, W2, b2
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backward_pass(X, y, z1, a1, a2):
    global W1, b1, W2, b2
    dz2 = a2 - y
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    dz1 = (dz2 @ W2.T) * relu_derivative(z1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

def train_model(X_train, y_train):
    global W1, b1, W2, b2
    loss_history = []
    patience_counter = 0
    min_loss = float('inf')

    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            z1, a1, z2, a2 = forward_pass(X_batch)

            epsilon = 1e-12
            a2 = np.clip(a2, epsilon, 1.0 - epsilon)
            loss = -np.sum(y_batch * np.log(a2)) / len(y_batch)

            backward_pass(X_batch, y_batch, z1, a1, a2)

        loss_history.append(loss)

        if loss < min_loss:
            min_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Stopping early at epoch {epoch}")
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss_history)), loss_history, label="Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def predict(X_test):
    _, _, _, a2 = forward_pass(X_test)
    return np.argmax(a2, axis=1)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    train_model(X_train, y_train)

    predictions = predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    for i in range(5):
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Predicted: {predictions[i]}, Actual: {y_test_labels[i]}")
        plt.axis("off")
        plt.show()
