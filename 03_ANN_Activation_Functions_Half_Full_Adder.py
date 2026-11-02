"""
EXPERIMENT 3: Activation Functions and ANN
Train a small feed-forward ANN for Half Adder and Full Adder truth tables.
Uses NumPy only for the neural-network training.
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def tanh_fn(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# Show activation function outputs
sample = np.array([-2, -1, 0, 1, 2], dtype=float)
print("x:         ", sample)
print("Sigmoid:   ", sigmoid(sample).round(3))
print("Tanh:      ", tanh_fn(sample).round(3))
print("ReLU:      ", relu(sample).round(3))
print("Leaky ReLU:", leaky_relu(sample).round(3))
print("Softmax:   ", softmax(sample.reshape(1, -1)).round(3))

# Half Adder: inputs A,B -> outputs Sum,Carry
X_half = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
Y_half = np.array([[0,0], [1,0], [1,0], [0,1]], dtype=float)

# Full Adder: inputs A,B,Cin -> outputs Sum,Carry
X_full = np.array([[a,b,c] for a in [0,1] for b in [0,1] for c in [0,1]], dtype=float)
Y_full = np.array([[int(int(a) ^ int(b) ^ int(c)), int((a*b) or (a*c) or (b*c))]
                   for a,b,c in X_full], dtype=float)

def train_ann(X, Y, epochs=10000, lr=1.0):
    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 1, (X.shape[1], 8))
    b1 = np.zeros((1, 8))
    W2 = rng.normal(0, 1, (8, Y.shape[1]))
    b2 = np.zeros((1, Y.shape[1]))

    for _ in range(epochs):
        # Forward propagation
        hidden = sigmoid(X @ W1 + b1)
        output = sigmoid(hidden @ W2 + b2)

        # Backpropagation
        error = output - Y
        d2 = error * output * (1 - output)
        d1 = (d2 @ W2.T) * hidden * (1 - hidden)

        W2 -= lr * (hidden.T @ d2) / len(X)
        b2 -= lr * np.mean(d2, axis=0, keepdims=True)
        W1 -= lr * (X.T @ d1) / len(X)
        b1 -= lr * np.mean(d1, axis=0, keepdims=True)

    predictions = (sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2) >= 0.5).astype(int)
    return predictions

print("\nHALF ADDER (A, B, Sum, Carry)")
print(np.column_stack((X_half.astype(int), Y_half.astype(int), train_ann(X_half, Y_half))))
print("\nFULL ADDER (A, B, Cin, Sum, Carry)")
print(np.column_stack((X_full.astype(int), Y_full.astype(int), train_ann(X_full, Y_full))))
