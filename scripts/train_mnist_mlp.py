#!/usr/bin/env python3
"""Train a simple MLP on MNIST, export weights as raw binary for HIP kernel."""
import struct, os, sys
import numpy as np

# Load MNIST from raw files
def load_mnist(img_path, lbl_path):
    with open(img_path, 'rb') as f:
        f.read(16)  # skip header
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0
    with open(lbl_path, 'rb') as f:
        f.read(8)  # skip header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/MNIST/raw'
X_train, y_train = load_mnist(f'{base}/train-images-idx3-ubyte', f'{base}/train-labels-idx1-ubyte')
X_test, y_test = load_mnist(f'{base}/t10k-images-idx3-ubyte', f'{base}/t10k-labels-idx1-ubyte')
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Simple MLP: 784 → 128 → 64 → 10
np.random.seed(42)

def xavier(fan_in, fan_out):
    s = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-s, s, (fan_out, fan_in)).astype(np.float32)

W1 = xavier(784, 128)
b1 = np.zeros(128, dtype=np.float32)
W2 = xavier(128, 64)
b2 = np.zeros(64, dtype=np.float32)
W3 = xavier(64, 10)
b3 = np.zeros(10, dtype=np.float32)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward(X):
    h1 = relu(X @ W1.T + b1)
    h2 = relu(h1 @ W2.T + b2)
    out = h2 @ W3.T + b3
    return h1, h2, out

def cross_entropy(probs, labels):
    n = len(labels)
    return -np.log(probs[np.arange(n), labels] + 1e-10).mean()

# SGD training
lr = 0.01
batch_size = 128
n_epochs = 15

for epoch in range(n_epochs):
    # Shuffle
    idx = np.random.permutation(len(X_train))
    X_shuf, y_shuf = X_train[idx], y_train[idx]

    epoch_loss = 0
    n_batches = 0

    for i in range(0, len(X_train), batch_size):
        Xb = X_shuf[i:i+batch_size]
        yb = y_shuf[i:i+batch_size]
        n = len(Xb)

        # Forward
        h1 = relu(Xb @ W1.T + b1)
        h2 = relu(h1 @ W2.T + b2)
        logits = h2 @ W3.T + b3
        probs = softmax(logits)

        # Loss
        epoch_loss += cross_entropy(probs, yb)
        n_batches += 1

        # Backward (manual gradients)
        # dL/dlogits
        dlogits = probs.copy()
        dlogits[np.arange(n), yb] -= 1
        dlogits /= n

        # Layer 3
        dW3 = dlogits.T @ h2
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ W3

        # ReLU
        dh2 = dh2 * (h2 > 0)

        # Layer 2
        dW2 = dh2.T @ h1
        db2 = dh2.sum(axis=0)
        dh1 = dh2 @ W2

        # ReLU
        dh1 = dh1 * (h1 > 0)

        # Layer 1
        dW1 = dh1.T @ Xb
        db1 = dh1.sum(axis=0)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

    # Test accuracy
    _, _, test_logits = forward(X_test)
    test_preds = test_logits.argmax(axis=1)
    test_acc = (test_preds == y_test).mean() * 100

    print(f"Epoch {epoch+1:2d}: loss={epoch_loss/n_batches:.4f} test_acc={test_acc:.1f}%")

# Final test
_, _, test_logits = forward(X_test)
test_preds = test_logits.argmax(axis=1)
final_acc = (test_preds == y_test).mean() * 100
print(f"\nFinal accuracy: {final_acc:.1f}%")

# Save weights as raw binary (row-major float32)
out_dir = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/models/mnist_mlp'
os.makedirs(out_dir, exist_ok=True)

for name, arr in [('w1', W1), ('b1', b1), ('w2', W2), ('b2', b2), ('w3', W3), ('b3', b3)]:
    path = f'{out_dir}/{name}.bin'
    arr.tofile(path)
    print(f"Saved {name}: {arr.shape} → {path}")

print(f"\nWeights saved to {out_dir}/")
