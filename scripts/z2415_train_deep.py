#!/usr/bin/env python3
"""Train 3, 6, 8 layer MLPs on MNIST, export weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, struct

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

def load_mnist(img_path, lbl_path):
    with open(img_path, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0
    with open(lbl_path, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    return torch.tensor(images), torch.tensor(labels)

X_train, y_train = load_mnist(f'{base}/data/MNIST/raw/train-images-idx3-ubyte',
                               f'{base}/data/MNIST/raw/train-labels-idx1-ubyte')
X_test, y_test = load_mnist(f'{base}/data/MNIST/raw/t10k-images-idx3-ubyte',
                             f'{base}/data/MNIST/raw/t10k-labels-idx1-ubyte')

class DeepMLP(nn.Module):
    def __init__(self, depth, width=256):
        super().__init__()
        layers = []
        in_d = 784
        for i in range(depth - 1):
            layers.append(nn.Linear(in_d, width))
            layers.append(nn.ReLU())
            in_d = width
        layers.append(nn.Linear(in_d, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

for depth in [3, 6, 8]:
    print(f"\n{'='*50}")
    print(f"Training depth={depth} MLP (784→{'→'.join(['256']*(depth-1))}→10)")

    model = DeepMLP(depth, 256)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 15 if depth <= 6 else 20
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 256):
            batch = idx[i:i+256]
            loss = F.cross_entropy(model(X_train[batch]), y_train[batch])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item() * 100
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: {acc:.1f}%")

    print(f"  Final: {acc:.1f}%")

    # Export weights
    out_dir = f'{base}/models/mnist_mlp_d{depth}'
    os.makedirs(out_dir, exist_ok=True)

    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear_layers):
        w = layer.weight.detach().numpy()  # [out, in]
        b = layer.bias.detach().numpy()    # [out]
        w.tofile(f'{out_dir}/w{i}.bin')
        b.tofile(f'{out_dir}/b{i}.bin')
        print(f"  Saved w{i}={w.shape} b{i}={b.shape}")

    # Save metadata
    with open(f'{out_dir}/meta.txt', 'w') as f:
        f.write(f"depth={depth}\n")
        f.write(f"width=256\n")
        f.write(f"accuracy={acc:.2f}\n")
        f.write(f"n_layers={len(linear_layers)}\n")
        for i, layer in enumerate(linear_layers):
            f.write(f"layer{i}_in={layer.in_features}\n")
            f.write(f"layer{i}_out={layer.out_features}\n")

print("\nAll models trained and exported.")
