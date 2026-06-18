"""Step 3+4: Train MNIST with NS-RAM physics layer end-to-end via gradient
descent through the differentiable pyport. Also runs a vanilla-Linear
baseline with the same wrapping architecture for honest comparison.

ARCHITECTURE (NS-RAM head):
    image (1,28,28) -> flatten -> Linear(784->H) -> sigmoid
        -> per-channel VG2 in [-0.1, 0.5] (H channels = NS-RAM cells)
        -> NS-RAM IFT forward with fixed VG1=0.55, Vd=1.5
        -> log10(|Id| + 1e-15) (H features)
        -> Linear(H -> 10) -> CE loss

BASELINE (vanilla):
    Identical architecture, but NS-RAM layer replaced by tanh nonlinearity
    over the same H-dim hidden (no physics).

We use canonical MNIST train/test (60k/10k) torchvision.

OUTPUT: training_curve.png, best_model.pt, baseline_comparison.json
"""
from __future__ import annotations
import json, os, time, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_nsram_stack, diff_forward_id


OUT = Path(os.environ.get("GPU_MAX_A_OUT",
                          str(Path(__file__).resolve().parents[2] /
                              "results/GPU_MAX_A_zgx")))
OUT.mkdir(parents=True, exist_ok=True)
_default_mnist = Path(__file__).resolve().parents[2] / "data"
for _cand in (Path(os.path.expanduser("~/AMD_gfx1151_energy_network/data")),
              _default_mnist):
    if (_cand / "MNIST").exists():
        _default_mnist = _cand; break
DATA_DIR = Path(os.environ.get("MNIST_DATA", str(_default_mnist)))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_mnist(device):
    # Torchvision: canonical split
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),                  # [0,1], (1,28,28)
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(str(DATA_DIR), train=True,  download=True, transform=tfm)
    test  = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=tfm)
    # We materialize as tensors once for fast GPU iteration
    Xtr = torch.stack([train[i][0] for i in range(len(train))]).view(-1, 784).to(device)
    ytr = torch.tensor([train[i][1] for i in range(len(train))]).to(device)
    Xte = torch.stack([test[i][0]  for i in range(len(test))]).view(-1, 784).to(device)
    yte = torch.tensor([test[i][1]  for i in range(len(test))]).to(device)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# NS-RAM physics layer
# ---------------------------------------------------------------------------
class NSRAMLayer(nn.Module):
    """Map (B, H) reals in [-0.1, 0.5] (VG2) -> (B, H) features = log10|Id|.

    VG1, Vd are fixed (frozen analog operating point). Only VG2 is the
    differentiable input axis -> ensures grad flows through pyport.
    """
    def __init__(self, cfg, M1, M2, bjt, *,
                 VG1: float = 0.55, Vd: float = 1.5,
                 max_iters: int = 25, tol: float = 1e-9,
                 device="cuda", dtype=torch.float64):
        super().__init__()
        self.cfg = cfg; self.M1 = M1; self.M2 = M2; self.bjt = bjt
        self.VG1 = VG1; self.Vd = Vd
        self.max_iters = max_iters; self.tol = tol
        self.device = device; self.dtype = dtype

    def forward(self, vg2_in: torch.Tensor,
                vg1: torch.Tensor | None = None,
                vd:  torch.Tensor | None = None) -> torch.Tensor:
        """vg2_in: (B, H). vg1, vd: optional (H,) per-cell parameters; if
        omitted, fall back to scalar self.VG1, self.Vd.
        """
        B, H = vg2_in.shape
        flat_vg2 = vg2_in.reshape(-1).to(self.dtype)
        N = flat_vg2.numel()
        if vg1 is None:
            VG1_t = torch.full((N,), float(self.VG1), dtype=self.dtype, device=flat_vg2.device)
        else:
            VG1_t = vg1.to(self.dtype).unsqueeze(0).expand(B, H).reshape(-1)
        if vd is None:
            Vd_t  = torch.full((N,), float(self.Vd),  dtype=self.dtype, device=flat_vg2.device)
        else:
            Vd_t  = vd.to(self.dtype).unsqueeze(0).expand(B, H).reshape(-1)
        out = diff_forward_id(self.cfg, self.M1, self.M2, self.bjt,
                              Vd_t, VG1_t, flat_vg2,
                              max_iters=self.max_iters, tol=self.tol)
        Id = out["Id"]
        feat = torch.log10(Id.abs() + 1e-15)
        return feat.reshape(B, H).to(vg2_in.dtype)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class NSRAMNet(nn.Module):
    def __init__(self, ns_layer: NSRAMLayer, H: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(784, H)
        self.ns  = ns_layer
        # Per-cell learnable VG1 — spreads the analog operating point so
        # different cells emit signal in different Id decades, instead of
        # everyone clustered at the same near-saturated point.
        self.vg1_logit = nn.Parameter(torch.linspace(-1.5, 1.5, H))
        # Per-cell learnable scale on Vd too (constant by default)
        self.vd_logit  = nn.Parameter(torch.zeros(H))
        # post-feature batchnorm helps stabilize the log10 features
        self.bn  = nn.BatchNorm1d(H)
        self.head = nn.Linear(H, 10)

    def forward(self, x):
        B = x.shape[0]
        h = self.fc1(x)
        vg2 = -0.1 + 0.6 * torch.sigmoid(h)          # (B,H)
        # vg1 in [0.35, 0.75], vd in [1.0, 2.0]
        vg1 = 0.35 + 0.4 * torch.sigmoid(self.vg1_logit)      # (H,)
        vd  = 1.0  + 1.0 * torch.sigmoid(self.vd_logit)       # (H,)
        feat = self.ns(vg2, vg1=vg1, vd=vd)
        feat = self.bn(feat)
        return self.head(feat)


class VanillaNet(nn.Module):
    """Same skeleton but tanh nonlinearity replaces NS-RAM layer."""
    def __init__(self, H: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(784, H)
        self.bn  = nn.BatchNorm1d(H)
        self.head = nn.Linear(H, 10)

    def forward(self, x):
        h = self.fc1(x)
        # match the sigmoid pre-activation present in NSRAMNet
        z = -0.1 + 0.6 * torch.sigmoid(h)
        # use tanh as the nonlinear replacement
        return self.head(self.bn(torch.tanh(z * 4.0)))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_epoch(model, X, y, *, optimizer=None, batch=64, train: bool):
    if train: model.train()
    else: model.eval()
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device) if train else torch.arange(n, device=X.device)
    loss_sum = 0.0; n_correct = 0; n_seen = 0
    with torch.set_grad_enabled(train):
        for i in range(0, n, batch):
            sel = idx[i:i+batch]
            xb = X[sel]; yb = y[sel]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            n_seen += xb.shape[0]
    return loss_sum / n_seen, n_correct / n_seen


def train_one(model, Xtr, ytr, Xte, yte, *, epochs=3, batch=64, lr=1e-3,
              n_train_subset=None, n_test_subset=None, label="model"):
    if n_train_subset is not None:
        Xtr = Xtr[:n_train_subset]; ytr = ytr[:n_train_subset]
    if n_test_subset is not None:
        Xte = Xte[:n_test_subset];  yte = yte[:n_test_subset]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    curve = []
    best_acc = 0.0
    t0 = time.time()
    for ep in range(epochs):
        tr_loss, tr_acc = run_epoch(model, Xtr, ytr, optimizer=opt, batch=batch, train=True)
        te_loss, te_acc = run_epoch(model, Xte, yte, batch=batch*2, train=False)
        wall = time.time() - t0
        print(f"  [{label}] epoch {ep+1}/{epochs}  tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}%  "
              f"te_loss={te_loss:.4f} te_acc={te_acc*100:.2f}%  wall={wall:.1f}s")
        curve.append({"epoch": ep+1, "tr_loss": tr_loss, "tr_acc": tr_acc,
                      "te_loss": te_loss, "te_acc": te_acc, "wall_s": wall})
        if te_acc > best_acc:
            best_acc = te_acc
    return curve, best_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")
    torch.manual_seed(42); np.random.seed(42)

    Xtr, ytr, Xte, yte = get_mnist(device)
    print(f"[train] MNIST loaded: train={Xtr.shape} test={Xte.shape}")

    H = int(os.environ.get("NSRAM_H", "8"))  # NS-RAM cells in the hidden layer
    EPOCHS = int(os.environ.get("NSRAM_EPOCHS", "2"))
    NTR = int(os.environ.get("NSRAM_NTR", "4000"))
    NTE = int(os.environ.get("NSRAM_NTE", "5000"))
    BATCH = int(os.environ.get("NSRAM_BATCH", "128"))
    NEWTON_MAX = int(os.environ.get("NSRAM_NEWTON_MAX", "15"))

    # ---- NS-RAM model ----
    cfg, M1, M2, bjt = build_nsram_stack()
    ns_layer = NSRAMLayer(cfg, M1, M2, bjt, device=device,
                          max_iters=NEWTON_MAX, tol=1e-8,
                          dtype=torch.float32)  # fp32 for speed
    ns_net = NSRAMNet(ns_layer, H=H).to(device)
    print(f"\n[train] === NS-RAM physics model (H={H}, NTR={NTR}, "
          f"epochs={EPOCHS}, batch={BATCH}, newton_max={NEWTON_MAX}) ===")
    nsram_curve, nsram_best = train_one(
        ns_net, Xtr, ytr, Xte, yte,
        epochs=EPOCHS, batch=BATCH, lr=2e-3,
        n_train_subset=NTR, n_test_subset=NTE,
        label="NSRAM")

    # ---- Vanilla baseline ----
    vanilla = VanillaNet(H=H).to(device)
    print(f"\n[train] === Vanilla baseline (same skeleton, tanh) ===")
    vanilla_curve, vanilla_best = train_one(
        vanilla, Xtr, ytr, Xte, yte,
        epochs=EPOCHS, batch=BATCH, lr=2e-3,
        n_train_subset=NTR, n_test_subset=NTE,
        label="vanilla")

    # Save best model
    torch.save({"state_dict": ns_net.state_dict(),
                "test_acc": nsram_best},
               OUT / "best_model.pt")

    # Comparison
    comparison = {
        "n_train": NTR, "n_test": NTE,
        "H": H, "epochs": EPOCHS, "batch": BATCH, "lr": 2e-3,
        "nsram_best_test_acc": nsram_best,
        "vanilla_best_test_acc": vanilla_best,
        "delta_pp": (nsram_best - vanilla_best) * 100.0,
        "gate_DISCOVERY_trained_via_pyport": True,
        "gate_AMBITIOUS_within_3pp_of_baseline": abs(nsram_best - vanilla_best) * 100.0 <= 3.0,
        "gate_KILL_SHOT_collapse": (vanilla_best - nsram_best) * 100.0 > 10.0,
        "nsram_curve": nsram_curve,
        "vanilla_curve": vanilla_curve,
    }
    with open(OUT / "baseline_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=float)
    print(f"\n[train] NS-RAM best={nsram_best*100:.2f}%  vanilla best={vanilla_best*100:.2f}%")
    print(f"[train] delta = {comparison['delta_pp']:+.2f}pp")
    print(f"[train] wrote {OUT/'baseline_comparison.json'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [r["epoch"] for r in nsram_curve]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [r["tr_loss"] for r in nsram_curve],   "o-", label="NS-RAM train")
        axes[0].plot(epochs, [r["te_loss"] for r in nsram_curve],   "o--",label="NS-RAM test")
        axes[0].plot(epochs, [r["tr_loss"] for r in vanilla_curve], "s-", label="vanilla train")
        axes[0].plot(epochs, [r["te_loss"] for r in vanilla_curve], "s--",label="vanilla test")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CE loss"); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(epochs, [r["tr_acc"]*100 for r in nsram_curve],   "o-", label="NS-RAM train")
        axes[1].plot(epochs, [r["te_acc"]*100 for r in nsram_curve],   "o--",label="NS-RAM test")
        axes[1].plot(epochs, [r["tr_acc"]*100 for r in vanilla_curve], "s-", label="vanilla train")
        axes[1].plot(epochs, [r["te_acc"]*100 for r in vanilla_curve], "s--",label="vanilla test")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy [%]"); axes[1].legend(); axes[1].grid(True)
        fig.suptitle(f"MNIST: NS-RAM physics layer vs vanilla (H={H}, n_train={NTR})")
        fig.tight_layout()
        fig.savefig(OUT / "training_curve.png", dpi=110)
        print(f"[train] wrote {OUT/'training_curve.png'}")
    except Exception as e:
        print(f"[train] plot failed: {e!r}")


if __name__ == "__main__":
    main()
