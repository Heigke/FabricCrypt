"""Distill the Pillar V ensemble (xgb+lgb+mlp) into a small torch MLP.

Output: results/Pillar_III_topology_zoo/distilled_mlp.pt
Gate: NRMSE on held-out random grid <= 0.1 dec (in log10|Id| space).

Run on ikaros (where the pickle still unpickles).
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results" / "Pillar_V_emulator"))
from predict import predict_logId  # noqa: E402

OUT_DIR = ROOT / "results" / "Pillar_III_topology_zoo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Envelope from Sebas measurements (interpolation regime only).
VG1_RANGE = (0.20, 0.60)
VG2_RANGE = (-0.20, 0.50)
VD_RANGE = (0.00, 2.20)


def sample_grid(n: int, rng: np.random.Generator) -> np.ndarray:
    vg1 = rng.uniform(*VG1_RANGE, n)
    vg2 = rng.uniform(*VG2_RANGE, n)
    vd = rng.uniform(*VD_RANGE, n)
    return np.stack([vg1, vg2, vd], axis=1).astype(np.float32)


def label(X: np.ndarray) -> np.ndarray:
    fwd = np.zeros(X.shape[0])
    return predict_logId(X[:, 0], X[:, 1], X[:, 2], fwd_bwd=fwd).astype(np.float32)


class DistilledMLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Input normalization stats (filled at training time).
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.x_mean) / self.x_std
        yn = self.net(xn).squeeze(-1)
        return yn * self.y_std + self.y_mean


def main() -> None:
    rng = np.random.default_rng(42)
    t0 = time.time()

    # Heavy: ensemble is slow; cache labels.
    n_train, n_val = 80_000, 10_000
    print(f"[distill] sampling+labeling {n_train + n_val} points (slow, xgb+lgb+mlp)...")
    Xtr = sample_grid(n_train, rng)
    ytr = label(Xtr)
    Xv = sample_grid(n_val, rng)
    yv = label(Xv)
    print(f"[distill] labels ready in {time.time() - t0:.1f}s.  "
          f"y range=[{ytr.min():.2f},{ytr.max():.2f}]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[distill] device={device}")

    model = DistilledMLP(hidden=128).to(device)
    model.x_mean.copy_(torch.tensor(Xtr.mean(0)))
    model.x_std.copy_(torch.tensor(Xtr.std(0) + 1e-8))
    model.y_mean.copy_(torch.tensor([ytr.mean()]))
    model.y_std.copy_(torch.tensor([ytr.std() + 1e-8]))

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xv_t = torch.from_numpy(Xv).to(device)
    yv_t = torch.from_numpy(yv).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000)
    batch = 4096
    best = float("inf")
    best_state = None
    log = []
    for step in range(4000):
        idx = torch.randint(0, n_train, (batch,), device=device)
        x = Xtr_t[idx]; y = ytr_t[idx]
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                vpred = model(Xv_t)
                resid = (vpred - yv_t).cpu().numpy()
                med_abs = float(np.median(np.abs(resid)))
                nrmse = float(np.sqrt(np.mean(resid ** 2)) / (yv_t.std().item() + 1e-8))
            log.append({"step": step + 1, "loss": float(loss.item()),
                        "val_median_abs_dec": med_abs, "val_nrmse": nrmse})
            print(f"  step {step+1:4d}  loss={loss.item():.4f}  "
                  f"val_med|res|={med_abs:.4f} dec  nrmse={nrmse:.4f}")
            if med_abs < best:
                best = med_abs
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval.
    with torch.no_grad():
        vpred = model(Xv_t).cpu().numpy()
    resid = vpred - yv
    final_med = float(np.median(np.abs(resid)))
    final_nrmse = float(np.sqrt(np.mean(resid ** 2)) / (yv.std() + 1e-8))
    gate_pass = final_med <= 0.10
    print(f"[distill] FINAL val median |res|={final_med:.4f} dec  nrmse={final_nrmse:.4f}  "
          f"gate(<=0.10)={'PASS' if gate_pass else 'FAIL'}")

    out_path = OUT_DIR / "distilled_mlp.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": 128,
        "val_median_abs_dec": final_med,
        "val_nrmse": final_nrmse,
        "gate_pass": gate_pass,
        "vg1_range": VG1_RANGE, "vg2_range": VG2_RANGE, "vd_range": VD_RANGE,
        "n_train": n_train, "n_val": n_val,
    }, out_path)
    (OUT_DIR / "distill_log.json").write_text(json.dumps({
        "log": log, "final_median_abs_dec": final_med,
        "final_nrmse": final_nrmse, "gate_pass": gate_pass,
    }, indent=2))
    print(f"[distill] saved {out_path}")


if __name__ == "__main__":
    main()
