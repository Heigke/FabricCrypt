#!/usr/bin/env python3
"""
NES-GD smoke — Noise-Exploiting Gradient Descent via SPSA on small MLP/MNIST.

Compares three trainers (like-for-like SPSA steps):
  A) Gaussian PRNG SPSA  (baseline weight-perturbation)
  B) NS-RAM device-noise SPSA  (impact-ionization shot + 1/f, sampled from surrogate)
  C) Backprop reference

NS-RAM noise model:
  Source: results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz (Id, Iii grids)
  At each operating bias, the impact-ionization current Iii dominates excess noise.
  Per Sah/Hooge, the noise PSD ≈ 2qIii (shot) + KH * Iii^2 / f (1/f).
  We synthesize samples per "cell" by:
     - shot noise:  Poisson-equivalent Gaussian with sigma = sqrt(2 q Iii dt)
       (Gaussian limit, but combined with non-stationary Iii makes mixture non-Gaussian)
     - 1/f noise:  inverse-FFT of 1/sqrt(f) spectrum, scaled by sqrt(KH) * Iii
     - device-to-device offset:  per-cell DC bias drawn from cell vt/vg variation
  We deliberately add CROSS-CELL CORRELATION via shared rail/temperature drift to
  measure the K2 bias gate honestly.

Outputs to results/NES_GD_smoke/:
  training_curves.png, noise_correlation_matrix.json, gradient_variance.json,
  summary.json, honest_analysis.md
"""

import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results", "NES_GD_smoke")
os.makedirs(OUTDIR, exist_ok=True)
SURROGATE = os.path.join(ROOT, "results", "z271_pmp3_dense_surrogate", "surrogate_4d_v2.npz")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device={DEV} torch={torch.__version__}", flush=True)

# -----------------------------------------------------------------------------
# 1) NS-RAM noise atlas
# -----------------------------------------------------------------------------
class NSRAMNoiseSampler:
    """Synthesize per-cell NS-RAM impact-ionization noise from the 4D surrogate.

    Each "cell" is a fixed operating point (vg1,vg2,vd,vb) → Iii.
    We pre-build per-cell noise reservoirs combining shot + 1/f, with optional
    shared correlated drift to honestly measure cross-cell correlation.
    """
    Q_E = 1.602176634e-19
    KH = 2.0e-3   # Hooge-like coefficient (dimensionless, paper-typical 1e-4..1e-2)

    def __init__(self, npz_path, n_cells, dt=1.0e-6, reservoir_len=131072,
                 shared_drift_frac=0.05, seed=0):
        data = np.load(npz_path)
        Iii = data["Iii"]  # shape (7,10,5,13)
        conv = data["converged"]
        # restrict to converged, Iii > 0
        idx = np.argwhere(conv & (Iii > 0.0))
        if len(idx) < n_cells:
            # fallback to all converged, take abs
            idx = np.argwhere(conv)
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(idx), size=n_cells, replace=(len(idx) < n_cells))
        self.cell_idx = idx[pick]
        self.Iii_cell = np.array([abs(Iii[tuple(i)]) for i in self.cell_idx])
        # Ensure non-zero scale
        self.Iii_cell = np.clip(self.Iii_cell, 1e-12, None)
        # Operating-point IIi range
        self.dt = dt
        self.n_cells = n_cells
        self.reservoir_len = reservoir_len
        self.shared_drift_frac = shared_drift_frac
        self._build_reservoirs(seed)

    def _one_over_f(self, n, rng):
        # Build 1/f via inverse FFT of 1/sqrt(f) spectrum
        nf = n // 2 + 1
        f = np.arange(nf)
        f[0] = 1.0
        amp = 1.0 / np.sqrt(f)
        phase = rng.uniform(0, 2*np.pi, size=nf)
        spec = amp * np.exp(1j*phase)
        spec[0] = 0
        x = np.fft.irfft(spec, n=n)
        x -= x.mean()
        s = x.std()
        if s > 0:
            x /= s
        return x.astype(np.float32)

    def _build_reservoirs(self, seed):
        rng = np.random.default_rng(seed + 1)
        n = self.reservoir_len
        # Shared correlated drift component (slow, common-mode rail / temp)
        shared = self._one_over_f(n, rng) * self.shared_drift_frac
        self.reservoirs = np.empty((self.n_cells, n), dtype=np.float32)
        for c in range(self.n_cells):
            Iii = self.Iii_cell[c]
            sigma_shot = math.sqrt(2.0 * self.Q_E * Iii * self.dt)
            sigma_1f   = math.sqrt(self.KH) * Iii
            shot = rng.standard_normal(n).astype(np.float32) * sigma_shot
            oneoverf = self._one_over_f(n, rng) * sigma_1f
            x = shot + oneoverf + shared * sigma_1f
            # standardise per cell to unit variance for use as perturbation
            x = (x - x.mean()) / (x.std() + 1e-30)
            self.reservoirs[c] = x
        self.cursor = 0

    def sample(self, n_samples):
        """Returns (n_samples, n_cells) noise samples, advancing cursor."""
        if self.cursor + n_samples > self.reservoir_len:
            self.cursor = 0
        out = self.reservoirs[:, self.cursor:self.cursor + n_samples].T.copy()
        self.cursor += n_samples
        return out  # shape (n_samples, n_cells)

    def correlation_matrix(self, n_eval=4096):
        # Use beginning of reservoirs to compute pairwise corr
        X = self.reservoirs[:, :n_eval]  # (n_cells, n_eval)
        X = X - X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True) + 1e-30
        Xn = X / s
        C = (Xn @ Xn.T) / n_eval
        return C

    def histogram_stats(self):
        x = self.reservoirs.flatten()
        return dict(
            mean=float(x.mean()),
            std=float(x.std()),
            skew=float(((x - x.mean())**3).mean() / (x.std()**3 + 1e-30)),
            kurt_excess=float(((x - x.mean())**4).mean() / (x.std()**4 + 1e-30) - 3.0),
        )


# -----------------------------------------------------------------------------
# 2) Tiny MLP
# -----------------------------------------------------------------------------
class TinyMLP(nn.Module):
    def __init__(self, in_dim=784, hidden=32, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def flat_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def param_count(model):
    return sum(p.numel() for p in flat_params(model))

def get_flat(model):
    return torch.cat([p.detach().view(-1) for p in flat_params(model)])

def set_flat(model, vec):
    i = 0
    for p in flat_params(model):
        n = p.numel()
        p.data.copy_(vec[i:i+n].view_as(p))
        i += n

def add_perturbation(model, delta):
    """delta is a 1-D tensor with same length as flat params."""
    i = 0
    for p in flat_params(model):
        n = p.numel()
        p.data.add_(delta[i:i+n].view_as(p))
        i += n


# -----------------------------------------------------------------------------
# 3) Training loops
# -----------------------------------------------------------------------------
def loss_on_batch(model, xb, yb):
    logits = model(xb)
    return F.cross_entropy(logits, yb)

def eval_acc(model, loader):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEV); yb = yb.to(DEV)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def spsa_step(model, xb, yb, eps, noise_vec):
    """One SPSA gradient estimate using two forward passes.
    noise_vec: (D,) tensor on DEV (already perturbation direction; we use +/- eps).
    Returns (grad_estimate, loss_plus, loss_minus).
    """
    w0 = get_flat(model).clone()
    # +eps
    set_flat(model, w0 + eps * noise_vec)
    with torch.no_grad():
        lp = loss_on_batch(model, xb, yb).item()
    # -eps
    set_flat(model, w0 - eps * noise_vec)
    with torch.no_grad():
        lm = loss_on_batch(model, xb, yb).item()
    set_flat(model, w0)
    g = ((lp - lm) / (2.0 * eps)) * noise_vec  # SPSA: g_i = (L+ - L-)/(2 eps n_i)
    # Standard SPSA uses 1/n_i, but for Rademacher-like Gaussian we use multiplicative
    # weight-perturbation form (Cauwenberghs 1993): grad = ((L+-L-)/(2eps)) * noise
    return g, lp, lm


def train_spsa(noise_source, model, train_loader, test_loader,
               n_steps=600, eps=1e-3, lr=1e-3, log_every=20,
               grad_var_record=None, tag=""):
    """noise_source: callable(D) -> torch tensor (D,) on DEV."""
    opt = torch.optim.Adam(flat_params(model), lr=lr)
    D = param_count(model)
    history = {"step": [], "loss": [], "test_acc": []}
    train_iter = iter(train_loader)
    grad_norms = []
    grad_sample = []
    t0 = time.time()
    for step in range(n_steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(DEV); yb = yb.to(DEV)
        noise_vec = noise_source(D).to(DEV)
        g, lp, lm = spsa_step(model, xb, yb, eps, noise_vec)
        opt.zero_grad()
        # write g into params' .grad
        i = 0
        for p in flat_params(model):
            n = p.numel()
            p.grad = g[i:i+n].view_as(p).clone()
            i += n
        opt.step()
        grad_norms.append(float(g.norm().item()))
        if grad_var_record is not None and step < 200:
            # store first 1000 entries of g for variance comparison across steps
            grad_sample.append(g[:1000].detach().cpu().numpy().copy())
        if (step+1) % log_every == 0 or step == 0:
            acc = eval_acc(model, test_loader)
            history["step"].append(step+1)
            history["loss"].append(0.5*(lp+lm))
            history["test_acc"].append(acc)
            print(f"[{tag}] step {step+1}/{n_steps} loss={0.5*(lp+lm):.4f} acc={acc:.3f} "
                  f"|g|={grad_norms[-1]:.3e} t={time.time()-t0:.1f}s", flush=True)
            model.train()
    if grad_var_record is not None and grad_sample:
        arr = np.stack(grad_sample, axis=0)  # (S, 1000)
        grad_var_record[tag] = dict(
            per_step_norm_mean=float(np.mean(grad_norms)),
            per_step_norm_std=float(np.std(grad_norms)),
            per_coord_var_mean=float(arr.var(axis=0).mean()),
            per_coord_var_std=float(arr.var(axis=0).std()),
        )
    return history


def train_bp(model, train_loader, test_loader, n_steps=600, lr=1e-3, log_every=20):
    opt = torch.optim.Adam(flat_params(model), lr=lr)
    history = {"step": [], "loss": [], "test_acc": []}
    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(n_steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(DEV); yb = yb.to(DEV)
        opt.zero_grad()
        loss = loss_on_batch(model, xb, yb)
        loss.backward()
        opt.step()
        if (step+1) % log_every == 0 or step == 0:
            acc = eval_acc(model, test_loader)
            history["step"].append(step+1)
            history["loss"].append(float(loss.item()))
            history["test_acc"].append(acc)
            print(f"[BP] step {step+1}/{n_steps} loss={loss.item():.4f} acc={acc:.3f} t={time.time()-t0:.1f}s", flush=True)
            model.train()
    return history


# -----------------------------------------------------------------------------
# 4) Main
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # MNIST  -- smoke: 200 train / 200 test
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_full = datasets.MNIST(root=os.path.expanduser("~/data/mnist"), train=True, download=True, transform=tfm)
    test_full  = datasets.MNIST(root=os.path.expanduser("~/data/mnist"), train=False, download=True, transform=tfm)
    rng = np.random.default_rng(0)
    train_idx = rng.choice(len(train_full), size=200, replace=False)
    test_idx  = rng.choice(len(test_full),  size=200, replace=False)
    train_set = Subset(train_full, train_idx.tolist())
    test_set  = Subset(test_full,  test_idx.tolist())
    bs = 32
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=bs, shuffle=False)
    print(f"[data] train=200 test=200 bs={bs}", flush=True)

    # Model
    HIDDEN = 32
    model_g = TinyMLP(hidden=HIDDEN).to(DEV)
    model_n = TinyMLP(hidden=HIDDEN).to(DEV)
    model_b = TinyMLP(hidden=HIDDEN).to(DEV)
    # Identical init for fair comparison
    init_state = model_g.state_dict()
    model_n.load_state_dict(init_state)
    model_b.load_state_dict(init_state)
    D = param_count(model_g)
    print(f"[model] D={D}", flush=True)

    # NS-RAM noise atlas — one cell per parameter
    print(f"[noise] building NS-RAM atlas: {D} cells", flush=True)
    sampler = NSRAMNoiseSampler(SURROGATE, n_cells=D, reservoir_len=4096,
                                shared_drift_frac=0.05, seed=42)
    nsram_stats = sampler.histogram_stats()
    print(f"[noise] NS-RAM stats: {nsram_stats}", flush=True)
    Cmat = sampler.correlation_matrix(n_eval=2048)
    # take only off-diag
    off = Cmat[np.triu_indices_from(Cmat, k=1)]
    corr_stats = dict(
        mean_abs_corr=float(np.mean(np.abs(off))),
        max_abs_corr=float(np.max(np.abs(off))),
        frac_gt_0p3=float((np.abs(off) > 0.3).mean()),
        n_pairs=int(off.size),
    )
    print(f"[K2] cross-cell corr: {corr_stats}", flush=True)

    # Build noise sources
    def noise_gauss(d):
        return torch.randn(d, device=DEV)
    def noise_nsram(d):
        v = sampler.sample(1)[0]  # (D,)
        return torch.from_numpy(v).to(DEV)

    # Hyperparams
    N_STEPS = 600
    EPS = 5e-3
    LR  = 5e-3

    grad_var_record = {}

    print("\n=== A) Gaussian SPSA ===", flush=True)
    hist_g = train_spsa(noise_gauss, model_g, train_loader, test_loader,
                        n_steps=N_STEPS, eps=EPS, lr=LR,
                        grad_var_record=grad_var_record, tag="GAUSS")

    print("\n=== B) NS-RAM SPSA ===", flush=True)
    hist_n = train_spsa(noise_nsram, model_n, train_loader, test_loader,
                        n_steps=N_STEPS, eps=EPS, lr=LR,
                        grad_var_record=grad_var_record, tag="NSRAM")

    print("\n=== C) BP baseline ===", flush=True)
    hist_b = train_bp(model_b, train_loader, test_loader, n_steps=N_STEPS, lr=LR)

    final_g = hist_g["test_acc"][-1]
    final_n = hist_n["test_acc"][-1]
    final_b = hist_b["test_acc"][-1]

    # --- save artifacts ---
    # 1) curves
    plt.figure(figsize=(8,5))
    plt.plot(hist_g["step"], hist_g["test_acc"], label=f"Gauss SPSA  (final {final_g:.3f})", marker='o', lw=1)
    plt.plot(hist_n["step"], hist_n["test_acc"], label=f"NS-RAM SPSA (final {final_n:.3f})", marker='s', lw=1)
    plt.plot(hist_b["step"], hist_b["test_acc"], label=f"BP          (final {final_b:.3f})", marker='^', lw=1)
    plt.xlabel("step")
    plt.ylabel("test acc (200-sample split)")
    plt.title(f"NES-GD smoke: MNIST 784-{HIDDEN}-10, batch {bs}, lr={LR}, eps={EPS}")
    plt.grid(alpha=0.3); plt.legend()
    out_png = os.path.join(OUTDIR, "training_curves.png")
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"[save] {out_png}", flush=True)

    # 2) correlation matrix (downsampled for storage)
    C_save = Cmat[:64,:64].tolist()
    with open(os.path.join(OUTDIR, "noise_correlation_matrix.json"), "w") as f:
        json.dump({
            "n_cells_total": D,
            "matrix_64x64_preview": C_save,
            "stats": corr_stats,
        }, f, indent=2)

    # 3) gradient variance
    with open(os.path.join(OUTDIR, "gradient_variance.json"), "w") as f:
        json.dump(grad_var_record, f, indent=2)

    # 4) summary
    K2_BIAS = corr_stats["mean_abs_corr"] > 0.3 or corr_stats["frac_gt_0p3"] > 0.5
    INFRA   = (not math.isnan(final_g)) and (not math.isnan(final_n))
    DISC    = final_n > 0.30
    AMBIT   = final_n >= 0.80 * final_g
    summary = {
        "device": DEV,
        "n_steps": N_STEPS, "eps": EPS, "lr": LR, "hidden": HIDDEN, "D": D,
        "final_acc": {"gauss_spsa": final_g, "nsram_spsa": final_n, "bp": final_b},
        "nsram_noise_stats": nsram_stats,
        "cross_cell_corr": corr_stats,
        "grad_variance": grad_var_record,
        "gates": {"INFRA": bool(INFRA), "DISCOVERY": bool(DISC),
                  "AMBITIOUS": bool(AMBIT), "K2_BIAS": bool(K2_BIAS)},
    }
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 5) honest analysis
    md = []
    md.append("# NES-GD smoke — honest analysis\n")
    md.append(f"- device: `{DEV}`\n")
    md.append(f"- params D={D}, hidden={HIDDEN}, batch={bs}, steps={N_STEPS}, eps={EPS}, lr={LR}\n")
    md.append(f"- MNIST: 200 train / 200 test (smoke)\n\n")
    md.append("## Final test accuracy\n")
    md.append(f"- Gaussian SPSA : **{final_g:.3f}**\n")
    md.append(f"- NS-RAM SPSA   : **{final_n:.3f}**\n")
    md.append(f"- BP (Adam)     : **{final_b:.3f}**\n\n")
    md.append("## NS-RAM noise (synthesized from z271 4D surrogate Iii)\n")
    md.append(f"- mean={nsram_stats['mean']:.3e}, std={nsram_stats['std']:.3e}\n")
    md.append(f"- skew={nsram_stats['skew']:.3f}, excess kurt={nsram_stats['kurt_excess']:.3f}  ← non-zero kurtosis ⇒ non-Gaussian\n\n")
    md.append("## K2 cross-cell correlation audit\n")
    md.append(f"- mean |corr| over {corr_stats['n_pairs']} pairs: **{corr_stats['mean_abs_corr']:.4f}**\n")
    md.append(f"- max |corr|: {corr_stats['max_abs_corr']:.4f}\n")
    md.append(f"- fraction with |corr|>0.3: {corr_stats['frac_gt_0p3']:.4f}\n")
    md.append(f"- KILL_SHOT K2 triggered: **{K2_BIAS}**\n\n")
    md.append("## Gates\n")
    md.append(f"- INFRA (loop runs, no NaN): **{INFRA}**\n")
    md.append(f"- DISCOVERY (NS-RAM SPSA > 30%): **{DISC}**\n")
    md.append(f"- AMBITIOUS (NS-RAM ≥ 80% of Gauss): **{AMBIT}**  ({final_n:.3f} vs {0.8*final_g:.3f})\n")
    md.append(f"- K2 BIAS (mean|corr|>0.3 or >50% pairs>0.3): **{K2_BIAS}**\n\n")
    md.append("## Honest caveats\n")
    md.append("- 200/200 smoke split → noisy acc; absolute numbers are smoke-grade.\n")
    md.append("- NS-RAM noise is *synthesized* from the offline surrogate Iii field (shot + 1/f), not live-sampled from hardware. We injected a small shared-drift component (5% std) to honestly probe K2.\n")
    md.append("- SPSA uses the Cauwenberghs weight-perturbation form g = ((L+-L-)/(2ε))·ξ. With Gaussian ξ, this is unbiased only on average over many directions; non-Gaussian ξ introduces a finite-sample bias term proportional to E[ξ³]/D.\n")
    with open(os.path.join(OUTDIR, "honest_analysis.md"), "w") as f:
        f.write("".join(md))

    print("\n========== SUMMARY ==========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
