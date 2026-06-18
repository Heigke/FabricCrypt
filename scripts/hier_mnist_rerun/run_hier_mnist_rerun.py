"""HIER-MNIST-RERUN — brutal multi-seed re-verification of 97.15% claim.

This re-uses the architecture from scripts/N_Hier_MNIST.py (2-layer FF-LIF SNN with
skip-connection readout) and adds:
  - n=4 seeds with CI (mean +/- std)
  - vanilla LIF baseline (NSRAM_GAIN=0, no body-state bias)
  - peripheral-aware energy model (DAC + ADC for spike encoding per timestep)
  - explicit calibrated-cell constants are recorded but the SNN surrogate-gradient
    training does NOT invoke the BSIM4 cell (z469/z474b fixes are INVARIANT here).

Outputs to results/N_Hier_MNIST_RERUN/
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("HIER_RERUN_ROOT", "/home/naorw/AMD_gfx1151_energy_network"))
OUT_DIR = ROOT / "results" / "N_Hier_MNIST_RERUN"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Architecture (matches original)
N_CLASSES, T_BINS, INPUT_DIM = 10, 20, 784
HIDDEN_1, HIDDEN_2 = 256, 128
EPOCHS, BATCH = 3, 128
LR, WEIGHT_DECAY = 2e-3, 1e-4

LIF_V_THR, LIF_LEAK, LIF_RESET = 1.0, 0.85, 1.0
NSRAM_ALPHA, NSRAM_BETA, NSRAM_GAIN = 0.05, 0.99, 0.2

# Energy model
ENERGY_PER_SPIKE_PJ = 6.4e-3   # NS-RAM analog soma (original cell-only)
# Peripheral: DAC (input current per pixel, per timestep) + ADC (readout sample)
# Reference: Murmann ISSCC tracker — modern 8-bit DAC ~0.5 pJ/sample, 8-bit ADC ~0.3 pJ/sample
E_DAC_PJ_PER_SAMPLE = 0.5
E_ADC_PJ_PER_SAMPLE = 0.3
# Each inference: T_BINS * INPUT_DIM DAC events, N_CLASSES * 1 ADC events at readout
# (Readout integrates over time then ADCs once per class -- conservative.)
PERIPHERAL_DAC_PJ_PER_INF = T_BINS * INPUT_DIM * E_DAC_PJ_PER_SAMPLE  # 20*784*0.5 = 7840 pJ
# Conservative: only ADC the 10 class accumulators once per inference
PERIPHERAL_ADC_PJ_PER_INF = N_CLASSES * E_ADC_PJ_PER_SAMPLE          # 10*0.3 = 3 pJ

# Post-fix calibrated NS-RAM cell constants (RECORDED; not used in surrogate training)
CALIBRATED_CELL = {
    "snap_Is_A": 4.5192e-12,
    "R_body_Ohm": 1e7,
    "ift_sign_correct": True,
    "note": "z469 I_snap_d transient + z474b IFT gradient. SNN surrogate-grad does "
            "NOT invoke BSIM4 cell, so these are INVARIANT for this script. "
            "Recorded for traceability per ablation pre-reg.",
}


def load_mnist():
    from torchvision import datasets, transforms
    data_root = ROOT / "data"
    tfm = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    te = datasets.MNIST(str(data_root), train=False, download=True, transform=tfm)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0
    ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int64)
    return Xtr.reshape(-1, INPUT_DIM), ytr, Xte.reshape(-1, INPUT_DIM), yte


def rate_encode(x_np, device):
    x = torch.from_numpy(x_np).to(device)
    return x.unsqueeze(1).expand(-1, T_BINS, -1).contiguous()


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return (x > 0).float()
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (1.0 / (1.0 + 10.0 * x.abs()) ** 2)
spike_fn = SurrogateSpike.apply


class FFLIFLayer(nn.Module):
    def __init__(self, in_dim, n, nsram_gain=NSRAM_GAIN):
        super().__init__()
        self.n = n
        self.gain = nsram_gain
        self.W_in = nn.Linear(in_dim, n, bias=True)
        nn.init.kaiming_normal_(self.W_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.W_in.bias)
    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        V_b = torch.zeros(B, self.n, device=x_seq.device)
        V_body_slow = torch.zeros(B, self.n, device=x_seq.device)
        s_prev = torch.zeros(B, self.n, device=x_seq.device)
        I_in = self.W_in(x_seq)
        sp = []
        for t in range(T):
            V_b = LIF_LEAK * V_b + I_in[:, t] - LIF_RESET * s_prev
            V_thr = LIF_V_THR + self.gain * V_body_slow
            s_t = spike_fn(V_b - V_thr)
            V_body_slow = NSRAM_BETA * V_body_slow + NSRAM_ALPHA * s_t
            s_prev = s_t
            sp.append(s_t)
        return torch.stack(sp, dim=1)


class HierLIFNet(nn.Module):
    def __init__(self, nsram_gain=NSRAM_GAIN):
        super().__init__()
        self.layer1 = FFLIFLayer(INPUT_DIM, HIDDEN_1, nsram_gain)
        self.layer2 = FFLIFLayer(HIDDEN_1, HIDDEN_2, nsram_gain)
        self.readout = nn.Linear(HIDDEN_1 + HIDDEN_2, N_CLASSES)
    def forward(self, x_seq):
        s1 = self.layer1(x_seq); s2 = self.layer2(s1)
        rate_cat = torch.cat([s1.mean(1), s2.mean(1)], dim=1)
        return self.readout(rate_cat), s1, s2


def iter_batches(X, y, batch, shuffle, rng):
    idx = rng.permutation(len(y)) if shuffle else np.arange(len(y))
    for i in range(0, len(y), batch):
        sel = idx[i:i + batch]
        yield X[sel], y[sel]


def train_one_epoch(model, X, y, opt, rng, device):
    model.train()
    tot_loss = tot_correct = n = 0
    t0 = time.time()
    for xb, yb in iter_batches(X, y, BATCH, True, rng):
        B = xb.shape[0]
        xs = rate_encode(xb, device); ys = torch.from_numpy(yb).to(device)
        opt.zero_grad()
        logits, _, _ = model(xs)
        loss = F.cross_entropy(logits, ys)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        tot_loss += loss.item() * B
        tot_correct += (logits.argmax(1) == ys).sum().item()
        n += B
    return tot_loss / n, tot_correct / n, time.time() - t0


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    correct = n_inf = 0
    spikes = 0.0
    t0 = time.time()
    for xb, yb in iter_batches(X, y, BATCH, False, np.random.default_rng(0)):
        B = xb.shape[0]
        xs = rate_encode(xb, device); ys = torch.from_numpy(yb).to(device)
        logits, s1, s2 = model(xs)
        correct += (logits.argmax(1) == ys).sum().item()
        n_inf += B
        spikes += float(s1.sum().item()) + float(s2.sum().item())
    dt = time.time() - t0
    acc = correct / n_inf
    sp_per_inf = spikes / n_inf
    e_cell = sp_per_inf * ENERGY_PER_SPIKE_PJ
    e_peripheral = PERIPHERAL_DAC_PJ_PER_INF + PERIPHERAL_ADC_PJ_PER_INF
    e_total = e_cell + e_peripheral
    return dict(acc=acc, spikes_per_inf=sp_per_inf, throughput_ips=n_inf/dt,
                e_cell_pJ=e_cell, e_peripheral_pJ=e_peripheral, e_total_pJ=e_total,
                wall=dt)


def run_one(seed, nsram_gain, device, Xtr, ytr, Xte, yte):
    torch.manual_seed(seed); np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = HierLIFNet(nsram_gain=nsram_gain).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed + 7)
    losses, accs = [], []
    t0 = time.time()
    for epoch in range(EPOCHS):
        loss, acc, dt = train_one_epoch(model, Xtr, ytr, opt, rng, device)
        losses.append(loss); accs.append(acc)
        print(f"  [seed={seed} gain={nsram_gain:.2f} ep {epoch+1}/{EPOCHS}] "
              f"loss={loss:.4f} train_acc={acc:.4f} ({dt:.1f}s)", flush=True)
    metrics = evaluate(model, Xte, yte, device)
    metrics["seed"] = seed
    metrics["nsram_gain"] = nsram_gain
    metrics["params"] = n_params
    metrics["train_loss"] = losses
    metrics["train_acc"] = accs
    metrics["wall_total"] = time.time() - t0
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)
    if device == "cuda":
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"[data] train={Xtr.shape} test={Xte.shape}", flush=True)

    seeds = [0, 1, 2, 3]
    nsram_results, baseline_results = [], []
    t_all = time.time()

    print("=== NS-RAM-flavored (gain=0.2) — 4 seeds ===", flush=True)
    for s in seeds:
        nsram_results.append(run_one(s, NSRAM_GAIN, device, Xtr, ytr, Xte, yte))
        m = nsram_results[-1]
        print(f"  -> seed={s}: test_acc={m['acc']:.4f} "
              f"e_total={m['e_total_pJ']:.1f} pJ/inf", flush=True)

    print("=== Vanilla LIF baseline (gain=0.0) — 4 seeds ===", flush=True)
    for s in seeds:
        baseline_results.append(run_one(s, 0.0, device, Xtr, ytr, Xte, yte))
        m = baseline_results[-1]
        print(f"  -> seed={s}: test_acc={m['acc']:.4f} "
              f"e_total={m['e_total_pJ']:.1f} pJ/inf", flush=True)

    def stats(rs, key):
        v = np.array([r[key] for r in rs], dtype=np.float64)
        return dict(mean=float(v.mean()), std=float(v.std(ddof=1)),
                    min=float(v.min()), max=float(v.max()),
                    values=v.tolist())

    summary = dict(
        seeds=seeds, epochs=EPOCHS, batch=BATCH, T_bins=T_BINS,
        hidden_1=HIDDEN_1, hidden_2=HIDDEN_2,
        calibrated_cell=CALIBRATED_CELL,
        nsram=dict(
            acc=stats(nsram_results, "acc"),
            e_cell_pJ=stats(nsram_results, "e_cell_pJ"),
            e_peripheral_pJ=stats(nsram_results, "e_peripheral_pJ"),
            e_total_pJ=stats(nsram_results, "e_total_pJ"),
            spikes_per_inf=stats(nsram_results, "spikes_per_inf"),
            throughput_ips=stats(nsram_results, "throughput_ips"),
            params=nsram_results[0]["params"],
            per_seed=nsram_results,
        ),
        baseline=dict(
            acc=stats(baseline_results, "acc"),
            e_cell_pJ=stats(baseline_results, "e_cell_pJ"),
            e_peripheral_pJ=stats(baseline_results, "e_peripheral_pJ"),
            e_total_pJ=stats(baseline_results, "e_total_pJ"),
            spikes_per_inf=stats(baseline_results, "spikes_per_inf"),
            params=baseline_results[0]["params"],
            per_seed=baseline_results,
        ),
        wall_total_s=time.time() - t_all,
        device=device,
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
    )

    # --- pre-registered verdict ---
    m_acc = summary["nsram"]["acc"]["mean"]
    m_eper = summary["nsram"]["e_total_pJ"]["mean"]
    min_acc = summary["nsram"]["acc"]["min"]
    if min_acc < 0.90:
        verdict = "KILL"
    elif m_acc < 0.96 or m_eper > 100.0:
        verdict = "DEMOTE"
    else:
        verdict = "SURVIVES"
    summary["verdict"] = verdict
    summary["verdict_inputs"] = dict(
        mean_acc_4seed=m_acc, min_acc=min_acc,
        peripheral_aware_energy_pJ_inf=m_eper,
        thresholds=dict(survive_acc=0.96, survive_e=100.0, kill_min_acc=0.90),
    )

    # --- write artifacts ---
    (OUT_DIR / "mean_std_seeds.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "peripheral_energy.json").write_text(json.dumps(dict(
        E_DAC_PJ_PER_SAMPLE=E_DAC_PJ_PER_SAMPLE,
        E_ADC_PJ_PER_SAMPLE=E_ADC_PJ_PER_SAMPLE,
        dac_samples_per_inf=T_BINS * INPUT_DIM,
        adc_samples_per_inf=N_CLASSES,
        peripheral_pJ_per_inf=PERIPHERAL_DAC_PJ_PER_INF + PERIPHERAL_ADC_PJ_PER_INF,
        cell_pJ_per_spike=ENERGY_PER_SPIKE_PJ,
        nsram=dict(
            cell_pJ_per_inf=summary["nsram"]["e_cell_pJ"]["mean"],
            total_pJ_per_inf=summary["nsram"]["e_total_pJ"]["mean"],
        ),
        baseline=dict(
            cell_pJ_per_inf=summary["baseline"]["e_cell_pJ"]["mean"],
            total_pJ_per_inf=summary["baseline"]["e_total_pJ"]["mean"],
        ),
    ), indent=2))
    (OUT_DIR / "baseline_compare.json").write_text(json.dumps(dict(
        nsram_acc_mean=summary["nsram"]["acc"]["mean"],
        nsram_acc_std=summary["nsram"]["acc"]["std"],
        baseline_acc_mean=summary["baseline"]["acc"]["mean"],
        baseline_acc_std=summary["baseline"]["acc"]["std"],
        delta_pp=(summary["nsram"]["acc"]["mean"] -
                  summary["baseline"]["acc"]["mean"]) * 100.0,
    ), indent=2))

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for r in nsram_results:
            axes[0].plot(range(1, EPOCHS + 1), r["train_acc"],
                         label=f"seed={r['seed']}", marker="o")
        axes[0].set_title("NS-RAM-flavored train acc (4 seeds)")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train acc"); axes[0].legend()
        nsram_acc = [r["acc"] for r in nsram_results]
        base_acc = [r["acc"] for r in baseline_results]
        axes[1].boxplot([nsram_acc, base_acc], labels=["NS-RAM gain=0.2", "vanilla LIF"])
        axes[1].axhline(0.9715, color="r", ls="--", label="original 97.15%")
        axes[1].axhline(0.96, color="orange", ls=":", label="survive threshold")
        axes[1].set_title("test acc — 4-seed CI vs original"); axes[1].legend()
        axes[1].set_ylabel("test acc")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "training_curves_4seed.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"[plot] FAILED: {e}", flush=True)

    # --- verdict.md ---
    md = [
        "# HIER-MNIST-RERUN — honest verdict",
        "",
        f"**Verdict: {verdict}**", "",
        "## Pre-registered thresholds",
        "- SURVIVES: mean(test_acc, 4 seeds) >= 0.96 AND peripheral-aware E_per_inf < 100 pJ",
        "- DEMOTE:   mean < 0.96 OR peripheral-aware E > 100 pJ",
        "- KILL:     ANY seed test_acc < 0.90",
        "",
        "## Results (NS-RAM-flavored, gain=0.2, 4 seeds)",
        f"- test_acc mean = {summary['nsram']['acc']['mean']:.4f} "
        f"± {summary['nsram']['acc']['std']:.4f}",
        f"- test_acc range = [{summary['nsram']['acc']['min']:.4f}, "
        f"{summary['nsram']['acc']['max']:.4f}]",
        f"- per-seed = {summary['nsram']['acc']['values']}",
        f"- spikes/inf = {summary['nsram']['spikes_per_inf']['mean']:.1f}",
        f"- E_cell = {summary['nsram']['e_cell_pJ']['mean']:.2f} pJ/inf  "
        "(original quoted 17.7 pJ/inf — cell-only)",
        f"- E_peripheral (DAC+ADC) = {PERIPHERAL_DAC_PJ_PER_INF + PERIPHERAL_ADC_PJ_PER_INF:.1f} pJ/inf",
        f"- **E_total (peripheral-aware) = {summary['nsram']['e_total_pJ']['mean']:.1f} pJ/inf**",
        f"- throughput = {summary['nsram']['throughput_ips']['mean']:.0f} inf/s",
        "",
        "## Vanilla LIF baseline (gain=0.0, 4 seeds)",
        f"- test_acc mean = {summary['baseline']['acc']['mean']:.4f} "
        f"± {summary['baseline']['acc']['std']:.4f}",
        f"- Δ (NSRAM − baseline) = "
        f"{(summary['nsram']['acc']['mean']-summary['baseline']['acc']['mean'])*100:.2f} pp",
        "",
        "## Calibrated cell traceability (RECORDED, NOT INVOKED)",
        f"- snap_Is = {CALIBRATED_CELL['snap_Is_A']:.4e} A",
        f"- R_body  = {CALIBRATED_CELL['R_body_Ohm']:.2e} Ω",
        f"- IFT sign correct (z474b) = {CALIBRATED_CELL['ift_sign_correct']}",
        "",
        "## Material caveat",
        "The N_Hier_MNIST surrogate-gradient SNN does NOT invoke the BSIM4 NS-RAM "
        "cell — neuron dynamics use a simple LIF + slow-bias adapter "
        "(NSRAM_ALPHA/BETA/GAIN). The z469 I_snap_d transient fix and z474b IFT "
        "gradient sign fix are therefore INVARIANT for the test_acc number on this "
        "script. They DO matter for any downstream physical-cell deployment. "
        "The honest re-run here is multi-seed CI + peripheral energy accounting; "
        "the original 17.7 pJ/inf number under-counted DAC/ADC.",
    ]
    (OUT_DIR / "honest_verdict.md").write_text("\n".join(md))
    print("=" * 60, flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"  acc 4-seed: {summary['nsram']['acc']['mean']:.4f} ± "
          f"{summary['nsram']['acc']['std']:.4f}", flush=True)
    print(f"  E_total: {summary['nsram']['e_total_pJ']['mean']:.1f} pJ/inf "
          f"(cell {summary['nsram']['e_cell_pJ']['mean']:.2f} + "
          f"peripheral {PERIPHERAL_DAC_PJ_PER_INF + PERIPHERAL_ADC_PJ_PER_INF:.1f})",
          flush=True)
    print(f"  artifacts -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
