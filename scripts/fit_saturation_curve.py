"""Fit linear vs saturating curves to the 5-task Δ-vs-projection data.

Linear (MNIST-band): Δ = a + b·proj  (already done; valid 43-72%)
Saturating (5-point):  Δ = A·tanh(B·(C - proj))
Logistic-like (5-point): Δ = A·(1 - 1/(1 + exp(-(proj - C)/D))) - E

Reports R² for each on all 5 points + visualization.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent

def load(p): return json.loads((ROOT / p).read_text())

S = {
    "MNIST":         load("results/z235_strong_input_30seed/summary.json"),
    "KMNIST":        load("results/z237_kmnist/summary.json"),
    "FMNIST_small":  load("results/z238_fmnist_smalltrain/summary.json"),
    "FashionMNIST":  load("results/z236_fashion_mnist/summary.json"),
    "CIFAR":         load("results/z240_cifar_local/summary.json"),
}
points = [(name, s["proj_mean"]*100, s["delta_mean_pp"]) for name, s in S.items()]
points.sort(key=lambda r: r[1])
names = [p[0] for p in points]
x = np.array([p[1] for p in points])
y = np.array([p[2] for p in points])

print(f"=== 5 datapoints (sorted by projection baseline) ===")
print(f"{'task':<14}  proj%   Δ_pp")
for n, px, py in zip(names, x, y):
    print(f"  {n:<14} {px:5.1f}  {py:+6.2f}")

# Model 1: linear (MNIST-band only, exclude CIFAR)
mask_band = (x >= 40)
slope_lin, intc_lin = np.polyfit(x[mask_band], y[mask_band], 1)
y_lin_band = slope_lin * x[mask_band] + intc_lin
ss_res_band = ((y[mask_band] - y_lin_band)**2).sum()
ss_tot_band = ((y[mask_band] - y[mask_band].mean())**2).sum()
r2_lin_band = 1 - ss_res_band / ss_tot_band

y_lin_all = slope_lin * x + intc_lin
ss_res_all = ((y - y_lin_all)**2).sum()
ss_tot_all = ((y - y.mean())**2).sum()
r2_lin_all = 1 - ss_res_all / ss_tot_all
print(f"\n--- Linear fit (MNIST-band, 4 pts) ---")
print(f"  Δ = {intc_lin:+.2f} {slope_lin:+.3f}·proj")
print(f"  R² on band (4 pts):  {r2_lin_band:.4f}")
print(f"  R² on ALL  (5 pts):  {r2_lin_all:.4f}  (CIFAR off the line)")

# Model 2: tanh saturation
def tanh_model(p, A, B, C):
    return A * np.tanh(B * (C - p))

try:
    popt, _ = curve_fit(tanh_model, x, y, p0=[10.0, 0.05, 53.0], maxfev=10000)
    A, B, C = popt
    y_tanh = tanh_model(x, *popt)
    ss_res_tanh = ((y - y_tanh)**2).sum()
    r2_tanh = 1 - ss_res_tanh / ss_tot_all
    print(f"\n--- tanh saturation fit (5 pts) ---")
    print(f"  Δ = {A:+.2f}·tanh({B:+.4f}·({C:+.2f} - proj))")
    print(f"  R² on ALL (5 pts):   {r2_tanh:.4f}")
    print(f"  Predictions:")
    for n, px, py in zip(names, x, y):
        pred = tanh_model(px, *popt)
        print(f"    {n:<14} actual {py:+6.2f}, predicted {pred:+6.2f}, |err|={abs(py-pred):.2f}")
except Exception as e:
    print(f"tanh fit failed: {e}")

# Model 3: simple sigmoid: Δ = A − B/(1 + exp(-(proj − C)/D))
def sig_model(p, A, B, C, D):
    return A - B / (1 + np.exp(-(p - C)/D))

try:
    popt2, _ = curve_fit(sig_model, x, y, p0=[5.0, 15.0, 53.0, 5.0], maxfev=10000)
    y_sig = sig_model(x, *popt2)
    ss_res_sig = ((y - y_sig)**2).sum()
    r2_sig = 1 - ss_res_sig / ss_tot_all
    print(f"\n--- sigmoid fit (5 pts, 4 params) ---")
    print(f"  Δ = {popt2[0]:+.2f} − {popt2[1]:.2f}/(1+exp(-({popt2[2]:.1f} − proj)/{popt2[3]:.1f}))")
    print(f"  R² on ALL (5 pts):   {r2_sig:.4f}")
except Exception as e:
    print(f"sigmoid fit failed: {e}")

print(f"\n=== Summary ===")
print(f"  Linear (MNIST-band only):  R²={r2_lin_band:.3f}  on 4 pts (43-72%)")
print(f"  Linear extrapolated to 5:  R²={r2_lin_all:.3f}  (poor, CIFAR off)")
print(f"  tanh saturation, 5 pts:    R²={r2_tanh:.3f}  (better)")
print(f"  sigmoid, 5 pts (4 params): R²={r2_sig:.3f}")
