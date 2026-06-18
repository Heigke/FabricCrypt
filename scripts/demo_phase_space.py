"""3D phase-space animation of a 2T NS-RAM cell trajectory.

Sweeps Vd from 0 → 2.0 V at three different VG1/VG2 settings,
plotting the (Vb, Vsint, log10|Id|) trajectory in 3D as Vd advances.
The snapback fold is visible as a sharp deflection in (Vb, Id).

Output: figures/demos/phase_space_3d.{png, mp4}
"""
from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/demos"; OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results/demo_phase_space"; RESULTS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

import importlib.util
sp = importlib.util.spec_from_file_location("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(sp); sp.loader.exec_module(z91f)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


def build_models():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    sp = parse_param_blocks(text_M2)
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=sp)
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=sp)
    z91f.patch_model_values(M1, type_n=True)
    z91f.patch_model_values(M2, type_n=True)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=50)
    cfg._sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M2 = compute_size_dep(M2, Geometry(L=cfg.Ln*cfg.M2_length_factor,
                                                W=cfg.Wn), T_C=cfg.T_C)
    return M1, M2, cfg


def run_trace(M1, M2, cfg, VG1: float, VG2: float, n_pts: int = 60):
    """Run forward_2t_arclength_grad and return (Vd, Vb, Vsint, Id) arrays."""
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 100.0   # M3b honest physical (post-O18/O19 walk-back from 2e4)
    Vd_seq = torch.linspace(0.05, 2.0, n_pts, dtype=torch.float64)
    with torch.no_grad():
        out = forward_2t_arclength_grad(
            cfg, model_M1=M1, model_M2=M2, bjt=bjt,
            Vd_seq=Vd_seq, VG1=torch.tensor(VG1), VG2=torch.tensor(VG2))
    return {
        "Vd":    Vd_seq.numpy(),
        "Vb":    out["Vb"].numpy(),
        "Vsint": out["Vsint"].numpy(),
        "Id":    np.abs(out["Id"].numpy()),
        "VG1":   VG1, "VG2": VG2,
    }


def main():
    t0 = time.time()
    print(f"[demo_ps] starting at {time.strftime('%H:%M:%S')}", flush=True)
    M1, M2, cfg = build_models()

    biases = [
        (0.6, +0.30),  # high VG1, mid VG2 — clean snapback regime
        (0.4, +0.10),  # mid  VG1, low VG2 — high-residual row at honest cell
        (0.2, +0.20),  # low  VG1, mid VG2 — sub-threshold-dominant
    ]
    traces = []
    for vg1, vg2 in biases:
        print(f"  trace VG1={vg1} VG2={vg2:+.2f}", flush=True)
        tr = run_trace(M1, M2, cfg, vg1, vg2, n_pts=60)
        traces.append(tr)
    print(f"  reservoir traces wall: {time.time()-t0:.1f}s", flush=True)

    # Static 3D plot — all three traces overlaid, color = Vd
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    for tr, color in zip(traces, ['#1f77b4', '#d62728', '#2ca02c']):
        log_Id = np.log10(tr['Id'] + 1e-15)
        ax.plot(tr['Vb'], tr['Vsint'], log_Id, '-', lw=1.8, color=color, alpha=0.9,
                 label=f"VG1={tr['VG1']} VG2={tr['VG2']:+.2f}")
        ax.scatter(tr['Vb'][0], tr['Vsint'][0], log_Id[0],
                   color=color, s=60, marker='o', edgecolors='k')
        ax.scatter(tr['Vb'][-1], tr['Vsint'][-1], log_Id[-1],
                   color=color, s=60, marker='s', edgecolors='k')

    ax.set_xlabel('Vb [V] (body)')
    ax.set_ylabel('Vsint [V] (M1.S = M2.D)')
    ax.set_zlabel('log10 |Id| [A]')
    ax.set_title(
        '2T NS-RAM cell — 3D phase-space (Vb, Vsint, log|Id|) as Vd sweeps 0.05 → 2.0 V\n'
        'circle = low-Vd start, square = high-Vd end. Bf=100 (physical), η-bounded honest cell.',
        fontsize=10, weight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=22, azim=-58)
    out_png = OUT / "phase_space_3d.png"
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"[demo_ps] saved {out_png}", flush=True)

    # Animation: rotate the camera while traces appear sequentially
    print(f"[demo_ps] rendering rotating mp4 (this may take ~60s)...", flush=True)
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')

    n_frames = 180
    log_Id_all = [np.log10(tr['Id'] + 1e-15) for tr in traces]
    Vb_min = min(tr['Vb'].min() for tr in traces) - 0.05
    Vb_max = max(tr['Vb'].max() for tr in traces) + 0.05
    Vs_min = min(tr['Vsint'].min() for tr in traces) - 0.05
    Vs_max = max(tr['Vsint'].max() for tr in traces) + 0.05
    Id_min = min(l.min() for l in log_Id_all) - 0.5
    Id_max = max(l.max() for l in log_Id_all) + 0.5

    def update(i):
        ax2.clear()
        ax2.set_xlim(Vb_min, Vb_max); ax2.set_ylim(Vs_min, Vs_max)
        ax2.set_zlim(Id_min, Id_max)
        ax2.set_xlabel('Vb [V]'); ax2.set_ylabel('Vsint [V]')
        ax2.set_zlabel('log10 |Id|')
        for tr, color, log_Id in zip(traces, ['#1f77b4', '#d62728', '#2ca02c'],
                                     log_Id_all):
            # Draw whole trace; animate camera + a moving "current Vd" marker
            ax2.plot(tr['Vb'], tr['Vsint'], log_Id, '-', lw=1.5, color=color,
                     alpha=0.7,
                     label=f"VG1={tr['VG1']} VG2={tr['VG2']:+.2f}")
            j = int((i / n_frames) * (len(tr['Vd']) - 1))
            ax2.scatter([tr['Vb'][j]], [tr['Vsint'][j]], [log_Id[j]],
                        s=120, color=color, edgecolors='k', linewidths=1.5,
                        zorder=10)
        Vd_now = 0.05 + (2.0 - 0.05) * (i / n_frames)
        ax2.set_title(f'Phase-space evolution as Vd → {Vd_now:.2f} V',
                       weight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.view_init(elev=22, azim=-58 + 0.5*i)
        return []

    anim = animation.FuncAnimation(fig2, update, frames=n_frames,
                                    interval=80, blit=False)
    out_mp4 = OUT / "phase_space_3d.mp4"
    anim.save(out_mp4, writer='ffmpeg', fps=18, dpi=110)
    plt.close(fig2)
    print(f"[demo_ps] saved {out_mp4}", flush=True)

    json.dump({"biases": biases, "Bf": 100.0, "n_pts": 60,
                "wall_s": time.time() - t0,
                "traces": [{"VG1": tr["VG1"], "VG2": tr["VG2"],
                              "Vd_min": float(tr["Vd"].min()),
                              "Vd_max": float(tr["Vd"].max()),
                              "Vb_range": [float(tr["Vb"].min()),
                                           float(tr["Vb"].max())],
                              "Id_range": [float(tr["Id"].min()),
                                           float(tr["Id"].max())]}
                             for tr in traces]},
               (RESULTS / "summary.json").open("w"), indent=2)
    print(f"[demo_ps] DONE  wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
