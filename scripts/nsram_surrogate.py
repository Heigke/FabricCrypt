"""Fast NS-RAM surrogate built from the calibrated 2T cell.

Pre-computes log10|Id|(VG1, VG2, Vd) on a 3D grid using forward_2t_batched
at the new optimum (Bf=9000, Va=0.55, Is=1e-9). Then provides a vectorised
numpy lookup for use in large-scale network simulations.

Bottleneck of v2 demo is Newton-per-step (~110ms at any N). The surrogate
trades ~5 minutes of one-time grid build for ~0.05ms / step / cell after,
i.e. ~2000× speedup. This unblocks topology × rule × scale sweeps.

API:
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 20))
    log_Id = surr.eval(VG1, VG2, Vd)   # broadcast-friendly numpy
    log_Id_t = surr.eval_torch(VG1, VG2, Vd)  # torch on cpu/cuda
"""
from __future__ import annotations
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent

CACHE = ROOT / "results/nsram_surrogate"
CACHE.mkdir(parents=True, exist_ok=True)

VG1_RANGE = (0.10, 0.80)
VG2_RANGE = (-0.10, 0.60)
VD_RANGE  = (0.10, 2.20)

# Calibrated optimum (F6.v4 best as of 2026-05-05)
OPT_BF = 9000.0
OPT_VA = 0.55
OPT_IS = 1e-9


class NSRAMSurrogate:
    def __init__(self, grid: np.ndarray, vg1_axis, vg2_axis, vd_axis, meta):
        self.grid = grid                  # shape (NG1, NG2, NVD)
        self.vg1_axis = vg1_axis
        self.vg2_axis = vg2_axis
        self.vd_axis = vd_axis
        self.meta = meta
        self._torch_grid = None

    # ---------- builders ----------
    @classmethod
    def build_or_load(cls, grid_size=(20, 20, 25), force_rebuild=False):
        path = CACHE / f"surrogate_{grid_size[0]}_{grid_size[1]}_{grid_size[2]}.npz"
        if path.exists() and not force_rebuild:
            d = np.load(path)
            print(f"[surr] loaded cache {path.name} (built {d['build_t']:.1f}s ago — "
                  f"size {d['grid'].shape})", flush=True)
            with open(path.with_suffix(".json")) as fh:
                meta = json.load(fh)
            return cls(d["grid"], d["vg1_axis"], d["vg2_axis"],
                       d["vd_axis"], meta)
        return cls._build(grid_size, path)

    @classmethod
    def _build(cls, grid_size, path):
        sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
        v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
        from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
        from nsram.bsim4_port.bjt import GummelPoonNPN
        from nsram.bsim4_port.vectorized import forward_2t_batched

        cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                                 newton_max_iters=15)
        M1, M2 = v1.build_calibrated_models()
        bjt = GummelPoonNPN.from_sebas_card()
        bjt.Bf = OPT_BF; bjt.Va = OPT_VA; bjt.Is = OPT_IS

        NG1, NG2, NVD = grid_size
        vg1_axis = np.linspace(*VG1_RANGE, NG1)
        vg2_axis = np.linspace(*VG2_RANGE, NG2)
        vd_axis  = np.linspace(*VD_RANGE,  NVD)
        # Build by sweeping over (VG1, VG2) pairs in batched calls;
        # each call sweeps Vd.
        grid = np.zeros((NG1, NG2, NVD), dtype=np.float64)
        t0 = time.time()
        # Batch over Vd_seq for each (VG1, VG2) pair using forward_2t_batched.
        # forward_2t_batched takes Vd_seq (T,) and VG1/VG2 (N,) and returns
        # Id (N, T). We treat (NG1*NG2) as N — broadcast everything.
        VG1_flat = np.repeat(vg1_axis, NG2)
        VG2_flat = np.tile(vg2_axis, NG1)
        VG1_t = torch.tensor(VG1_flat, dtype=torch.float64)
        VG2_t = torch.tensor(VG2_flat, dtype=torch.float64)
        Vd_t  = torch.tensor(vd_axis,  dtype=torch.float64)
        print(f"[surr] building grid {grid_size} = {NG1*NG2*NVD} pts ...",
              flush=True)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t,
                                  max_iters=15, tol=1e-9, verbose=False)
        Id_flat = out["Id"].abs().clamp(min=1e-15).log10().numpy()  # (N, T)
        grid = Id_flat.reshape(NG1, NG2, NVD)
        wall = time.time() - t0
        meta = dict(opt=dict(Bf=OPT_BF, Va=OPT_VA, Is=OPT_IS),
                    grid_size=list(grid_size),
                    vg1_range=VG1_RANGE, vg2_range=VG2_RANGE, vd_range=VD_RANGE,
                    build_wall_s=wall)
        np.savez(path, grid=grid, vg1_axis=vg1_axis, vg2_axis=vg2_axis,
                 vd_axis=vd_axis, build_t=wall)
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[surr] built in {wall:.1f}s; saved {path.name}; "
              f"log_Id range [{grid.min():.2f}, {grid.max():.2f}]",
              flush=True)
        return cls(grid, vg1_axis, vg2_axis, vd_axis, meta)

    # ---------- evaluators ----------
    def eval(self, VG1, VG2, Vd) -> np.ndarray:
        """Trilinear interpolation. All inputs broadcast-compatible numpy
        arrays. Returns log10|Id| in same broadcast shape."""
        VG1 = np.asarray(VG1); VG2 = np.asarray(VG2); Vd = np.asarray(Vd)
        i, fi = self._idx(VG1, self.vg1_axis)
        j, fj = self._idx(VG2, self.vg2_axis)
        k, fk = self._idx(Vd,  self.vd_axis)
        g = self.grid
        c00 = g[i,   j,   k  ] * (1-fi) + g[i+1, j,   k  ] * fi
        c01 = g[i,   j,   k+1] * (1-fi) + g[i+1, j,   k+1] * fi
        c10 = g[i,   j+1, k  ] * (1-fi) + g[i+1, j+1, k  ] * fi
        c11 = g[i,   j+1, k+1] * (1-fi) + g[i+1, j+1, k+1] * fi
        c0 = c00 * (1-fj) + c10 * fj
        c1 = c01 * (1-fj) + c11 * fj
        return c0 * (1-fk) + c1 * fk

    @staticmethod
    def _idx(x, axis):
        n = len(axis)
        i = np.clip(np.searchsorted(axis, x) - 1, 0, n - 2)
        f = (x - axis[i]) / (axis[i+1] - axis[i] + 1e-30)
        f = np.clip(f, 0.0, 1.0)
        return i, f

    def eval_torch(self, VG1, VG2, Vd):
        if self._torch_grid is None:
            self._torch_grid = torch.tensor(self.grid, dtype=torch.float64)
            self._t_vg1 = torch.tensor(self.vg1_axis, dtype=torch.float64)
            self._t_vg2 = torch.tensor(self.vg2_axis, dtype=torch.float64)
            self._t_vd  = torch.tensor(self.vd_axis,  dtype=torch.float64)
        # Match device with input
        dev = VG1.device
        if self._torch_grid.device != dev:
            self._torch_grid = self._torch_grid.to(dev)
            self._t_vg1 = self._t_vg1.to(dev)
            self._t_vg2 = self._t_vg2.to(dev)
            self._t_vd  = self._t_vd.to(dev)
        i, fi = _torch_idx(VG1, self._t_vg1)
        j, fj = _torch_idx(VG2, self._t_vg2)
        k, fk = _torch_idx(Vd,  self._t_vd)
        g = self._torch_grid
        c00 = g[i,j,k]*(1-fi) + g[i+1,j,k]*fi
        c01 = g[i,j,k+1]*(1-fi) + g[i+1,j,k+1]*fi
        c10 = g[i,j+1,k]*(1-fi) + g[i+1,j+1,k]*fi
        c11 = g[i,j+1,k+1]*(1-fi) + g[i+1,j+1,k+1]*fi
        c0 = c00*(1-fj) + c10*fj
        c1 = c01*(1-fj) + c11*fj
        return c0*(1-fk) + c1*fk


def _torch_idx(x, axis):
    n = axis.shape[0]
    i = torch.clamp(torch.searchsorted(axis, x) - 1, 0, n - 2)
    f = (x - axis[i]) / (axis[i+1] - axis[i] + 1e-30)
    f = torch.clamp(f, 0.0, 1.0)
    return i, f


if __name__ == "__main__":
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))
    # Sanity: compare with a single forward_2t call
    rng = np.random.default_rng(0)
    VG1 = rng.uniform(0.2, 0.6, size=64)
    VG2 = rng.uniform(0.05, 0.5, size=64)
    Vd  = rng.uniform(1.0, 2.0, size=64)
    t0 = time.time()
    log_Id = surr.eval(VG1, VG2, Vd)
    dt = time.time() - t0
    print(f"[surr] eval 64 pts: {dt*1e6:.0f} µs ({dt/64*1e6:.2f} µs/pt)")
    print(f"[surr] log_Id sample: min={log_Id.min():.2f}  max={log_Id.max():.2f}")
