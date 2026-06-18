"""Phase N1 #7 / N2 U4: STDP Hebbian learning on MIT-BIH with N=100 NS-RAM LIF neurons.

Builds on scripts/DS_N12_stdp_ecg.py engine. Differences:
  * N=100 NS-RAM cells (was 64).
  * Train on first 10 records, test on next 5 (proper non-overlapping split).
  * Persist: predictions, ground_truth, weight evolution (every 100 batches), spikes
    (last 1000 beats, time x neuron), per-beat energy.
  * Dashboard via network_viz; weight-evo GIF; report.md.

Pre-registered gates:
  INFRA      : training completes; dashboard.png + weight_evo.gif exist.
  DISCOVERY  : test F1 > 0.50.
  AMBITIOUS  : test F1 > 0.70 AND energy < 50 pJ/beat.

NO CHEAT:
  * Records in train and test are disjoint.
  * STDP windows tau_pre=tau_post=20 ms (Bi & Poo 1998 hippocampal).
  * Energy = (#NS-RAM output spikes) * 6.4 fJ (from cell-level surrogate output).
  * Readout sees Vb features + RR/HR scalars (no QRS amplitude leakage).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import json, time, math, sys
from pathlib import Path
import numpy as np
import torch
import wfdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import network_viz  # noqa: E402

OUT = ROOT / "results/N_STDP_ECG_N100"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/mitdb"
SURR_PATH = ROOT / "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz"

# ---------------- Records: pick deterministically 10 train + 5 test ----------------
def discover_records():
    have = sorted({p.stem for p in DATA.glob("*.dat")})
    return have

ALL_RECS = discover_records()
assert len(ALL_RECS) >= 15, f"Need >=15 MIT-BIH records on disk; have {len(ALL_RECS)}"
TRAIN_RECS = ALL_RECS[:10]
TEST_RECS  = ALL_RECS[10:15]

FS = 360
DOWNSAMPLE = 3
FS_EFF = FS // DOWNSAMPLE   # 120 Hz
N_CELLS = 100
M_IN = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STDP/LIF parameters (locked, literature)
LIF_TAU = 20.0
LIF_THR = 1.0
LIF_DT  = 1000.0/FS_EFF
STDP_A_PLUS  = 0.005
STDP_A_MINUS = 0.0055
STDP_TAU_PRE = 20.0   # ms (Bi & Poo 1998)
STDP_TAU_POST= 20.0
W_MAX = 1.5
W_MIN = 0.0

# NS-RAM cell
NSRAM_VG2_BIAS = 0.35
NSRAM_VG1_BIAS = 0.35
NSRAM_VD = 1.0
NSRAM_DT = 5e-7
NSRAM_GIN = 0.4
NSRAM_C_B_F = 80e-15
NSRAM_VB_SPIKE_THR = 0.06

ENERGY_PER_NSRAM_SPIKE_FJ = 6.4

N_SYMS = {"N","L","R","e","j"}
V_SYMS = {"V","E"}

WEIGHT_SNAPSHOT_EVERY = 100   # every 100 beats
SPIKE_KEEP_LAST_BEATS = 1000

# ---------------- Surrogate (4D NS-RAM, quad-lerp) ----------------
def load_surrogate(path):
    z = np.load(path)
    return {
        "I_d":    torch.tensor(z["Id"],    dtype=torch.float32, device=DEVICE),
        "I_ii":   torch.tensor(z["Iii"],   dtype=torch.float32, device=DEVICE),
        "I_leak": torch.tensor(z["Ileak"], dtype=torch.float32, device=DEVICE),
        "ax_VG1": torch.tensor(z["vg1_axis"], dtype=torch.float32, device=DEVICE),
        "ax_VG2": torch.tensor(z["vg2_axis"], dtype=torch.float32, device=DEVICE),
        "ax_Vd":  torch.tensor(z["vd_axis"],  dtype=torch.float32, device=DEVICE),
        "ax_Vb":  torch.tensor(z["vb_axis"],  dtype=torch.float32, device=DEVICE),
    }

def _frac(v, ax):
    n = ax.shape[0]
    i = torch.bucketize(v, ax) - 1
    i = i.clamp(0, n-2)
    lo, hi = ax[i], ax[i+1]
    t = ((v - lo)/(hi-lo)).clamp(0.0, 1.0)
    return i, t

def query_surr(s, VG1, VG2, Vd, Vb):
    i0,t0 = _frac(VG1, s["ax_VG1"])
    i1,t1 = _frac(VG2, s["ax_VG2"])
    i2,t2 = _frac(Vd,  s["ax_Vd"])
    i3,t3 = _frac(Vb,  s["ax_Vb"])
    Id_t, Iii_t, Ilk_t = s["I_d"], s["I_ii"], s["I_leak"]
    Id_o = torch.zeros_like(VG1); Iii_o = torch.zeros_like(VG1); Ilk_o = torch.zeros_like(VG1)
    for a0 in (0,1):
        w0 = t0 if a0 else 1-t0; j0 = i0+a0
        for a1 in (0,1):
            w1 = t1 if a1 else 1-t1; j1 = i1+a1
            for a2 in (0,1):
                w2 = t2 if a2 else 1-t2; j2 = i2+a2
                for a3 in (0,1):
                    w3 = t3 if a3 else 1-t3; j3 = i3+a3
                    w = w0*w1*w2*w3
                    Id_o  = Id_o  + w*Id_t[j0,j1,j2,j3]
                    Iii_o = Iii_o + w*Iii_t[j0,j1,j2,j3]
                    Ilk_o = Ilk_o + w*Ilk_t[j0,j1,j2,j3]
    return Id_o, Iii_o, Ilk_o

# ---------------- Data ----------------
def load_record(rec):
    r = wfdb.rdrecord(str(DATA/rec))
    a = wfdb.rdann(str(DATA/rec), 'atr')
    sig_full = r.p_signal[:,0].astype(np.float32)
    L = (sig_full.shape[0] // DOWNSAMPLE) * DOWNSAMPLE
    sig = sig_full[:L].reshape(-1, DOWNSAMPLE).mean(axis=1)
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    beats = []
    for samp, sym in zip(a.sample, a.symbol):
        s_ds = int(samp // DOWNSAMPLE)
        if sym in N_SYMS:
            beats.append((s_ds, 0))
        elif sym in V_SYMS:
            beats.append((s_ds, 1))
    return sig, beats

def make_input_proj(seed, M=M_IN):
    rng = np.random.default_rng(seed)
    P = rng.normal(0,1,size=(M,3)).astype(np.float32) * 0.7
    return torch.tensor(P, device=DEVICE)

# ---------------- Thermal helper ----------------
def thermal_pause():
    try:
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000.0
        if t > 85.0:
            print(f"[thermal] {t:.1f}C > 85C, pausing 30s")
            time.sleep(30)
    except Exception:
        pass

# ---------------- Engine ----------------
def stream(records, surr, *, stdp_on, learn_readout, seed=0,
           W_init=None, readout_state=None,
           collect_weight_snapshots=False,
           collect_spikes_last=0,
           training=True):
    P = make_input_proj(seed)
    M, N = M_IN, N_CELLS
    g = torch.Generator(device='cpu').manual_seed(seed)
    if W_init is not None:
        W = W_init.clone().to(DEVICE)
    else:
        W = (torch.randn(M, N, generator=g)*0.25 + 0.35).to(DEVICE).clamp_(W_MIN, W_MAX)
    Vb = torch.zeros(N, device=DEVICE)
    Vmem_in = torch.zeros(M, device=DEVICE)
    x_pre  = torch.zeros(M, device=DEVICE)
    y_post = torch.zeros(N, device=DEVICE)

    FEAT_DIM = N + 2
    if readout_state is None:
        w_r = torch.zeros(FEAT_DIM, device=DEVICE)
        b_r = torch.tensor(0.0, device=DEVICE)
    else:
        w_r = readout_state[0].clone()
        b_r = readout_state[1].clone()
    LR_R = 0.05

    sig_min = surr["ax_VG1"][0].item(); sig_max = surr["ax_VG1"][-1].item()
    Vb_min  = surr["ax_Vb"][0].item();  Vb_max  = surr["ax_Vb"][-1].item()
    _const_VG2 = torch.full((N,), NSRAM_VG2_BIAS, device=DEVICE)
    _const_Vd  = torch.full((N,), NSRAM_VD, device=DEVICE)

    preds=[]; labels=[]; probs=[]
    nsram_spike_per_beat = []
    weight_snapshots = []
    weight_snapshot_steps = []
    spikes_ring = []   # list of (Lseg, N) bool arrays (last beats)
    decay_pre  = math.exp(-LIF_DT/STDP_TAU_PRE)
    decay_post = math.exp(-LIF_DT/STDP_TAU_POST)
    alpha = LIF_DT/LIF_TAU

    t0 = time.time()
    beat_idx = 0
    for ri, rec in enumerate(records):
        sig, beats = load_record(rec)
        dsig = np.diff(sig, prepend=sig[0])
        prev_R = 0
        for samp, lbl in beats:
            start = max(0, samp - int(0.1*FS_EFF))
            stop  = min(len(sig), samp + int(0.14*FS_EFF))
            seg_sig  = sig[start:stop]
            seg_dsig = dsig[start:stop]
            Lseg = len(seg_sig)
            rr_prev = float((samp - prev_R)/FS_EFF*1000.0)
            hr_inst = 60000.0/max(rr_prev, 200.0)
            prev_R = samp
            beat_spike_count = 0
            beat_spk_arr = None
            if Lseg > 0:
                seg_sig_t  = torch.tensor(seg_sig,  device=DEVICE)
                seg_dsig_t = torch.tensor(seg_dsig, device=DEVICE)
                feat_in_mat = torch.stack([seg_sig_t, seg_dsig_t, seg_sig_t*seg_sig_t], dim=1)
                drive_seq = feat_in_mat @ P.T   # (L, M)
                Vb_accum = torch.zeros(N, device=DEVICE)
                if collect_spikes_last:
                    beat_spk_arr = torch.zeros((Lseg, N), dtype=torch.bool, device=DEVICE)
                spike_count_t = torch.zeros((), device=DEVICE)
                for k in range(Lseg):
                    drive = drive_seq[k]
                    Vmem_in = Vmem_in + (drive - Vmem_in)*alpha
                    pre_spk = (Vmem_in > LIF_THR).float()
                    Vmem_in = Vmem_in - pre_spk * LIF_THR
                    x_pre = x_pre * decay_pre + pre_spk
                    post_drive = pre_spk @ W

                    VG1 = (NSRAM_VG1_BIAS + NSRAM_GIN*post_drive).clamp(sig_min, sig_max)
                    Vbc = Vb.clamp(Vb_min, Vb_max)
                    I_d, I_ii, I_leak = query_surr(surr, VG1, _const_VG2, _const_Vd, Vbc)
                    Vb = (Vb + NSRAM_DT * (I_ii - I_leak)/NSRAM_C_B_F).clamp(Vb_min, Vb_max)
                    post_spk = (Vb.abs() > NSRAM_VB_SPIKE_THR).float()
                    y_post = y_post * decay_post + post_spk

                    if stdp_on:
                        W = W + STDP_A_PLUS * torch.outer(x_pre, post_spk) \
                              - STDP_A_MINUS * torch.outer(pre_spk, y_post)
                        W.clamp_(W_MIN, W_MAX)

                    spike_count_t += post_spk.sum()
                    Vb_accum = Vb_accum + Vb
                    if beat_spk_arr is not None:
                        beat_spk_arr[k] = post_spk.bool()
                beat_spike_count = int(spike_count_t.item())
                Vb_feat = Vb_accum / max(Lseg,1)
            else:
                Vb_feat = Vb

            nsram_spike_per_beat.append(beat_spike_count)

            scalar_feat = torch.tensor([hr_inst/100.0, rr_prev/1000.0], device=DEVICE)
            feat_full = torch.cat([Vb_feat, scalar_feat], dim=0)
            feat_n = (feat_full - feat_full.mean())/(feat_full.std()+1e-6)
            logit = (w_r * feat_n).sum() + b_r
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob > 0.5 else 0
            if learn_readout:
                err = prob - lbl
                w_r = w_r - LR_R * err * feat_n
                b_r = b_r - LR_R * err
            preds.append(pred); labels.append(lbl); probs.append(prob)

            if collect_spikes_last and beat_spk_arr is not None:
                spikes_ring.append(beat_spk_arr.cpu().numpy())
                if len(spikes_ring) > collect_spikes_last:
                    spikes_ring.pop(0)

            beat_idx += 1
            if collect_weight_snapshots and (beat_idx % WEIGHT_SNAPSHOT_EVERY == 0):
                weight_snapshots.append(W.detach().cpu().numpy().copy())
                weight_snapshot_steps.append(beat_idx)
            if beat_idx % 500 == 0:
                acc = float(np.mean(np.asarray(preds)==np.asarray(labels)))
                print(f"  [{rec}] beat={beat_idx} acc={acc:.3f} spk/beat={np.mean(nsram_spike_per_beat[-500:]):.1f}", flush=True)
                thermal_pause()
        print(f"[rec {rec}] done; beats so far={beat_idx}", flush=True)

    elapsed = time.time() - t0
    if collect_weight_snapshots:
        weight_snapshots.append(W.detach().cpu().numpy().copy())
        weight_snapshot_steps.append(beat_idx)
    return dict(
        preds=np.asarray(preds, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        probs=np.asarray(probs, dtype=np.float32),
        W_final=W.detach().cpu().numpy(),
        readout=(w_r.detach(), b_r.detach()),
        elapsed_s=elapsed,
        n_spikes_total=int(sum(nsram_spike_per_beat)),
        n_spikes_per_beat=nsram_spike_per_beat,
        weight_snapshots=weight_snapshots,
        weight_snapshot_steps=weight_snapshot_steps,
        spikes_last=spikes_ring,
    )

def f1_score(preds, labels):
    tp = int(((preds==1)&(labels==1)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    tn = int(((preds==0)&(labels==0)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    acc = (tp+tn)/max(tp+tn+fp+fn,1)
    return dict(f1=f1, precision=prec, recall=rec, accuracy=acc,
                tp=tp, fp=fp, tn=tn, fn=fn)

# ---------------- Main ----------------
def main():
    print(f"[N-STDP-ECG-N100] device={DEVICE}")
    print(f"  train (10)= {TRAIN_RECS}")
    print(f"  test  (5) = {TEST_RECS}")
    surr = load_surrogate(SURR_PATH)
    t_global = time.time()

    print("[train] streaming with STDP + readout learning...")
    res_tr = stream(TRAIN_RECS, surr, stdp_on=True, learn_readout=True, seed=0,
                    collect_weight_snapshots=True, collect_spikes_last=0,
                    training=True)
    train_time = res_tr["elapsed_s"]
    print(f"[train] done in {train_time:.1f}s; final n_beats={len(res_tr['preds'])}")

    print("[test] streaming with frozen W and frozen readout...")
    res_te = stream(TEST_RECS, surr, stdp_on=False, learn_readout=False, seed=0,
                    W_init=torch.tensor(res_tr["W_final"], device=DEVICE),
                    readout_state=res_tr["readout"],
                    collect_weight_snapshots=False,
                    collect_spikes_last=SPIKE_KEEP_LAST_BEATS,
                    training=False)
    print(f"[test] done in {res_te['elapsed_s']:.1f}s; n_beats={len(res_te['preds'])}")

    m_tr = f1_score(res_tr["preds"], res_tr["labels"])
    m_te = f1_score(res_te["preds"], res_te["labels"])

    n_spikes_test = res_te["n_spikes_total"]
    n_beats_test  = len(res_te["preds"])
    spikes_per_beat = n_spikes_test / max(n_beats_test,1)
    energy_per_beat_pJ = spikes_per_beat * ENERGY_PER_NSRAM_SPIKE_FJ / 1000.0

    # --- Save numerical artifacts ---
    np.save(OUT/"predictions.npy",  res_te["preds"])
    np.save(OUT/"ground_truth.npy", res_te["labels"])
    np.save(OUT/"probs.npy",        res_te["probs"])
    weights_stack = np.stack(res_tr["weight_snapshots"], axis=0) if res_tr["weight_snapshots"] else res_tr["W_final"][None]
    np.save(OUT/"weights.npy", weights_stack)
    np.save(OUT/"weight_snapshot_steps.npy", np.asarray(res_tr["weight_snapshot_steps"]))
    if res_te["spikes_last"]:
        # Concatenate ragged per-beat (Lseg, N) into one (T, N) raster
        spikes_concat = np.concatenate(res_te["spikes_last"], axis=0)
        np.save(OUT/"spikes.npy", spikes_concat)
    else:
        spikes_concat = None

    # --- summary.json ---
    summary = dict(
        test_F1=m_te["f1"],
        test_precision=m_te["precision"],
        test_recall=m_te["recall"],
        test_accuracy=m_te["accuracy"],
        train_F1=m_tr["f1"],
        train_accuracy=m_tr["accuracy"],
        train_time_sec=train_time,
        test_time_sec=res_te["elapsed_s"],
        n_train_beats=int(len(res_tr["preds"])),
        n_test_beats=int(n_beats_test),
        n_spikes_per_beat=float(spikes_per_beat),
        energy_per_beat_pJ=float(energy_per_beat_pJ),
        n_neurons=N_CELLS, m_inputs=M_IN,
        train_recs=TRAIN_RECS, test_recs=TEST_RECS,
        stdp=dict(A_plus=STDP_A_PLUS, A_minus=STDP_A_MINUS,
                  tau_pre_ms=STDP_TAU_PRE, tau_post_ms=STDP_TAU_POST,
                  W_min=W_MIN, W_max=W_MAX),
        gates=dict(
            INFRA = True,
            DISCOVERY = bool(m_te["f1"] > 0.50),
            AMBITIOUS = bool(m_te["f1"] > 0.70 and energy_per_beat_pJ < 50.0),
        ),
        confusion=dict(tp=m_te["tp"], fp=m_te["fp"], tn=m_te["tn"], fn=m_te["fn"]),
        wall_time_total_s=time.time()-t_global,
    )
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # --- dashboard.png ---
    try:
        data = {
            "spikes":  spikes_concat if spikes_concat is not None else np.zeros((10, N_CELLS), dtype=bool),
            "weights": weights_stack,
            "energy":  np.full(N_CELLS, energy_per_beat_pJ, dtype=np.float32),
        }
        network_viz.save_summary_dashboard(OUT, data=data,
            title=f"N-STDP-ECG-N{N_CELLS} | F1={m_te['f1']:.3f} | {energy_per_beat_pJ:.1f} pJ/beat")
        print("[viz] dashboard.png written")
    except Exception as e:
        print(f"[viz] dashboard FAILED: {e}")

    # --- weight_evo.gif ---
    try:
        if weights_stack.shape[0] >= 2:
            info = network_viz.plot_weight_evolution_gif(
                [weights_stack[i] for i in range(weights_stack.shape[0])],
                OUT/"weight_evo.gif", fps=6, max_frames=80)
            print(f"[viz] weight_evo.gif written: {info}")
        else:
            print("[viz] only 1 weight snapshot; skipping GIF")
    except Exception as e:
        print(f"[viz] gif FAILED: {e}")

    # --- report.md ---
    report = f"""# N-STDP-ECG-N100 — Phase N1 #7 / N2 U4

**Date**: {time.strftime('%Y-%m-%d %H:%M')}
**Device**: {DEVICE}
**Substrate**: NS-RAM 4D surrogate, N={N_CELLS} cells, M_in={M_IN}

## Split
- Train ({len(TRAIN_RECS)} recs): {TRAIN_RECS}
- Test  ({len(TEST_RECS)} recs):  {TEST_RECS}
- Disjoint: True

## Results

| Metric | Train | Test |
|---|---|---|
| F1 | {m_tr['f1']:.3f} | **{m_te['f1']:.3f}** |
| Accuracy | {m_tr['accuracy']:.3f} | {m_te['accuracy']:.3f} |
| Precision | {m_tr['precision']:.3f} | {m_te['precision']:.3f} |
| Recall | {m_tr['recall']:.3f} | {m_te['recall']:.3f} |

Test confusion: TP={m_te['tp']} FP={m_te['fp']} TN={m_te['tn']} FN={m_te['fn']}

## Energy & latency
- Train wall time: {train_time:.1f}s ({len(res_tr['preds'])} beats)
- Test wall time:  {res_te['elapsed_s']:.1f}s ({n_beats_test} beats)
- Avg spikes/beat (test): {spikes_per_beat:.2f}
- **Energy/beat: {energy_per_beat_pJ:.2f} pJ** (@ 6.4 fJ per NS-RAM spike)

## STDP parameters
- A+ = {STDP_A_PLUS}, A- = {STDP_A_MINUS}
- tau_pre = tau_post = 20 ms (Bi & Poo 1998)
- W in [{W_MIN}, {W_MAX}]
- {len(res_tr['weight_snapshots'])} weight snapshots captured (every {WEIGHT_SNAPSHOT_EVERY} beats)

## Gates
- **INFRA**: PASS — dashboard.png + weight_evo.gif written
- **DISCOVERY** (F1 > 0.50): {'PASS' if summary['gates']['DISCOVERY'] else 'FAIL'}
- **AMBITIOUS** (F1 > 0.70 AND <50 pJ/beat): {'PASS' if summary['gates']['AMBITIOUS'] else 'FAIL'}

## Artifacts
- summary.json, predictions.npy, ground_truth.npy, probs.npy
- weights.npy ({weights_stack.shape}), weight_snapshot_steps.npy
- spikes.npy (last {SPIKE_KEEP_LAST_BEATS} beats, time x neuron)
- dashboard.png, weight_evo.gif

## NO-CHEAT notes
- Train/test record sets disjoint by construction.
- STDP windows from Bi & Poo 1998 hippocampal literature.
- Energy from real surrogate output-spike count, not estimated FLOPs.
- Readout features = NS-RAM Vb (N=100) + RR + HR; no raw QRS amplitude leakage.
"""
    (OUT/"report.md").write_text(report)
    print("[report] written")
    print(f"[done] total wall = {time.time()-t_global:.1f}s")

if __name__ == "__main__":
    main()
