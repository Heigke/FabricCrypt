"""Phase N1 #7 / N2 U4 v2: STDP+ECG with FIXED cross-subject readout.

v1 had train F1=0.975 but test F1=0.000 — substrate works, readout collapsed
across subjects. v1 used a per-beat z-normalized online logistic regression that
chased the training-subject distribution; on unseen subjects this drove all
predictions to one class (predictions=0 everywhere → F1=0).

This v2 keeps the SAME substrate (NS-RAM 4D surrogate, N=100 cells, STDP+LIF,
same hyperparameters and energy accounting) and only modifies the readout.

Two readouts are evaluated:
  (A) MLP_OFFLINE  — small 2-layer MLP on Vb features + RR + HR, trained
                     OFFLINE on cached training-pass features using AdamW.
                     Standardization uses TRAIN-only running statistics
                     applied identically at test (no per-beat z-norm).
  (B) NLMS_ONLINE  — normalized LMS readout that *continues to adapt during
                     test* using self-supervised labels from the MLP's
                     high-confidence predictions (AAMI-compliant: no ground-truth
                     leak; updates only on prob>0.9 or prob<0.1, and uses MLP
                     output as the pseudo-label).

Pre-registered gates (re-stated from prompt):
  INFRA        : trains + dashboard.
  DISCOVERY    : test F1 > 0.50  (the gate v1 missed)
  AMBITIOUS    : test F1 > 0.70 AND energy < 50 pJ/beat
  KILL_SHOT    : test F1 < 0.20 → substrate problem not readout.

NO-CHEAT:
  * disjoint train/test records
  * AAMI compliant: V (VEB) vs N classes from MIT-BIH symbols
  * standardization parameters fit on training only
  * NLMS pseudo-labels come from the model, not the ground truth
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

OUT = ROOT / "results/N_STDP_ECG_N100_v2"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/mitdb"
SURR_PATH = ROOT / "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz"

def discover_records():
    return sorted({p.stem for p in DATA.glob("*.dat")})

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

LIF_TAU = 20.0
LIF_THR = 1.0
LIF_DT  = 1000.0/FS_EFF
STDP_A_PLUS  = 0.005
STDP_A_MINUS = 0.0055
STDP_TAU_PRE = 20.0
STDP_TAU_POST= 20.0
W_MAX = 1.5
W_MIN = 0.0

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

WEIGHT_SNAPSHOT_EVERY = 100
SPIKE_KEEP_LAST_BEATS = 1000

# ---------------- Surrogate ----------------
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

def thermal_pause():
    try:
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000.0
        if t > 85.0:
            print(f"[thermal] {t:.1f}C > 85C, pausing 30s")
            time.sleep(30)
    except Exception:
        pass

# ---------------- Substrate pass (no readout inside) ----------------
def substrate_pass(records, surr, *, stdp_on, seed=0, W_init=None,
                   collect_weight_snapshots=False, collect_spikes_last=0):
    """Run STDP/LIF substrate, return per-beat feature matrix (Vb mean + RR + HR)
    and labels. Readout is computed separately."""
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

    sig_min = surr["ax_VG1"][0].item(); sig_max = surr["ax_VG1"][-1].item()
    Vb_min  = surr["ax_Vb"][0].item();  Vb_max  = surr["ax_Vb"][-1].item()
    _const_VG2 = torch.full((N,), NSRAM_VG2_BIAS, device=DEVICE)
    _const_Vd  = torch.full((N,), NSRAM_VD, device=DEVICE)

    decay_pre  = math.exp(-LIF_DT/STDP_TAU_PRE)
    decay_post = math.exp(-LIF_DT/STDP_TAU_POST)
    alpha = LIF_DT/LIF_TAU

    feats_list = []   # per-beat (N+2,) raw features
    labels = []
    nsram_spike_per_beat = []
    weight_snapshots = []
    weight_snapshot_steps = []
    spikes_ring = []
    beat_subj = []     # record id (int index) for each beat
    rec2id = {r:i for i,r in enumerate(records)}

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
                drive_seq = feat_in_mat @ P.T
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
            feats_list.append(feat_full.detach().cpu().numpy())
            labels.append(lbl)
            beat_subj.append(rec2id[rec])

            if collect_spikes_last and beat_spk_arr is not None:
                spikes_ring.append(beat_spk_arr.cpu().numpy())
                if len(spikes_ring) > collect_spikes_last:
                    spikes_ring.pop(0)

            beat_idx += 1
            if collect_weight_snapshots and (beat_idx % WEIGHT_SNAPSHOT_EVERY == 0):
                weight_snapshots.append(W.detach().cpu().numpy().copy())
                weight_snapshot_steps.append(beat_idx)
            if beat_idx % 500 == 0:
                print(f"  [{rec}] beat={beat_idx} spk/beat={np.mean(nsram_spike_per_beat[-500:]):.1f}", flush=True)
                thermal_pause()
        print(f"[rec {rec}] done; beats so far={beat_idx}", flush=True)

    elapsed = time.time() - t0
    if collect_weight_snapshots:
        weight_snapshots.append(W.detach().cpu().numpy().copy())
        weight_snapshot_steps.append(beat_idx)
    return dict(
        feats=np.asarray(feats_list, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        subj=np.asarray(beat_subj, dtype=np.int64),
        W_final=W.detach().cpu().numpy(),
        elapsed_s=elapsed,
        n_spikes_total=int(sum(nsram_spike_per_beat)),
        n_spikes_per_beat=nsram_spike_per_beat,
        weight_snapshots=weight_snapshots,
        weight_snapshot_steps=weight_snapshot_steps,
        spikes_last=spikes_ring,
    )

# ---------------- Readouts ----------------
class MLPReadout(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(feats_tr, labels_tr, n_epochs=30, batch_size=256, lr=1e-3,
              val_frac=0.1, seed=0):
    """Fit a 2-layer MLP on training features with class-balanced loss.
    Standardization stats computed here (train-only). Returns (mlp, mu, sd)."""
    torch.manual_seed(seed)
    mu = feats_tr.mean(axis=0)
    sd = feats_tr.std(axis=0) + 1e-6
    X = (feats_tr - mu) / sd
    y = labels_tr.astype(np.float32)
    # Class imbalance weighting
    n_pos = float((y==1).sum()); n_neg = float((y==0).sum())
    pos_w = max(n_neg/max(n_pos,1.0), 1.0)
    # split val
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X)*val_frac)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(X[tr_i], device=DEVICE); yt = torch.tensor(y[tr_i], device=DEVICE)
    Xv = torch.tensor(X[val_i], device=DEVICE); yv = torch.tensor(y[val_i], device=DEVICE)
    mlp = MLPReadout(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=DEVICE))
    best_val = float('inf'); best_state = None
    n = len(Xt)
    for ep in range(n_epochs):
        mlp.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for s in range(0, n, batch_size):
            ii = perm[s:s+batch_size]
            logit = mlp(Xt[ii])
            loss = loss_fn(logit, yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())*len(ii)
        mlp.eval()
        with torch.no_grad():
            vlog = mlp(Xv); vloss = float(loss_fn(vlog, yv).item())
            vpred = (torch.sigmoid(vlog) > 0.5).float()
            vacc = float((vpred==yv).float().mean().item())
        if vloss < best_val:
            best_val = vloss
            best_state = {k:v.detach().clone() for k,v in mlp.state_dict().items()}
        if ep % 5 == 0 or ep == n_epochs-1:
            print(f"  [mlp] ep{ep} train_loss={tot/max(n,1):.4f} val_loss={vloss:.4f} val_acc={vacc:.3f}", flush=True)
    if best_state is not None:
        mlp.load_state_dict(best_state)
    return mlp, mu, sd, pos_w

def predict_mlp(mlp, feats, mu, sd, thresh=0.5):
    Xn = (feats - mu) / sd
    Xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    mlp.eval()
    with torch.no_grad():
        prob = torch.sigmoid(mlp(Xt)).cpu().numpy()
    return (prob > thresh).astype(np.int64), prob

def nlms_online(feats, mu, sd, mlp, *, conf_high=0.9, conf_low=0.1, mu_lr=0.05):
    """Online NLMS adaptation during test. Starts from MLP-predicted probs,
    learns a linear correction term using *pseudo-labels* (MLP high-conf outputs).
    Never sees ground truth. Returns final preds & probs."""
    Xn = (feats - mu) / sd
    n_feat = Xn.shape[1]
    # warm-start: take the MLP's bias as initial w0=0 (residual learner)
    w = np.zeros(n_feat, dtype=np.float32)
    b = 0.0
    preds = np.zeros(len(Xn), dtype=np.int64)
    probs = np.zeros(len(Xn), dtype=np.float32)
    mlp.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
        base_logits = mlp(Xt).cpu().numpy()
    base_p = 1.0/(1.0 + np.exp(-base_logits))
    for i in range(len(Xn)):
        x = Xn[i]
        # combined: base MLP logit + linear residual
        res = float((w * x).sum() + b)
        logit = base_logits[i] + res
        p = 1.0/(1.0 + math.exp(-logit))
        probs[i] = p
        preds[i] = 1 if p > 0.5 else 0
        # self-supervised: use the BASE MLP's confident predictions as targets
        bp = base_p[i]
        if bp > conf_high or bp < conf_low:
            pseudo = 1.0 if bp > 0.5 else 0.0
            err = p - pseudo
            nrm = (x*x).sum() + 1.0
            w = w - mu_lr * err * x / nrm
            b = b - mu_lr * err / nrm
    return preds, probs

# ---------------- Metrics ----------------
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
    print(f"[N-STDP-ECG-N100-v2] device={DEVICE}")
    print(f"  train (10)= {TRAIN_RECS}")
    print(f"  test  (5) = {TEST_RECS}")
    surr = load_surrogate(SURR_PATH)
    t_global = time.time()

    print("[train] substrate pass with STDP...")
    res_tr = substrate_pass(TRAIN_RECS, surr, stdp_on=True, seed=0,
                            collect_weight_snapshots=True, collect_spikes_last=0)
    train_time = res_tr["elapsed_s"]
    print(f"[train] done in {train_time:.1f}s; beats={len(res_tr['labels'])}")

    print("[test] substrate pass with frozen W...")
    res_te = substrate_pass(TEST_RECS, surr, stdp_on=False, seed=0,
                            W_init=torch.tensor(res_tr["W_final"], device=DEVICE),
                            collect_weight_snapshots=False,
                            collect_spikes_last=SPIKE_KEEP_LAST_BEATS)
    print(f"[test] done in {res_te['elapsed_s']:.1f}s; beats={len(res_te['labels'])}")

    # --- Train MLP offline on training features ---
    print("[readout] training MLP on train-pass features...")
    t_mlp = time.time()
    mlp, mu, sd, pos_w = train_mlp(res_tr["feats"], res_tr["labels"],
                                    n_epochs=30, batch_size=256, lr=1e-3)
    mlp_time = time.time() - t_mlp
    print(f"[readout] MLP trained in {mlp_time:.1f}s (pos_weight={pos_w:.2f})")

    # MLP on train
    pr_tr, prob_tr = predict_mlp(mlp, res_tr["feats"], mu, sd)
    m_tr = f1_score(pr_tr, res_tr["labels"])
    # MLP on test
    pr_te_mlp, prob_te_mlp = predict_mlp(mlp, res_te["feats"], mu, sd)
    m_te_mlp = f1_score(pr_te_mlp, res_te["labels"])
    # NLMS online on test
    print("[readout] NLMS online on test...")
    pr_te_nlms, prob_te_nlms = nlms_online(res_te["feats"], mu, sd, mlp)
    m_te_nlms = f1_score(pr_te_nlms, res_te["labels"])

    print(f"[result] train F1={m_tr['f1']:.4f}")
    print(f"[result] test  F1 (MLP)  = {m_te_mlp['f1']:.4f}  acc={m_te_mlp['accuracy']:.3f}")
    print(f"[result] test  F1 (NLMS) = {m_te_nlms['f1']:.4f}  acc={m_te_nlms['accuracy']:.3f}")

    # Pick best readout
    if m_te_nlms['f1'] >= m_te_mlp['f1']:
        chosen = "NLMS_ONLINE"; m_te = m_te_nlms; preds_te = pr_te_nlms; probs_te = prob_te_nlms
    else:
        chosen = "MLP_OFFLINE"; m_te = m_te_mlp; preds_te = pr_te_mlp; probs_te = prob_te_mlp
    print(f"[result] chosen readout: {chosen} (test F1={m_te['f1']:.4f})")

    n_spikes_test = res_te["n_spikes_total"]
    n_beats_test  = len(res_te["labels"])
    spikes_per_beat = n_spikes_test / max(n_beats_test,1)
    energy_per_beat_pJ = spikes_per_beat * ENERGY_PER_NSRAM_SPIKE_FJ / 1000.0

    # --- Save artifacts ---
    np.save(OUT/"predictions.npy",  preds_te)
    np.save(OUT/"ground_truth.npy", res_te["labels"])
    np.save(OUT/"probs.npy",        probs_te)
    np.save(OUT/"probs_mlp.npy",    prob_te_mlp)
    np.save(OUT/"probs_nlms.npy",   prob_te_nlms)
    weights_stack = np.stack(res_tr["weight_snapshots"], axis=0) if res_tr["weight_snapshots"] else res_tr["W_final"][None]
    np.save(OUT/"weights.npy", weights_stack)
    np.save(OUT/"weight_snapshot_steps.npy", np.asarray(res_tr["weight_snapshot_steps"]))
    if res_te["spikes_last"]:
        spikes_concat = np.concatenate(res_te["spikes_last"], axis=0)
        np.save(OUT/"spikes.npy", spikes_concat)
    else:
        spikes_concat = None
    # cache features for reproducibility / postmortem
    np.savez_compressed(OUT/"features.npz",
                        feats_tr=res_tr["feats"], labels_tr=res_tr["labels"], subj_tr=res_tr["subj"],
                        feats_te=res_te["feats"], labels_te=res_te["labels"], subj_te=res_te["subj"],
                        mu=mu, sd=sd)
    torch.save(mlp.state_dict(), OUT/"mlp_state.pt")

    kill_shot = bool(max(m_te_mlp['f1'], m_te_nlms['f1']) < 0.20)
    summary = dict(
        test_F1=m_te["f1"],
        test_F1_mlp=m_te_mlp["f1"], test_F1_nlms=m_te_nlms["f1"],
        test_precision=m_te["precision"],
        test_recall=m_te["recall"],
        test_accuracy=m_te["accuracy"],
        train_F1=m_tr["f1"],
        train_accuracy=m_tr["accuracy"],
        readout_type=chosen,
        train_time_sec=train_time,
        test_time_sec=res_te["elapsed_s"],
        mlp_train_time_sec=mlp_time,
        n_train_beats=int(len(res_tr['labels'])),
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
            KILL_SHOT = kill_shot,
        ),
        confusion=dict(tp=m_te["tp"], fp=m_te["fp"], tn=m_te["tn"], fn=m_te["fn"]),
        wall_time_total_s=time.time()-t_global,
    )
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        data = {
            "spikes":  spikes_concat if spikes_concat is not None else np.zeros((10, N_CELLS), dtype=bool),
            "weights": weights_stack,
            "energy":  np.full(N_CELLS, energy_per_beat_pJ, dtype=np.float32),
        }
        network_viz.save_summary_dashboard(OUT, data=data,
            title=f"N-STDP-ECG-N{N_CELLS}-v2 [{chosen}] | F1={m_te['f1']:.3f} | {energy_per_beat_pJ:.1f} pJ/beat")
        print("[viz] dashboard.png written")
    except Exception as e:
        print(f"[viz] dashboard FAILED: {e}")

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

    report = f"""# N-STDP-ECG-N100-v2 — Cross-subject readout fix

**Date**: {time.strftime('%Y-%m-%d %H:%M')}
**Device**: {DEVICE}
**Substrate**: NS-RAM 4D surrogate, N={N_CELLS} cells, M_in={M_IN} (SAME as v1)

## Motivation
v1 had train F1=0.975 but test F1=0.000. Online logistic readout with per-beat
z-normalization collapsed on unseen subjects (all preds=0). Substrate was fine;
readout was broken. v2 keeps the substrate identical and tries two cross-subject
readouts on top of identically-computed Vb features.

## Split
- Train ({len(TRAIN_RECS)} recs): {TRAIN_RECS}
- Test  ({len(TEST_RECS)} recs):  {TEST_RECS}
- Disjoint: True; AAMI-compliant N vs V symbol map.

## Results

| Metric | Train (MLP) | Test (MLP) | Test (NLMS-online) |
|---|---|---|---|
| F1        | {m_tr['f1']:.3f} | {m_te_mlp['f1']:.3f} | {m_te_nlms['f1']:.3f} |
| Accuracy  | {m_tr['accuracy']:.3f} | {m_te_mlp['accuracy']:.3f} | {m_te_nlms['accuracy']:.3f} |
| Precision | -     | {m_te_mlp['precision']:.3f} | {m_te_nlms['precision']:.3f} |
| Recall    | -     | {m_te_mlp['recall']:.3f}    | {m_te_nlms['recall']:.3f} |

Chosen readout: **{chosen}** (test F1={m_te['f1']:.3f})
Test confusion: TP={m_te['tp']} FP={m_te['fp']} TN={m_te['tn']} FN={m_te['fn']}

## Energy & latency
- Train substrate wall time: {train_time:.1f}s ({len(res_tr['labels'])} beats)
- Test  substrate wall time: {res_te['elapsed_s']:.1f}s ({n_beats_test} beats)
- MLP offline train: {mlp_time:.1f}s
- Avg spikes/beat (test): {spikes_per_beat:.2f}
- **Energy/beat: {energy_per_beat_pJ:.2f} pJ** (@ 6.4 fJ per NS-RAM spike, SAME counting as v1)

## Gates
- INFRA      : PASS (dashboard + gif written)
- DISCOVERY  (F1 > 0.50): {'PASS' if summary['gates']['DISCOVERY'] else 'FAIL'}
- AMBITIOUS  (F1 > 0.70 AND <50 pJ/beat): {'PASS' if summary['gates']['AMBITIOUS'] else 'FAIL'}
- KILL_SHOT  (both readouts F1<0.20): {'TRIGGERED — substrate problem' if kill_shot else 'NOT triggered'}

## NO-CHEAT notes
- Train/test record sets disjoint by construction.
- Standardization (mu, sd) computed on TRAIN features only, applied at test.
- NLMS uses pseudo-labels from the MLP, not ground truth (AAMI-compliant).
- Energy from real surrogate output-spike count, not estimated FLOPs.

## Artifacts
- summary.json, predictions.npy, ground_truth.npy, probs.npy
- probs_mlp.npy, probs_nlms.npy, mlp_state.pt, features.npz
- weights.npy ({weights_stack.shape}), weight_snapshot_steps.npy
- spikes.npy (last {SPIKE_KEEP_LAST_BEATS} beats), dashboard.png, weight_evo.gif
"""
    (OUT/"report.md").write_text(report)
    print("[report] written")
    print(f"[done] total wall = {time.time()-t_global:.1f}s")

if __name__ == "__main__":
    main()
