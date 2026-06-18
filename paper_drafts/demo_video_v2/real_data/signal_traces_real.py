"""Render REAL signal traces from captured Phase 12 / 12B / 13 / 14C data.

NO synthesis. Every value plotted is read from JSON or .npz on disk.

Outputs PNGs into ../frames_real/ for ingestion by the video build pipeline.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(REPO, 'results', 'IDENTITY_BENCHMARK_2026-05-30')
P12  = os.path.join(DATA, 'embodiment12')
P12B = os.path.join(DATA, 'embodiment12b')
P13  = os.path.join(DATA, 'embodiment13')
P14C = os.path.join(DATA, 'embodiment14c')
OUT  = os.path.abspath(os.path.join(HERE, '..', 'frames_real'))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0a0a0a',
    'axes.facecolor':   '#0a0a0a',
    'savefig.facecolor':'#0a0a0a',
    'axes.edgecolor':   '#888',
    'axes.labelcolor':  '#ddd',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#eee',
    'font.size':        12,
    'axes.titlesize':   14,
})

C_IKAROS = '#3ab0ff'
C_DAED   = '#ff6b6b'


def _load(path):
    with open(path) as f:
        return json.load(f)


def render_syscall_p999():
    """Channel 1: syscall nanosleep p99.9 tail — REAL raw_samples_ns."""
    di = _load(os.path.join(P12, 'task_D_syscall_ikaros.json'))
    dd = _load(os.path.join(P12, 'task_D_syscall_daedalus.json'))
    si = np.array(di['raw_samples_ns_nanosleep0'], dtype=float)
    sd = np.array(dd['raw_samples_ns_nanosleep0'], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    bins = np.linspace(min(si.min(), sd.min()), np.percentile(np.concatenate([si, sd]), 99.95), 120)
    ax.hist(si, bins=bins, color=C_IKAROS, alpha=0.6, label=f'ikaros (n={len(si)}, p99.9={di["nanosleep0"]["p99_9"]:.0f}ns)')
    ax.hist(sd, bins=bins, color=C_DAED,   alpha=0.6, label=f'daedalus (n={len(sd)}, p99.9={dd["nanosleep0"]["p99_9"]:.0f}ns)')
    ax.set_xlabel('nanosleep(0) latency [ns]')
    ax.set_ylabel('count')
    ax.set_title('Channel 1 — syscall p99.9 tail (Phase 12 task D, REAL)')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    p = os.path.join(OUT, 'sig1_syscall_p999.png')
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def render_nvme_tail():
    """Channel 2: NVMe pread tail latency — REAL raw_samples_ns."""
    di = _load(os.path.join(P12, 'task_F_nvme_ikaros.json'))
    dd = _load(os.path.join(P12, 'task_F_nvme_daedalus.json'))
    si = np.array(di['raw_samples_ns'], dtype=float) / 1e3   # µs
    sd = np.array(dd['raw_samples_ns'], dtype=float) / 1e3
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    hi = np.percentile(np.concatenate([si, sd]), 99.5)
    bins = np.linspace(0, hi, 120)
    ax.hist(si, bins=bins, color=C_IKAROS, alpha=0.6, label=f'ikaros (p99.9={di["nvme_latency"]["p99_9"]/1e3:.1f}µs)')
    ax.hist(sd, bins=bins, color=C_DAED,   alpha=0.6, label=f'daedalus (p99.9={dd["nvme_latency"]["p99_9"]/1e3:.1f}µs)')
    ax.set_xlabel('pread() latency [µs]')
    ax.set_ylabel('count')
    ax.set_title('Channel 2 — NVMe queue tail (Phase 12 task F, REAL)')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    p = os.path.join(OUT, 'sig2_nvme_tail.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def render_rdrand():
    """Channel 3: RDRAND cycle latency — REAL raw_samples_cyc."""
    di = _load(os.path.join(P12, 'task_E_rdrand_ikaros.json'))
    dd = _load(os.path.join(P12, 'task_E_rdrand_daedalus.json'))
    si = np.array(di['raw_samples_cyc'], dtype=float)
    sd = np.array(dd['raw_samples_cyc'], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    hi = np.percentile(np.concatenate([si, sd]), 99)
    bins = np.linspace(min(si.min(), sd.min()), hi, 120)
    ax.hist(si, bins=bins, color=C_IKAROS, alpha=0.6,
            label=f'ikaros (median={di["rdrand_cycles"]["p50"]:.0f} cyc)')
    ax.hist(sd, bins=bins, color=C_DAED,   alpha=0.6,
            label=f'daedalus (median={dd["rdrand_cycles"]["p50"]:.0f} cyc)')
    ax.set_xlabel('RDRAND latency [cycles]')
    ax.set_ylabel('count')
    ax.set_title('Channel 3 — RDRAND latency (Phase 12 task E, REAL)')
    ax.legend(loc='upper right')
    p = os.path.join(OUT, 'sig3_rdrand.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def render_tsc_intercore():
    """Channel 4: inter-core TSC offsets — REAL aggregated stats per pair."""
    di = _load(os.path.join(P12B, 'task_B_ikaros.json'))
    dd = _load(os.path.join(P12B, 'task_B_daedalus.json'))
    pairs = sorted(di['pairs'].keys(), key=lambda x: int(x))
    # Use the same pair set for both hosts
    pairs = [p for p in pairs if p in dd['pairs']]
    means_i = [di['pairs'][p]['mean'] for p in pairs]
    p99_i   = [di['pairs'][p]['p99']  for p in pairs]
    means_d = [dd['pairs'][p]['mean'] for p in pairs]
    p99_d   = [dd['pairs'][p]['p99']  for p in pairs]
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    w = 0.35
    ax.bar(x - w/2, means_i, w, color=C_IKAROS, alpha=0.85, label='ikaros mean')
    ax.bar(x + w/2, means_d, w, color=C_DAED,   alpha=0.85, label='daedalus mean')
    ax.plot(x - w/2, p99_i, 'o', color='white', ms=5, alpha=0.7, label='p99')
    ax.plot(x + w/2, p99_d, 'o', color='white', ms=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C0-C{p}' for p in pairs])
    ax.set_ylabel('TSC pair latency [ns]')
    ax.set_title('Channel 4 — Inter-core TSC offsets (Phase 12B task B, REAL)')
    ax.legend(loc='upper left')
    p = os.path.join(OUT, 'sig4_tsc_intercore.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def render_cacheline_pingpong():
    """Channel 5: cache-line ping-pong matrix — REAL stats per pair."""
    di = _load(os.path.join(P12B, 'task_E_ikaros.json'))
    dd = _load(os.path.join(P12B, 'task_E_daedalus.json'))
    keys = sorted(di['pairs'].keys(), key=lambda k: int(k.split('_')[1]))
    keys = [k for k in keys if k in dd['pairs']]
    means_i = [di['pairs'][k]['mean'] for k in keys]
    means_d = [dd['pairs'][k]['mean'] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    x = np.arange(len(keys))
    ax.plot(x, means_i, 'o-', color=C_IKAROS, lw=2, ms=8, label='ikaros')
    ax.plot(x, means_d, 's-', color=C_DAED,   lw=2, ms=8, label='daedalus')
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('_', '↔C') for k in keys])
    ax.set_ylabel('cache-line bounce mean [ns]')
    ax.set_title('Channel 5 — Cache-line ping-pong (Phase 12B task E, REAL)')
    ax.legend(loc='upper left')
    p = os.path.join(OUT, 'sig5_cacheline.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def render_dram_refresh():
    """Bonus: DRAM refresh fingerprint — REAL stats dict (walk + spike intervals)."""
    di = _load(os.path.join(P12B, 'task_G_ikaros.json'))
    dd = _load(os.path.join(P12B, 'task_G_daedalus.json'))
    wi = di.get('walk_ns', {}) or {}
    wd = dd.get('walk_ns', {}) or {}
    si = di.get('spike_interval_samples', {}) or {}
    sd = dd.get('spike_interval_samples', {}) or {}
    if not isinstance(wi, dict) or not isinstance(wd, dict) or not wi or not wd:
        return None
    stat_keys = [k for k in ('p50','p90','p99','p99_9','mean','std') if k in wi and k in wd]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    x = np.arange(len(stat_keys))
    w = 0.4
    axL.bar(x - w/2, [wi[k] for k in stat_keys], w, color=C_IKAROS, label='ikaros walk')
    axL.bar(x + w/2, [wd[k] for k in stat_keys], w, color=C_DAED,   label='daedalus walk')
    axL.set_xticks(x); axL.set_xticklabels(stat_keys)
    axL.set_ylabel('walk latency [ns]')
    axL.set_title(f'DRAM walk stats (ikaros spikes={di.get("spike_count","?")}, '
                  f'daedalus spikes={dd.get("spike_count","?")})')
    axL.legend(loc='upper left')

    if isinstance(si, dict) and isinstance(sd, dict) and si and sd:
        sk = [k for k in ('p50','p90','p99','mean') if k in si and k in sd]
        x2 = np.arange(len(sk))
        axR.bar(x2 - w/2, [si[k] for k in sk], w, color=C_IKAROS, label='ikaros spike-int')
        axR.bar(x2 + w/2, [sd[k] for k in sk], w, color=C_DAED,   label='daedalus spike-int')
        axR.set_xticks(x2); axR.set_xticklabels(sk)
        axR.set_ylabel('spike interval [samples]')
        axR.set_title('DRAM refresh spike-interval stats')
        axR.legend(loc='upper left')

    fig.suptitle('Channel 6 — DRAM refresh probing (Phase 12B task G, REAL)', color='#eee')
    p = os.path.join(OUT, 'sig6_dram_refresh.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def render_phase13_signatures():
    """Aggregated Phase 13 signatures — REAL 10×290 vectors, plotted as
    parallel-coords trace per host so the audience sees the actual fingerprint
    that the embodied model consumes."""
    zi = np.load(os.path.join(P13, 'ikaros_sig_v2.npz'))
    zd = np.load(os.path.join(P13, 'daedalus_sig_v2.npz'))
    vi = zi['vec']
    vd = zd['vec']
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    x = np.arange(vi.shape[1])
    for r in range(vi.shape[0]):
        ax.plot(x, vi[r], color=C_IKAROS, lw=0.6, alpha=0.4)
    for r in range(vd.shape[0]):
        ax.plot(x, vd[r], color=C_DAED, lw=0.6, alpha=0.4)
    ax.plot([], [], color=C_IKAROS, lw=2, label=f'ikaros ({vi.shape[0]} reps)')
    ax.plot([], [], color=C_DAED,   lw=2, label=f'daedalus ({vd.shape[0]} reps)')
    ax.set_xlabel('signature feature index')
    ax.set_ylabel('normalized value')
    ax.set_title('Phase 13 — Aggregated identity signatures (REAL, 290-dim)')
    ax.legend(loc='upper right')
    p = os.path.join(OUT, 'sig_phase13_fingerprint.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def render_identity_panel():
    """Section 2 / 3: pre vs post identity calls — REAL captured samples."""
    d = _load(os.path.join(HERE, 'identity_samples.json'))
    pre  = d['pre']
    post = d['post']
    pre_emb_correct = sum(1 for x in pre  if x['embodied']['correct'])
    post_emb_correct = sum(1 for x in post if x['embodied']['correct'])
    pre_van_correct = sum(1 for x in pre  if x['vanilla']['correct'])
    post_van_correct = sum(1 for x in post if x['vanilla']['correct'])
    pre_conf  = [x['embodied']['confidence'] for x in pre]
    post_conf = [x['embodied']['confidence'] for x in post]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    # Left: bar accuracy
    cats = ['embodied\nPRE', 'embodied\nPOST', 'vanilla\nPRE', 'vanilla\nPOST']
    vals = [100*pre_emb_correct/len(pre), 100*post_emb_correct/len(post),
            100*pre_van_correct/len(pre), 100*post_van_correct/len(post)]
    colors = [C_IKAROS, C_DAED, '#888', '#666']
    bars = axL.bar(cats, vals, color=colors)
    for b, v in zip(bars, vals):
        axL.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}%',
                 ha='center', color='white', fontsize=11, fontweight='bold')
    axL.set_ylim(0, 110)
    axL.set_ylabel('identity correct [%]')
    axL.set_title(f'Identity accuracy — REAL ({len(pre)} pre + {len(post)} post calls)')
    axL.axhline(50, color='#444', ls='--', lw=0.8)

    # Right: confidence trace
    axR.plot(pre_conf,  'o-', color=C_IKAROS, lw=1.5, ms=4, label=f'PRE  μ={np.mean(pre_conf):.3f}')
    axR.plot(np.arange(len(post_conf))+len(pre_conf), post_conf, 's-', color=C_DAED, lw=1.5, ms=4,
             label=f'POST μ={np.mean(post_conf):.3f}')
    axR.axvline(len(pre_conf)-0.5, color='#ffaa00', ls='--', lw=2, label='transplant')
    axR.set_xlabel('call #')
    axR.set_ylabel('embodied confidence')
    axR.set_ylim(0, 1.05)
    axR.set_title('Confidence stays HIGH but flips host (real)')
    axR.legend(loc='lower right')

    p = os.path.join(OUT, 'identity_pre_post_real.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def render_spoof_bars():
    """Section 4: spoof attack bars from REAL Phase 14C results."""
    d = _load(os.path.join(P14C, 'ikaros_spoof_v2.json'))
    atks = d['attacks']
    # Pick canonical attack set, accept rate
    order = ['honest_own', 'daedalus_peer', 'static_replay_no_nonce',
             'permute_replay', 'gaussian_proxy', 'mean_proxy', 'flip_proxy']
    labels = {
        'honest_own': 'honest\n(self)',
        'daedalus_peer': 'peer\n(daedalus)',
        'static_replay_no_nonce': 'static\nreplay',
        'permute_replay': 'permute\nreplay',
        'gaussian_proxy': 'gaussian\nproxy',
        'mean_proxy': 'mean\nproxy',
        'flip_proxy': 'flip\nproxy',
    }
    keys = [k for k in order if k in atks]
    accept = [atks[k].get('accept_rate', atks[k].get('classifier_accept_only', 0)) * 100 for k in keys]
    cats = [labels.get(k, k) for k in keys]
    colors = [C_IKAROS if k=='honest_own' else C_DAED for k in keys]
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
    bars = ax.bar(cats, accept, color=colors)
    for b, v in zip(bars, accept):
        ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%',
                ha='center', color='white', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_ylabel('accept rate [%]')
    ax.set_title('Phase 14C — Spoof attack accept rates (REAL, n_eval='+str(d.get('n_eval','?'))+')')
    ax.axhline(50, color='#444', ls='--', lw=0.8)
    p = os.path.join(OUT, 'spoof_bars_real.png')
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    return p


def main():
    out = {}
    for name, fn in [
        ('sig1_syscall_p999',       render_syscall_p999),
        ('sig2_nvme_tail',          render_nvme_tail),
        ('sig3_rdrand',             render_rdrand),
        ('sig4_tsc_intercore',      render_tsc_intercore),
        ('sig5_cacheline',          render_cacheline_pingpong),
        ('sig6_dram_refresh',       render_dram_refresh),
        ('sig_phase13_fingerprint', render_phase13_signatures),
        ('identity_pre_post_real',  render_identity_panel),
        ('spoof_bars_real',         render_spoof_bars),
    ]:
        try:
            p = fn()
            out[name] = p
            print(f'OK   {name:30s} -> {p}')
        except Exception as e:
            out[name] = f'ERR {e}'
            print(f'FAIL {name:30s} -> {e}')
    manifest = os.path.join(HERE, 'manifest.json')
    with open(manifest, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nmanifest -> {manifest}')


if __name__ == '__main__':
    main()
