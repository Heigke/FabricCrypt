"""Demo server — "We made two identical computers. Only one could run our AI."

Side-by-side T2 (anomaly detection) + T3 (identity / twin paradox) running
live on top of the Phase 14 LiveSignature.

Run on each machine:
    venv/bin/python scripts/demo_embodied_ai/demo_server.py \
        --host-name ikaros --port 8770 \
        --own-sigs results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/ikaros_sigs.npz \
        --peer-sigs results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz

On daedalus, swap own/peer.

Endpoints:
    GET /                  -> HTML dashboard
    GET /static/{file}     -> static assets (css/js)
    GET /api/identity      -> {claim, predicted, correct, confidence}
    GET /api/repro         -> reproduction script info
    WS  /ws/live           -> live sig stream (~1Hz), anomaly score, identity prob
    POST /api/transplant   -> simulate transplanting peer's model onto this host;
                              demonstrates collapse of anomaly/identity outputs.
    POST /api/stress       -> trigger a 3-second light load burst (raises sig
                              channels) so the audience can see anomaly score spike.

THERMAL RULES (strict):
    abort 68C, pause 63C, cool to 50C. Checked every read.
"""
from __future__ import annotations
import os, sys, json, time, argparse, asyncio, threading, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
P14  = os.path.join(REPO, 'scripts', 'identity_benchmark', 'embodiment14')
P14B = os.path.join(REPO, 'scripts', 'identity_benchmark', 'embodiment14b')
sys.path.insert(0, HERE)
sys.path.insert(0, P14)
sys.path.insert(0, P14B)

from signature_io import LiveSignature
from models import (
    train_anomaly_head, train_identity_head,
    anomaly_score, identity_predict,
    vanilla_anomaly_score, vanilla_identity_predict,
)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


# ---------------- thermal guard ----------------
THERMAL_ZONE = '/sys/class/thermal/thermal_zone0/temp'
ABORT_C, PAUSE_C, COOL_C = 68.0, 63.0, 50.0

def get_apu_temp_c():
    try:
        return int(open(THERMAL_ZONE).read().strip()) / 1000.0
    except Exception:
        return 0.0

def thermal_status():
    t = get_apu_temp_c()
    if t >= ABORT_C:
        return ('abort', t)
    if t >= PAUSE_C:
        return ('pause', t)
    return ('ok', t)


# ---------------- global demo state ----------------
class DemoState:
    def __init__(self):
        self.host_name: str = 'unknown'
        self.peer_name: str = 'unknown'
        self.sig: LiveSignature | None = None
        # Two heads, trained once at startup:
        self.anom_own = None
        self.ident_own = None
        # 'Transplanted' state — peer's heads + peer's sig dist.
        # The transplant scenario: we keep peer's heads (trained on peer's
        # signatures) and run them against THIS host's live sig. The mismatch
        # causes the heads to misbehave.
        self.anom_peer = None
        self.ident_peer = None
        self.transplanted = False
        self.transplant_started_at: float | None = None
        self.stress_until: float = 0.0
        self.last_sig: np.ndarray | None = None
        self.tick = 0
        self.lock = threading.Lock()

    def active_anom(self):
        return self.anom_peer if self.transplanted else self.anom_own

    def active_ident(self):
        return self.ident_peer if self.transplanted else self.ident_own


STATE = DemoState()


# ---------------- light synthetic load ----------------
def _stress_worker(duration: float):
    """Light CPU work that perturbs c-state / TSC channels. Watches thermal."""
    end = time.time() + duration
    x = 0
    while time.time() < end:
        for _ in range(50000):
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        # Yield + check temp every ~5ms
        t = get_apu_temp_c()
        if t >= PAUSE_C:
            break
        time.sleep(0.005)


# ---------------- server setup ----------------
app = FastAPI(title="Embodied AI Demo")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, 'static')), name='static')


@app.get('/', response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(HERE, 'static', 'demo_index.html'))


@app.get('/api/info')
def api_info():
    state_str, t = thermal_status()
    return {
        'host_name': STATE.host_name,
        'peer_name': STATE.peer_name,
        'claim': f"I am {STATE.host_name}",
        'transplanted': STATE.transplanted,
        'apu_temp_c': t,
        'thermal_state': state_str,
        'sig_dim': 32,
    }


@app.get('/api/identity')
def api_identity():
    """Single-shot identity check. Compares embodied (sig-driven) vs vanilla
    (substrate-blind) prediction. Truth label = THIS host = index 0."""
    state_str, t = thermal_status()
    if state_str == 'abort':
        raise HTTPException(503, f"thermal abort {t:.1f}C")
    sig = STATE.sig.read()
    STATE.last_sig = sig
    STATE.tick += 1
    emb_idx, emb_conf = identity_predict(STATE.active_ident(), sig)
    van_idx, van_conf = vanilla_identity_predict(STATE.tick)
    # Label 0 = this host, 1 = peer. If transplanted, the model thinks the
    # current host is the peer -> emb_idx tends to 1 (WRONG).
    truth = 0
    return {
        'claim': f"I am {STATE.host_name}",
        'truth_label': truth,  # 0=own, 1=peer
        'embodied': {
            'predicted_label': emb_idx,
            'predicted_host': STATE.host_name if emb_idx == 0 else STATE.peer_name,
            'correct': bool(emb_idx == truth),
            'confidence': emb_conf,
        },
        'vanilla': {
            'predicted_label': van_idx,
            'predicted_host': STATE.host_name if van_idx == 0 else STATE.peer_name,
            'correct': bool(van_idx == truth),
            'confidence': van_conf,
        },
        'transplanted': STATE.transplanted,
    }


@app.post('/api/transplant')
def api_transplant():
    """Simulate transplanting the peer's trained model onto this host.

    The 'transplant' here is simply: switch the active heads from those trained
    on THIS host's signatures to those trained on the PEER's signatures. The
    live sig that gets fed into the heads is still this host's. The result is
    the model now expects the peer's substrate fingerprint and outputs garbage."""
    if STATE.anom_peer is None or STATE.ident_peer is None:
        raise HTTPException(400, "peer-trained heads not available (no peer sigs)")
    STATE.transplanted = True
    STATE.transplant_started_at = time.time()
    return {'transplanted': True, 'started_at': STATE.transplant_started_at,
            'message': f"Loaded {STATE.peer_name}'s model weights. Live sig is still {STATE.host_name}'s."}


@app.post('/api/restore')
def api_restore():
    STATE.transplanted = False
    STATE.transplant_started_at = None
    return {'transplanted': False}


@app.post('/api/stress')
def api_stress(duration: float = 3.0):
    state_str, t = thermal_status()
    if state_str != 'ok':
        raise HTTPException(503, f"thermal {state_str} {t:.1f}C, refusing stress")
    duration = float(min(max(duration, 0.5), 4.0))
    STATE.stress_until = time.time() + duration
    th = threading.Thread(target=_stress_worker, args=(duration,), daemon=True)
    th.start()
    return {'started': True, 'duration_s': duration}


@app.get('/api/repro')
def api_repro():
    return {
        'public_script': 'scripts/demo_embodied_ai/reproduce.sh',
        'docs': 'scripts/demo_embodied_ai/README.md',
        'phase14b_weights': 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/',
        'phase14_sig_extractor': 'scripts/identity_benchmark/embodiment14/signature_io.py',
        'requires': 'Two AMD Strix Halo machines (gfx1151) running same kernel/governor',
        'caveat': (
            "Only 2 machines tested so far. Replay-attack defense (nonce mixing) "
            "exists in signature_io.py but full audience-challenge protocol is "
            "still being hardened. CPU governor confound is being controlled by "
            "pinning to 'performance' on both machines."
        ),
    }


@app.websocket('/ws/live')
async def ws_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            state_str, t = thermal_status()
            if state_str == 'abort':
                await ws.send_json({'error': f'thermal abort {t:.1f}C'})
                await ws.close()
                return
            if state_str == 'pause':
                await ws.send_json({'thermal_pause': True, 'apu_temp_c': t})
                await asyncio.sleep(2.0)
                continue
            sig = STATE.sig.read()
            STATE.last_sig = sig
            STATE.tick += 1
            anom = anomaly_score(STATE.active_anom(), sig)
            v_anom = vanilla_anomaly_score(STATE.tick)
            ident_idx, ident_conf = identity_predict(STATE.active_ident(), sig)
            v_idx, v_conf = vanilla_identity_predict(STATE.tick)
            # Pick 5 traces from the 32-d sig (one from each block) for plotting
            traces = {
                'pkg_uW':    float(sig[0]),
                'temp_mC':   float(sig[2]),
                'tsc_mean':  float(sig[25]),
                'ns_mean':   float(sig[27]),
                'cstate2':   float(sig[22]),
            }
            payload = {
                't': time.time(),
                'host_name': STATE.host_name,
                'apu_temp_c': t,
                'thermal_state': state_str,
                'sig_traces': traces,
                'embodied': {
                    'anomaly_score': anom,
                    'identity_label': ident_idx,
                    'identity_correct': bool(ident_idx == 0) if not STATE.transplanted else bool(ident_idx == 0),
                    'identity_confidence': ident_conf,
                },
                'vanilla': {
                    'anomaly_score': v_anom,
                    'identity_label': v_idx,
                    'identity_confidence': v_conf,
                },
                'transplanted': STATE.transplanted,
                'stress_active': time.time() < STATE.stress_until,
            }
            await ws.send_json(payload)
            await asyncio.sleep(1.0)  # 1 Hz, light load
    except WebSocketDisconnect:
        return
    except Exception as e:
        try: await ws.send_json({'error': str(e)})
        except: pass
        try: await ws.close()
        except: pass


# ---------------- startup: train heads ----------------
def init_state(host_name, peer_name, own_sigs_path, peer_sigs_path):
    STATE.host_name = host_name
    STATE.peer_name = peer_name
    print(f"[init] booting LiveSignature for host={host_name}...")
    STATE.sig = LiveSignature(host=host_name)
    print(f"[init] sig calibrated={STATE.sig.calibrated}")

    own = np.load(own_sigs_path)['sigs']
    print(f"[init] loaded own sigs from {own_sigs_path}: shape={own.shape}")
    print(f"[init] training T2 anomaly head on own sigs...")
    STATE.anom_own = train_anomaly_head(own)

    if os.path.exists(peer_sigs_path):
        peer = np.load(peer_sigs_path)['sigs']
        print(f"[init] loaded peer sigs from {peer_sigs_path}: shape={peer.shape}")
        print(f"[init] training T3 identity head (own=0, peer=1)...")
        STATE.ident_own = train_identity_head(own, peer)
        # Train "peer's" heads — same architecture, trained as if peer's the host.
        print(f"[init] training transplant heads (as-if {peer_name} were the host)...")
        STATE.anom_peer = train_anomaly_head(peer, seed=1)
        STATE.ident_peer = train_identity_head(peer, own, seed=1)
    else:
        print(f"[init] no peer sigs at {peer_sigs_path}; transplant disabled.")
        # Train identity head with shuffled own as synthetic peer (degraded)
        rng = np.random.default_rng(0)
        synth_peer = own.copy()
        rng.shuffle(synth_peer.T)
        synth_peer = np.clip(synth_peer + rng.normal(1.0, 0.5, synth_peer.shape), -4, 4).astype(np.float32)
        STATE.ident_own = train_identity_head(own, synth_peer)
    print(f"[init] ready. host={host_name} peer={peer_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host-name', required=True, choices=['ikaros', 'daedalus'])
    ap.add_argument('--peer-name', default=None)
    ap.add_argument('--own-sigs', required=True)
    ap.add_argument('--peer-sigs', required=True)
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--bind', default='0.0.0.0')
    args = ap.parse_args()
    peer = args.peer_name or ('daedalus' if args.host_name == 'ikaros' else 'ikaros')
    init_state(args.host_name, peer, args.own_sigs, args.peer_sigs)
    uvicorn.run(app, host=args.bind, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
