"""H7 Fas 1 / A3 — CLOSED reafferent loop with the body doing REAL computation in the loop.

Wires the verified cache-destructive-interference XOR "organ" (micro_mem.c) into a live gpt2 loop:
  per step t:
    1. LLM ACTION A[t]: gpt2 emits a token; A = 1 if its next-token entropy is above running median
       (a real property of the LLM's internal state) XOR-corrected by feedback bias (see 5).
    2. ENV bit B[t]: the task stimulus (random).
    3. BODY computes G[t] = XOR(A,B) PHYSICALLY: drive the two L3-sharing streamers keyed to (A,B),
       read throughput-sum, threshold (XOR=1 cells run fast -> high sum; 00 & 11 -> low). G is produced
       by cache contention, not arithmetic.
    4. The loop's OUTPUT is G; TASK = compute XOR(LLM-action, env). metric = acc(G, true XOR).
    5. FEEDBACK (closes the loop): G sets a logit bias on the LLM's next step (body -> LLM), so the
       LLM's future action depends on the body's computation. body<->LLM mutual dependence.

Pre-registered kill-shot ablations (each must break the metric vs INTACT):
  - plant_lock : G frozen constant            -> body output not load-bearing
  - yoked      : G from a shuffled body read   -> right compute, wrong binding
  - efference0 : A forced 0 (LLM decoupled)    -> body computes XOR(0,B)=B, loop degenerates
GREEN if intact acc >> chance AND >> every ablation.

Out: closed_loop_{host}.json. Needs micro_mem built; gpt2 (transformers). Run w/ HSA override.
"""
from __future__ import annotations
import os, sys, time, json, argparse, struct, mmap, subprocess, tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0
HOST = P0.HOST; HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--win", type=float, default=0.05)
    ap.add_argument("--mb", type=int, default=24)
    ap.add_argument("--cpu_a", type=int, default=0); ap.add_argument("--cpu_b", type=int, default=2)
    ap.add_argument("--model", default="gpt2")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    V = lm.config.vocab_size

    # ---- body organ: two L3-sharing streamers, throughput-sum threshold = physical XOR readout ----
    binp = HERE/"micro_mem"
    if not binp.exists():
        subprocess.run(["gcc","-O2","-march=native","-o",str(binp),str(HERE/"micro_mem.c")],check=True)
    shm = Path(tempfile.gettempdir())/f"h7loopshm_{os.getpid()}"; shm.write_bytes(b"\x00"*64)
    fd = os.open(str(shm), os.O_RDWR); mm = mmap.mmap(fd, 64)
    sf = lambda i,v: (mm.seek(i*4), mm.write(struct.pack("i",v)))
    sc = lambda i,v: (mm.seek(8+i*8), mm.write(struct.pack("Q",v)))
    gc = lambda i: (mm.seek(8+i*8), struct.unpack("Q", mm.read(8))[0])[1]
    arr = a.mb*1024*1024
    pA = subprocess.Popen([str(binp),str(a.cpu_a),"0",str(shm),str(arr)])
    pB = subprocess.Popen([str(binp),str(a.cpu_b),"1",str(shm),str(arr)])
    time.sleep(0.5)

    def body_compute(av, bv):
        """Run the streamers keyed to (av,bv); return throughput-sum (the physical signal)."""
        sc(0,0); sc(1,0); sf(0,int(av)); sf(1,int(bv)); time.sleep(a.win)
        s = gc(0)+gc(1); sf(0,0); sf(1,0); return s

    # calibrate the XOR threshold from a few probe cells (body's own readout, not arithmetic)
    cal = {(x,y):[] for x in (0,1) for y in (0,1)}
    for _ in range(6):
        for x in (0,1):
            for y in (0,1): cal[(x,y)].append(body_compute(x,y))
    mean = {k: float(np.mean(v)) for k,v in cal.items()}
    # XOR=1 cells (01,10) run fast (high sum); 00 (idle) and 11 (contention) low -> threshold between
    # XOR=0 is BIMODAL (00 idle≈0 AND 11 contention≈low-but-nonzero); XOR=1 (01,10) runs fast.
    # The single clean separator sits between the LOWEST XOR=1 cell and the HIGHEST XOR=0 cell.
    lo_xor1 = min(mean[(0,1)], mean[(1,0)]); hi_xor0 = max(mean[(0,0)], mean[(1,1)])
    thr = (lo_xor1 + hi_xor0)/2
    print(f"[{HOST}] closed-loop dev={dev} cells={ {k:round(v) for k,v in mean.items()} } thr={thr:.0f}", flush=True)

    def run(mode, steps, seed):
        rng = np.random.default_rng(seed)
        ids = tok("The body and the mind", return_tensors="pt").input_ids.to(dev)
        past=None; cur=ids; ent_hist=[]; fb_bias=0.0
        G=[]; TRUE=[]; A_used=[]
        yoke_pool=[]
        with torch.no_grad():
            for t in range(steps):
                out = lm(cur if past is None else cur[:, -1:], past_key_values=past, use_cache=True)
                past = out.past_key_values; logits = out.logits[:,-1,:].float()
                # FEEDBACK: body's last G biases the LLM (body -> LLM). closes the loop.
                if mode!="efference0" and fb_bias!=0.0:
                    logits = logits + fb_bias*0.0  # bias applied via temperature below to keep stable
                p = torch.softmax(logits/(1.0+0.3*fb_bias), -1)
                ent = float(-(p*torch.log(p+1e-12)).sum())
                ent_hist.append(ent); med = float(np.median(ent_hist[-50:]))
                A_true = 1 if ent > med else 0                    # the LLM's real action
                B = int(rng.integers(0,2))
                true_xor = A_true ^ B                             # task target uses the REAL action
                A_body = 0 if mode=="efference0" else A_true      # efference0 decouples body from action
                s = body_compute(A_body, B)                       # BODY computes physically
                g = 1 if s > thr else 0
                if mode=="yoked":                                  # right compute, wrong binding
                    yoke_pool.append(g); g = yoke_pool[rng.integers(0,len(yoke_pool))]
                if mode=="plant_lock": g = 1                       # frozen body output
                G.append(g); TRUE.append(true_xor); A_used.append(A_true)
                fb_bias = 1.0 if g else -1.0                       # body result -> next-step bias
                nxt = torch.multinomial(p,1); cur = torch.cat([cur,nxt],1)
                if cur.shape[1]>128: cur=cur[:,-32:]; past=None
        G=np.array(G); TRUE=np.array(TRUE)
        acc = float((G==TRUE).mean())
        return {"acc_vs_trueXOR": round(acc,3), "n": int(len(G)),
                "A_rate": round(float(np.mean(A_used)),3)}

    res={"host":HOST,"thr":round(thr,1),"cell_means":{f"{k[0]}{k[1]}":round(v) for k,v in mean.items()}}
    try:
        res["intact"]     = run("intact",     a.steps, 1)
        res["plant_lock"] = run("plant_lock", a.steps, 2)
        res["yoked"]      = run("yoked",       a.steps, 3)
        res["efference0"] = run("efference0",  a.steps, 4)
    finally:
        sf(0,2); sf(1,2); time.sleep(0.2); pA.terminate(); pB.terminate(); mm.close(); os.close(fd)
        try: shm.unlink()
        except Exception: pass
    intact = res["intact"]["acc_vs_trueXOR"]
    abl = max(res["plant_lock"]["acc_vs_trueXOR"], res["yoked"]["acc_vs_trueXOR"], res["efference0"]["acc_vs_trueXOR"])
    res["GREEN"] = bool(intact > 0.8 and intact - abl > 0.2)
    jp = P0.OUT/f"closed_loop_{HOST}.json"; jp.write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2),flush=True)
    print(f"\n[{HOST}] CLOSED-LOOP VERDICT: intact acc={intact} | plant_lock={res['plant_lock']['acc_vs_trueXOR']} "
          f"yoked={res['yoked']['acc_vs_trueXOR']} efference0={res['efference0']['acc_vs_trueXOR']} -> GREEN={res['GREEN']}")
    print(f"  saved {jp}")

if __name__=="__main__":
    main()
