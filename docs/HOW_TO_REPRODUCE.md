# Reproduce FabricCrypt on YOUR HP Z2 mini G1a (or any AMD Strix Halo box)

This walkthrough takes ~30 minutes on two machines. The headline result
is **100% leave-one-out per-die classification** at N=2 chassis.

## Prerequisites

- 2 machines (any AMD Ryzen AI Max+ PRO 395 / Strix Halo / Zen 5 should
  work; we tested HP Z2 mini G1a)
- Ubuntu 24.04 LTS
- Python 3.11+
- `sudo` access (for `linux-tools-common` install and thermal_zone reads)
- ~5 GB free disk (Python wheels + PyTorch CPU)

## Step 1 — Install (each machine)

```bash
git clone git@github.com:Heigke/FabricCrypt.git
cd FabricCrypt
./scripts/00_install_deps.sh
source venv/bin/activate
```

If you want to tune the thermal guard for your chassis:

```bash
cp example.env .env
$EDITOR .env   # raise ABORT/PAUSE/COOL if your machine runs cooler
```

## Step 2 — Collect a signature on each machine

```bash
# On machine A (e.g. "alice"):
./scripts/01_collect_signature.sh --host alice --reps 10
# -> data/alice_sig_v2.npz   (~5-8 minutes, with thermal pauses)

# On machine B (e.g. "bob"):
./scripts/01_collect_signature.sh --host bob --reps 10
# -> data/bob_sig_v2.npz
```

Copy both `.npz` files to one machine:
```bash
scp data/alice_sig_v2.npz user@machineB:~/FabricCrypt/data/
```

## Step 3 — Cross-classify

```bash
./scripts/02_classify.sh data/alice_sig_v2.npz data/bob_sig_v2.npz
```

Expected output:
```
loo_acc: 1.0
gate_gt_0_95_passed: true
```

If you see `gate_gt_0_95_passed: false`, the most common cause is that
one of the machines was thermal-throttled during capture (check the
`*_sig_v2_meta.json` for per-rep temperatures — anything sustained above
65 °C will degrade reproducibility on a SFF chassis).

## Step 4 — Train the nonce-keyed classifier

```bash
# On machine A, training against machine B's library:
PYTHONPATH=. python -m src.protocol.train \
    --n_train 400 --n_seeds 30 \
    --peer_npz data/bob_paired_sigs.npz
# -> data/alice_t3_best.pt + data/alice_paired_sigs.npz
```

(You first need a peer's `paired_sigs.npz` — easiest way is to run the
same training on machine B without `--peer_npz`, copy its
`bob_paired_sigs.npz` back here, then re-run with `--peer_npz` set.)

## Step 5 — Run the 7-attack spoof suite

```bash
./scripts/03_test_replay.sh --peer_npz data/bob_paired_sigs.npz
```

Expected gates (from our N=2 paper run; your numbers will be close):

| attack                              | gate              | observed |
|-------------------------------------|-------------------|----------|
| honest_own                          | ≥ 0.95            | 1.00     |
| peer                                | ≤ 0.05            | 0.02     |
| static_replay_no_nonce              | ≤ 0.05            | 0.006    |
| static_replay_with_correct_nonce    | ≥ 0.95            | 1.00     |
| dynamic_replay                      | ≤ 0.10            | 0.012    |
| nonce_only_mismatch                 | ≤ 0.05            | 0.006    |
| honest_own_wrong_nonce              | ≤ 0.05            | 0.006    |

## Step 6 — Interactive demo

```bash
./scripts/04_demo.sh
```

Open <http://localhost:8770> in your browser. Click:
1. **Issue challenge** — server picks a fresh 64-bit nonce
2. **Sign on this chip** — server runs `NonceSig.read(nonce)`
3. **Verify** — deterministic plan-consistency check
4. **try a stale replay** — re-uses an earlier signature under a fresh
   nonce; you should see **REJECT**.

## Reporting back

We want N ≥ 6 chips for the final paper version. Once you've successfully
run steps 1–3, please consider sharing your signature:

```bash
./examples/publish_signature.sh
# inspects data/<host>_sig_v2.npz so you can confirm there's no PII,
# then prints instructions for opening a GitHub issue with it attached.
```

The `.npz` contains only numeric measurements — no usernames, no paths,
no filesystem content. See [`examples/publish_signature.sh`](../examples/publish_signature.sh)
to verify.

## Need help?

- Open an issue: <https://github.com/Heigke/FabricCrypt/issues>
- Bring a `*_sig_v2_meta.json` + the output of `02_classify.sh`
  if the gate failed.
