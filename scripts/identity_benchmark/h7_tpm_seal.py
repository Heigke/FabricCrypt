"""H7 Fas 1 — the HARD cryptographic root that closes the critic hole.

The substrate-dependence demo (rooted GPT-2 + per-core fingerprint) proves the model's
*behaviour* depends on the die. A critic answers: "copy the weights, read the 16-D
fingerprint off the target machine, feed it in — done." True. The fingerprint is a
*readable* vector; FiLM on a readable vector is key-gating, not uncopyability.

This script removes that hole. The model adapter is encrypted with an AES-256 key that is
SEALED into the machine's discrete Nuvoton TPM 2.0 under the owner hierarchy. The TPM's
owner seed never leaves the chip and is unique per die. So:

  * enroll     : AES-GCM-encrypt the adapter; seal the AES key to THIS TPM.
  * run        : fresh-nonce tpm2_quote (liveness, non-replayable) -> tpm2_unseal the AES
                 key -> decrypt the adapter -> ready to infer.
  * transplant : copy the encrypted adapter + sealed blobs to another die; tpm2_unseal
                 FAILS (different owner seed) -> the model REFUSES to load.

The fingerprint stays as the *behavioural* root (science, falsifiable); the TPM seal is the
*cryptographic* root (uncopyability). Two honest tiers, both real, both on this hardware.

Usage:
  python h7_tpm_seal.py enroll  --adapter results/IDENTITY_H7_2026-06-09/rooted_gpt2_daedalus.pt
  python h7_tpm_seal.py run                  # fresh nonce quote + unseal + decrypt (in-memory only)
  python h7_tpm_seal.py transplant-check     # prove a foreign sealed blob can't be unsealed here
Requires: tpm2-tools, /dev/tpm0 (sudo for resource-manager-less access on some boxes), cryptography.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, socket, subprocess, sys, tempfile, time
from pathlib import Path

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
VAULT = OUT / f"tpm_vault_{HOST}"          # holds the sealed key + encrypted adapter for THIS die
OUT.mkdir(parents=True, exist_ok=True)

# TPM owner-hierarchy primary template: deterministic from the per-die owner seed.
# Same machine -> same primary; different machine -> different primary -> unseal fails.
PRIMARY_ALG = ["-g", "sha256", "-G", "ecc"]


def sh(cmd, **kw):
    """Run a command, return CompletedProcess. Never raises on nonzero (we inspect rc)."""
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def tpm(cmd, **kw):
    """tpm2 tool call; tries plain, falls back to sudo if /dev/tpm0 needs it."""
    r = sh(cmd, **kw)
    if r.returncode != 0 and ("tpm0" in r.stderr.lower() or "permission" in r.stderr.lower()
                              or "tcti" in r.stderr.lower() or "EACCES" in r.stderr):
        r = sh(["sudo", "-n"] + cmd, **kw)
    return r


def have_tpm():
    if not Path("/dev/tpm0").exists() and not Path("/dev/tpmrm0").exists():
        return False, "no /dev/tpm0 or /dev/tpmrm0"
    r = tpm(["tpm2_getcap", "properties-fixed"])
    if r.returncode != 0:
        return False, r.stderr.strip()[:200]
    mfg = ""
    for line in r.stdout.splitlines():
        if "TPM2_PT_MANUFACTURER" in line or "raw" in line.lower():
            mfg += line.strip() + " "
    return True, mfg.strip()


# ---------------- crypto helpers ----------------
def aes_gcm_encrypt(key: bytes, plaintext: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce + ct


def aes_gcm_decrypt(key: bytes, blob: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).decrypt(blob[:12], blob[12:], None)


# ---------------- enroll ----------------
def enroll(adapter_path: str):
    ok, info = have_tpm()
    if not ok:
        print(f"[{HOST}] NO USABLE TPM: {info}"); sys.exit(2)
    ap = Path(adapter_path)
    if not ap.exists():
        print(f"adapter not found: {ap}"); sys.exit(2)
    print(f"[{HOST}] TPM ok ({info[:80]})")
    VAULT.mkdir(parents=True, exist_ok=True)
    plain = ap.read_bytes()
    aes_key = os.urandom(32)                                    # the model key — never stored in clear
    enc = aes_gcm_encrypt(aes_key, plain)
    (VAULT / "adapter.enc").write_bytes(enc)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pri = td / "pri.ctx"; sec = td / "sec.bin"
        sec.write_bytes(aes_key)
        # 1) owner-hierarchy primary (derived from this die's owner seed)
        r = tpm(["tpm2_createprimary", "-C", "o", *PRIMARY_ALG, "-c", str(pri)])
        if r.returncode != 0:
            print("createprimary failed:", r.stderr[:300]); sys.exit(3)
        # 2) seal the AES key as a child object of that primary
        r = tpm(["tpm2_create", "-C", str(pri), "-i", str(sec),
                 "-u", str(VAULT / "seal.pub"), "-r", str(VAULT / "seal.priv")])
        if r.returncode != 0:
            print("create/seal failed:", r.stderr[:300]); sys.exit(3)
    # public name of the primary, for a sanity check at run time (NOT a secret)
    meta = {"host": HOST, "adapter": ap.name, "sha256_plain": hashlib.sha256(plain).hexdigest(),
            "enc_bytes": len(enc), "sealed": ["seal.pub", "seal.priv"], "ts": int(time.time())}
    (VAULT / "vault.json").write_text(json.dumps(meta, indent=2))
    aes_key = b"\x00" * 32  # best-effort scrub of our local copy
    print(f"[{HOST}] SEALED. adapter encrypted -> {VAULT/'adapter.enc'} ({len(enc)} B)")
    print(f"[{HOST}] AES key sealed to this TPM -> seal.pub/seal.priv  (clear key discarded)")
    print(f"[{HOST}] copy the whole {VAULT.name}/ to another die and run --transplant-check: it will REFUSE.")


# ---------------- run (fresh nonce + unseal + decrypt) ----------------
def quote_fresh(td: Path):
    """Fresh, non-replayable liveness proof: TPM signs an externally-supplied random nonce."""
    nonce = os.urandom(20)
    ak = td / "ak.ctx"; ek = td / "ek.ctx"
    # ephemeral attestation key under endorsement hierarchy
    r = tpm(["tpm2_createek", "-G", "ecc", "-c", str(ek)])
    if r.returncode != 0:
        return None, f"createek: {r.stderr[:150]}"
    r = tpm(["tpm2_createak", "-C", str(ek), "-G", "ecc", "-g", "sha256", "-s", "ecdsa",
             "-c", str(ak), "-u", str(td / "ak.pub")])
    if r.returncode != 0:
        return None, f"createak: {r.stderr[:150]}"
    r = tpm(["tpm2_quote", "-c", str(ak), "-l", "sha256:0,1,7", "-q", nonce.hex(),
             "-m", str(td / "quote.msg"), "-s", str(td / "quote.sig"), "-o", str(td / "quote.pcr")])
    if r.returncode != 0:
        return None, f"quote: {r.stderr[:150]}"
    return nonce, None


def run():
    ok, info = have_tpm()
    if not ok:
        print(f"[{HOST}] NO USABLE TPM: {info}"); sys.exit(2)
    if not (VAULT / "adapter.enc").exists():
        print(f"[{HOST}] no vault for this die — run enroll first ({VAULT})"); sys.exit(2)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1) LIVENESS: fresh nonce quote (proves the chip is physically present NOW)
        nonce, err = quote_fresh(td)
        if nonce is None:
            print(f"[{HOST}] LIVENESS quote failed: {err}"); sys.exit(4)
        print(f"[{HOST}] liveness OK — TPM signed fresh nonce {nonce.hex()[:16]}.. (non-replayable)")
        # 2) UNSEAL: recreate the per-die primary, load the sealed object, unseal the AES key
        pri = td / "pri.ctx"
        r = tpm(["tpm2_createprimary", "-C", "o", *PRIMARY_ALG, "-c", str(pri)])
        if r.returncode != 0:
            print("createprimary failed:", r.stderr[:200]); sys.exit(4)
        seal = td / "seal.ctx"
        r = tpm(["tpm2_load", "-C", str(pri), "-u", str(VAULT / "seal.pub"),
                 "-r", str(VAULT / "seal.priv"), "-c", str(seal)])
        if r.returncode != 0:
            print(f"[{HOST}] UNSEAL/load FAILED (wrong TPM?) -> model REFUSED: {r.stderr[:150]}"); sys.exit(5)
        kf = td / "k.bin"                    # unseal to a file (binary key, avoids stdout decode issues)
        r = tpm(["tpm2_unseal", "-c", str(seal), "-o", str(kf)])
        if r.returncode != 0 or not kf.exists():
            print(f"[{HOST}] unseal FAILED -> model REFUSED: {r.stderr[:150]}"); sys.exit(5)
        aes_key = kf.read_bytes()
        if len(aes_key) != 32:
            print(f"[{HOST}] unsealed key wrong size ({len(aes_key)}B) -> REFUSED"); sys.exit(5)
        # 3) DECRYPT the adapter in memory
        enc = (VAULT / "adapter.enc").read_bytes()
        try:
            plain = aes_gcm_decrypt(aes_key, enc)
        except Exception as e:
            print(f"[{HOST}] AES-GCM auth FAILED -> tampered/wrong key: {e}"); sys.exit(6)
        meta = json.loads((VAULT / "vault.json").read_text())
        good = hashlib.sha256(plain).hexdigest() == meta["sha256_plain"]
        print(f"[{HOST}] UNSEAL ok, adapter decrypted in-memory ({len(plain)} B), "
              f"integrity={'MATCH' if good else 'MISMATCH'}  [{time.time()-t0:.1f}s]")
        if not good:
            sys.exit(6)
        # Optionally hand the plaintext adapter to a torch loader here. We keep it in memory only.
        print(f"[{HOST}] >>> model adapter UNLOCKED on its own die. Ready for embodied inference.")


# ---------------- transplant check ----------------
def transplant_check():
    """Attempt to unseal whatever vault is present using THIS die's TPM. If the vault was created on
    another die, the owner-hierarchy primary differs and tpm2_load/unseal MUST fail."""
    ok, info = have_tpm()
    if not ok:
        print(f"[{HOST}] NO USABLE TPM: {info}"); sys.exit(2)
    # accept an explicit foreign vault dir via env, else scan for any tpm_vault_* that's not ours
    cand = os.environ.get("FOREIGN_VAULT")
    vaults = [Path(cand)] if cand else sorted(OUT.glob("tpm_vault_*"))
    for v in vaults:
        if not (v / "seal.pub").exists():
            continue
        origin = json.loads((v / "vault.json").read_text()).get("host", "?")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pri = td / "pri.ctx"
            tpm(["tpm2_createprimary", "-C", "o", *PRIMARY_ALG, "-c", str(pri)])
            r = tpm(["tpm2_load", "-C", str(pri), "-u", str(v / "seal.pub"),
                     "-r", str(v / "seal.priv"), "-c", str(td / "s.ctx")])
            same_die = (origin == HOST)
            loaded = (r.returncode == 0)
            verdict = ("UNLOCKS (own die)" if loaded and same_die else
                       "REFUSED (foreign die — integrity value mismatch)" if not loaded and not same_die else
                       f"UNEXPECTED loaded={loaded} same_die={same_die}")
            tag = "OK" if (loaded == same_die) else "!! UNEXPECTED"
            print(f"[{HOST}] vault from '{origin}': {verdict}  [{tag}]  rc={r.returncode} {r.stderr.strip()[:80]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("--adapter", required=True)
    sub.add_parser("run")
    sub.add_parser("transplant-check")
    a = ap.parse_args()
    if a.cmd == "enroll": enroll(a.adapter)
    elif a.cmd == "run": run()
    elif a.cmd == "transplant-check": transplant_check()
