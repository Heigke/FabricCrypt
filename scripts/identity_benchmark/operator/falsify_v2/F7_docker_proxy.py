"""F7 — cross-process isolation (proxy).

True docker test requires container with ROCm + HIP toolchain — heavy.
We use a process-level proxy: spawn divergent_matmul as a fully isolated
child process N times from a clean shell (env-stripped), and compare
its modal pattern to the in-process baseline.

If isolated children give same modal pattern as baseline -> OS env
doesn't matter (silicon dominant). If different -> userspace state
matters.

Also emits docker_protocol.md describing a true container test.
"""
from __future__ import annotations
import json, struct, subprocess, os
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
F7 = ROOT / "results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F7"
F7.mkdir(parents=True, exist_ok=True)
BIN = ROOT / "scripts/identity_benchmark/operator/divergent_matmul"


def load(p):
    raw = p.read_bytes()
    M, R, hl = struct.unpack("iii", raw[:12])
    return np.frombuffer(raw[12+hl:], dtype=np.float32).reshape(R, M).copy()


def modal(arr):
    bits = arr.view(np.uint32); R, M = bits.shape
    v = np.zeros(M, dtype=np.uint32); f = np.zeros(M)
    for m in range(M):
        c = Counter(bits[:, m].tolist())
        val, n = c.most_common(1)[0]
        v[m] = val; f[m] = n / R
    return v, f


def run_isolated(tag: str):
    out = F7 / f"iso_{tag}.bin"
    # env-stripped subprocess
    env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0",
           "PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}
    subprocess.run([str(BIN), "64", "4096", "32", str(out)],
                   env=env, check=True, capture_output=True)
    return out


def main():
    if not BIN.exists():
        print(json.dumps({"error": f"divergent_matmul not built at {BIN}"}, indent=2))
        return

    baseline = ROOT / "results/IDENTITY_OPERATOR_2026-05-31/ikaros_div.bin"
    A = load(baseline)
    va, fa = modal(A)

    # Run 2 isolated children
    out_paths = []
    for tag in ("a", "b"):
        try:
            p = run_isolated(tag)
            out_paths.append(p)
        except subprocess.CalledProcessError as e:
            print(f"isolated run {tag} failed: {e.stderr.decode()}")

    results = {"baseline_modal_stability": float(fa.mean())}
    for p in out_paths:
        I = load(p)
        vi, fi = modal(I)
        drift = int((va != vi).sum())
        results[p.name] = {
            "modal_stability": float(fi.mean()),
            "modal_drift_vs_baseline_count": drift,
            "modal_drift_frac": drift / A.shape[1],
        }
    results["verdict"] = (
        "OS-env irrelevant (drift ~0)"
        if all(v.get("modal_drift_frac", 1) < 0.05
               for v in results.values() if isinstance(v, dict))
        else "OS-env matters (drift > 5%)"
    )
    (F7 / "F7_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    (F7 / "docker_protocol.md").write_text(
        "# F7 true docker protocol\n\n"
        "1. `docker run --device=/dev/kfd --device=/dev/dri --group-add video "
        "rocm/dev-ubuntu-22.04 bash`\n"
        "2. Install hipcc inside container\n"
        "3. Build divergent_matmul.hip with same flags\n"
        "4. Run 32 reps, save bin\n"
        "5. Compare modal bits to bare-metal baseline\n"
        "6. Identical -> kernel+driver is OS-independent; "
        "different -> userspace state matters\n"
    )


if __name__ == "__main__":
    main()
