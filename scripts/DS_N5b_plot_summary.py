"""Plot accuracy_vs_N and write summary.md for DS-N5b."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    data = json.loads(Path(args.in_json).read_text())
    by_N = data["by_N"]
    Ns_sorted = sorted([int(k) for k in by_N.keys()])

    means, stds, walls = [], [], []
    for N in Ns_sorted:
        rec = by_N[str(N)]
        accs = [r["test_acc"] for r in rec["per_seed"] if "test_acc" in r]
        ws = [r["wall_s"] for r in rec["per_seed"] if "wall_s" in r]
        means.append(np.mean(accs) if accs else np.nan)
        stds.append(np.std(accs) if accs else 0)
        walls.append(np.mean(ws) if ws else np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(Ns_sorted, means, yerr=stds, marker="o", capsize=4)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Hypervector dim N (D)")
    ax[0].set_ylabel("UCI-HAR test accuracy")
    ax[0].set_title("HDC accuracy vs hypervector dimension")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].axhline(0.84, color="gray", linestyle="--",
                  label="N=1024 baseline ~0.84")
    ax[0].legend()

    ax[1].plot(Ns_sorted, walls, marker="s", color="C1")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Hypervector dim N")
    ax[1].set_ylabel("Wall time per seed (s)")
    ax[1].set_title("HDC wall time vs N (5 seeds, mean)")
    ax[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out_png = Path(args.out_dir) / "accuracy_vs_N.png"
    plt.savefig(out_png, dpi=120)
    plt.close()

    # summary.md
    lines = ["# DS-N5b HDC Scale on UCI-HAR (GB10)\n",
             "| N | mean acc | std | mean wall (s) |",
             "|---:|---:|---:|---:|"]
    for N, m, s, w in zip(Ns_sorted, means, stds, walls):
        lines.append(f"| {N:,} | {m:.4f} | {s:.4f} | {w:.2f} |")
    lines.append("")
    base = means[0]
    lines.append(f"Baseline N=1024 mean: **{base:.4f}**")
    for N, m in zip(Ns_sorted[1:], means[1:]):
        delta = (m - base) * 100
        lines.append(f"- N={N:,}: {m:.4f} ({delta:+.2f}pp vs N=1024)")
    Path(args.out_dir, "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_png} and summary.md")


if __name__ == "__main__":
    main()
