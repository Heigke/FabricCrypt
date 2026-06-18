# gemini response (gemini-2.5-pro) — 40s

This is an excellent, rigorous plan. Here is a critique to sharpen it before execution.

**1. Track ordering:** Your proposed order (P → V) is correct. Unblock your compute (Track P) first. Then, immediately validate the core finding with statistical power (Track V). This "fail-fast" approach correctly risks the least amount of time (2-3 days) before committing to the multi-week effort of Tracks T and R.

**2. Track P branch:** Implement the safer-CPU wrapper first. It's the fastest, lowest-risk path to unblocking Track V. The goal is to get moving *now*. A GPU port is a high-leverage optimization, but it's a separate engineering project with its own risks (ROCm drivers, numerical precision). Pursue the GPU port in parallel or after the CPU wrapper is functional and has unblocked the critical path.

**3. Validation power calibration:** 10 seeds × 240 samples is a pragmatic and significant increase. For a 5pp effect size, this should be sufficient to move from "maybe" to a confident claim. The key isn't a magic number, but the resulting confidence interval. If the 95% CI width from this run is still >5pp, you'll know you need more samples. For now, this is the right calibration.

**4. Track C cost model:** Literature-cited numbers are a defensible start. To make it more robust for Mario, frame it as a "zeroth-order estimate" and explicitly state the main caveat: routing overhead. The area cost of inhibition isn't just the 4 transistors per inverter; it's the metal layers and area consumed by the new wires in the crossbar, which your `O(N·r)` note correctly identifies. Acknowledge this is a lower bound.

**5. Decision gate at end of Week 1:** The "+5pp" cutoff is good, but be more specific. Define it as a **5 percentage point absolute improvement over the ER_SPARSE-only baseline, with non-overlapping 95% CIs**. This removes ambiguity and provides a clear, statistically grounded threshold for proceeding.

**6. Cron schedule additions:** The cadence is sound. It balances automated oversight with avoiding alert fatigue.
The single guardrail I would **ADD** is a **Baseline Performance Watchdog**. Add a cron (daily) that runs the *simplest baseline* (e.g., ER_SPARSE-only on Mackey-Glass) with a fixed seed. It should assert that the performance is within a tight, known range. This protects you from subtle bugs introduced into the core simulation code that might falsely improve your candidate's performance by degrading the baseline.

**7. Killer omission:** The plan's biggest blind spot is in **Track R (Realism)**. You validate the surrogate against `pyport-direct` on a tiny N=32 reservoir, but your candidate win is at N=256. The network dynamics, stability, and benefits of inhibition may not transfer from such a small model to a larger one. `pyport-direct` on N=256 is likely too slow, but you *must* address this scale gap.
**Fix:** Add a step to characterize the surrogate's accuracy as a function of network size (e.g., N=32, N=64, N=128) or provide a strong argument for why the N=32 validation is sufficient. Without this, Mario could rightly question if the entire finding is an artifact of a surrogate that's only accurate for toy-sized networks.

---
This plan is 95% of the way there. Implement these fixes, and you have a green light.
