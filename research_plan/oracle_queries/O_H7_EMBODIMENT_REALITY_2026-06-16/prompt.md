# Oracle request — hostile, expert critique of an "embodied / hardware-rooted LLM" research program

You are a panel of skeptical senior reviewers (ML security + cognitive science + hardware). A
research agent (Claude Code, working for Eric Bergvall) built a frozen GPT-2 + tiny learned
"steering" adapter whose text output depends, in real time, on a specific AMD machine's hardware
signals (per-core voltage/clock) plus a TPM-sealed key. The public framing ("operational
embodiment", "unique/fresh/uncopyable", "computation in the body") was pulled back as shallow.

The attached **PLAN_AND_SOTA.md** is our honest reality-check after 5 literature sweeps, plus a
proposed 2-direction plan: **(A) science** — climb to Butlin AE-2 "embodiment" by proving the
LLM's own token generation perturbs its host telemetry (power/thermal/clock) in a *modeled,
ablation-load-bearing closed loop*; **(B) security** — a properly-evaluated hardware-rooted
binding (challenge-response, DRAM/RowHammer PUF, attack battery), scoped honestly to economic
deterrence. We own 2× consumer AMD gfx1151 boxes + 1× NVIDIA GB10 (all lack a usable GPU TEE).

Read PLAN_AND_SOTA.md in full. Then answer, bluntly and specifically:

1. **Is Direction A (Butlin AE-2 closed loop) real science or a coming artifact?** The
   output→telemetry signal is contaminated by the CPU governor, DVFS, thermal inertia, and OS
   scheduling. Could a "passing" AE-2 result be an artifact of those rather than genuine
   reafference? Design the single confound that would most likely fool us, and the control that
   kills it. Give the strongest ablation/kill-shot design so a positive result is credible.

2. **Is Direction B worth doing at all** given Clifford et al. (SaTML 2025) already covers
   fingerprint-keyed model locking, and we have no TEE on any box? Or is the only honest move to
   drop the security framing entirely and go pure-science? If B is worth it, what is the *minimum*
   that would make it a contribution rather than a worse re-run of Clifford?

3. **Where are we STILL overclaiming** in the plan (§3–4)? Where would a hostile reviewer laugh?

4. **Is there a THIRD direction** the SOTA implies that we missed — a genuinely novel, honest,
   reachable target on commodity AMD silicon (no FPGA, no custom chips)?

5. **Priority call**: if we can only seriously pursue ONE direction in the next 2 weeks, which,
   and why? What is the single highest-value experiment to run first?

Be specific, cite mechanisms, and do not be polite. We want to find the holes before a reviewer does.
