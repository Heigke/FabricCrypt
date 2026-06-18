# r/programming post

**Subreddit:** r/programming
**Drop window:** T+72h (Friday).
**Angle:** demo-led — "we made two identical laptops and only one
could run our model" — engineering achievement, not academic.

---

## Title

```
We made two identical laptops. Only one could run our AI model. (FabricCrypt, MIT-licensed)
```

---

## Body

Two AMD Ryzen AI Max+ 395 "Strix Halo" laptops. Same SKU, same RAM
SKU, same BIOS, same microcode, same model file, same code.

Run the model on laptop A: PASS. Verifier accepts.
`scp model.pt daedalus:` — run on laptop B: REJECT.

Same code. Same model. Different die. That's the whole demo.

---

**How.** We bundle five micro-architectural signals that PSP firmware
does not (and cannot cheaply) sanitise — inter-core TSC offsets,
cacheline ping-pong, DRAM-refresh-aligned jitter, nanosleep p99.9
tails, NVMe queue-tail latency — into a 290-dimensional live device
signature. None of these are factory-programmed; all arise from
post-binning silicon variation that differs across nominally
identical chips.

**Replay defence.** A 64-bit audience nonce controls *which* CPUs,
*which* thermal zones, *which* core pairs, and *which* sleep
durations get sampled. A recording of yesterday's response is useless
against today's nonce because the sample plan is different.

**Numbers:**

- 100% leave-one-out per-die classification (n=2 chassis, 20 reps).
- median 1.12 ms / p99 2.79 ms sign-and-verify.
- All ten attack-battery gates pass, including a post-disclosure
  forgery (O115) that defeated our previous version.

**Why this matters.** Apple PCC and NVIDIA Confidential Compute
attest the *SKU class*, not the *individual die*. Two H100s with the
same firmware are indistinguishable to the verifier. FabricCrypt
gets you per-die identity on commodity AMD without a Secure Enclave,
TPM EK certificate, or any vendor key. You enroll your own laptop.
We are not in the loop.

**Honest about what we did NOT do:**

- n=2. Two laptops is not a study. Working on a 6-chassis array.
- No static-benchmark accuracy gain (we ran the experiment; it was null).
- Bit-security is bounded — ~60–80 bits against a source-code-aware
  attacker, ~15–20 bits if they steal your enrolled key file.
- Personality-attribution downstream task: 66.4% accuracy on
  chip-of-origin from generated text — above chance, below proof.

**Code:** https://github.com/Heigke/FabricCrypt (MIT)
**Paper:** arXiv:XXXX.XXXXX
**3-minute demo:** [YouTube]

If you own an AMD Strix Halo or Ryzen AI 300 laptop, please run
the enrollment script (~20 minutes) and post your signature. Three
independent chassis would more than triple our N.
