# FabricCrypt — X / Twitter launch thread

**Posting time:** Tuesday 14:00 UTC (07:00 PT, 10:00 ET, 16:00 CEST).
**Media plan:** Tweet 1 attaches `twitter_60s.mp4` (60-s edit). Tweet 8
attaches a 6-s GIF of the transplant moment.

Character counts (incl. spaces) shown in `[N]` at end of each tweet.

---

**1/ [hook + video]**

We made two identical computers. Only one could run our AI.

Same SKU, same RAM, same BIOS, same microcode. Move the model to
chassis B — it refuses to authenticate. Move it back to A — it works.

No vendor key. No Secure Enclave. Just physics.

[video: twitter_60s.mp4]

[265]

---

**2/**

The problem: Apple PCC and NVIDIA Confidential Compute can attest
"this is *some* H100 in CC mode" — but not *which* H100. The
attestation is bound to the SKU class, not the die.

And if you don't have Apple's PKI, you don't have PCC. Period.

[260]

---

**3/**

So we asked: can a commodity AMD laptop attest itself, per-die,
without a vendor key?

Turns out — yes, if you bypass the HAL and read what PSP firmware
cannot cheaply sanitise.

Five signals carry per-die identity:

[235]

---

**4/**

(a) Inter-core TSC offsets — picosecond-level wire-routing skew
(b) Cacheline ping-pong matrices — MOESI transition latencies
(c) DRAM-refresh-aligned jitter — phase-locked memory timing
(d) Nanosleep p99.9 tails — kernel scheduler micro-jitter
(e) NVMe queue-tail latency — per-namespace submission noise

None are factory-programmed. All are post-binning silicon noise.

[280]

---

**5/**

But a static fingerprint is replayable. So FabricCrypt binds each
challenge to a 64-bit audience nonce — and the nonce controls
*what gets measured*: which CPUs, which thermal zones, which
core pairs, which sleep durations.

Replay a recording → wrong plan → reject.

[280]

---

**6/**

The plan is derived as
SHAKE256(K_chip || domain || nonce)

K_chip is a per-die secret extracted from the chip's own
calibration fingerprint. It is enrolled with the verifier over
a physically-secure channel and never goes on the wire.

[245]

---

**7/**

Bit-security (honest):

• Source-code attacker, no K_chip:  ~60–80 bits (Tier 2)
• Calibration-file capture attacker: ~15–20 bits
• Generative-model attacker w/ 10^5 captured pairs: undefended

This is the headline future-work item. We're saying so up front.

[280]

---

**8/ [transplant GIF]**

Live demo. Same model file. Same code. Same prompts.

Run on chassis A: PASS. Verifier accepts.
Move to chassis B (`scp model.pt daedalus:`): REJECT. Same code, different die.

[gif: transplant_6s.gif]

[265]

---

**9/ HONEST LIMITATIONS**

• n=2 chassis. We say so in §7. Reviewer attack #1; we agree.
  Working on a 6-Strix-Halo array.
• Personality-attribution downstream: 66.4% (above chance,
  below ironclad).
• No static-benchmark accuracy gain (Phase 15/16 null).
• Persistent kernel adversary unmitigated.

[280]

---

**10/ Reproduce**

Repo (MIT-licensed):       https://github.com/Heigke/FabricCrypt
arXiv preprint:            arXiv:XXXX.XXXXX
60-s video:                attached to tweet 1
Long-form (3-min) video:   [YouTube link]

Everything is open. No enrollment with us. No accounts.

[265]

---

**11/ Call for replication**

If you own an AMD Strix Halo / Ryzen AI 300 laptop, please run
the enrollment script and post your `data/<hostname>_sig_v2.npz`
+ `dmidecode -t system` output.

Three independent chassis would more than triple our N. Two would
double it.

[265]

---

**12/ bonus — Star Wars droid analogy**

C-3PO and R2-D2 work for the same Empire (well, Republic). Same
manufacturer. Same protocol stack.

But Threepio is fluent in six million forms of communication and
R2 beeps. The instrument *is* the personality.

FabricCrypt is droid identity on commodity silicon.

[280]

---

## Posting checklist

- [ ] `twitter_60s.mp4` is < 2:20 and < 512 MB
- [ ] GIF for tweet 8 is < 15 MB and ≤ 6 seconds
- [ ] arXiv ID substituted into tweet 10
- [ ] YouTube long-form unlisted → public 5 min before tweet 1
- [ ] HN submission scheduled for T+48h (Wed 07:00 PT)
- [ ] Pin tweet 1
- [ ] Cross-post tweet 1 + tweet 8 to LinkedIn + Bluesky
- [ ] DM five named researchers (Mutlu, Yarom, Paccagnella, etc.)
      with the arXiv link 30 min before the public thread drops
