# Email to Casper @ HP — reproduction ask

**To:** [Casper's HP address]
**From:** bergvall.eric@gmail.com
**Subject:** 20 minutes of your time for a third FabricCrypt chassis?

---

Hej Casper,

Hope HP is treating you well. Quick favour, then back to your life.

I just released **FabricCrypt** — an open-source attestation primitive
that binds AI inference to a specific physical AMD die without any
vendor key. The repo and a 3-minute demo are here:

- Code: https://github.com/Heigke/FabricCrypt (MIT)
- Preprint: arXiv:XXXX.XXXXX
- Demo: [YouTube URL]

The single biggest weakness in the paper is **n=2 chassis** — I only
have two AMD Ryzen AI Max+ 395 laptops in hand (ikaros and daedalus).
Two laptops is not a study, and I say so in §7. Every reviewer is
going to attack this first.

You're sitting on the *exact* hardware I need a third sample from. If
you have 20 minutes free this week, would you be willing to:

1. `git clone https://github.com/Heigke/FabricCrypt && cd FabricCrypt`
2. `sudo ./scripts/enroll.sh` — this runs the five HAL-bypass samplers
   under SCHED_FIFO for ~15 minutes, no kernel modules, no GPU stress,
   nothing destructive.
3. Send me back two files:
   - `data/<hostname>_sig_v2.npz` (the 290-dim signature)
   - `sudo dmidecode -t system > <hostname>_dmi.txt`

Total time: ~20 minutes. The thermal load is mild (similar to a
matrix multiplication of moderate size). The samplers are read-only;
no firmware writes, no MSR pokes, no SMU mailbox traffic.

What you get: your laptop becomes the third independent chassis in
the public corpus. If you want anonymous attribution I can list it
as "chassis C — anonymous HP contributor"; if you want your name on
the artifact section I'm happy to do that too.

Why I'm asking *you* specifically: you know the HP X-series silicon
better than anyone I know in person, and a third die from a different
OEM / different BIOS lineage is worth more than a third die from
another Lenovo.

Take your time — no rush. Friday or weekend is fine. If you'd rather
just chat through the design first over coffee in town, that works
too.

Tack,

Eric
bergvall.eric@gmail.com
