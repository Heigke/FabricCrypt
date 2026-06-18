# Adversarial Defence Script

Five most likely sceptic attacks during/after the live demo, with the
exact-words response and the on-stage evidence we point to.

## Attack 1 — "It's just stage magic. The video is pre-rendered."

**Sceptic line:** "How do we know you didn't pre-record everything and the
laptops are just monitors?"

**Response:** "Two answers. First, the nonce. Volunteer A's dice rolls a
moment ago became the 64-bit input to our sampling plan. The hash of that plan
went onto the URL on the screen *before* sampling started. You can pull that
hash up on your phone right now and recompute it from the dice. Second, you
can do it yourself: the repo at the URL has the verifier binary and our public
chip signatures. Hand it a recording, get a reject."

**On-stage evidence:** Live nonce hash panel. Reproduce URL. Volunteer A still
holding the dice.

---

## Attack 2 — "The two laptops aren't really identical. You're just reading the serial number."

**Sceptic line:** "If you swap two genuinely identical chips at the silicon
fab, your demo would fail. Prove you aren't just dumping `/proc/cpuinfo`."

**Response:** "We don't read any vendor identifier. Here's the source on
screen: the five signals are inter-core TSC offsets, cacheline ping-pong
matrices, DRAM refresh jitter, syscall p99.9 tails, and NVMe queue-tail
latencies. None of these are programmed at the factory. They are *manufacturing
variation noise* — the silicon equivalent of how no two snowflakes refrigerate
identically. Two chassis with the same SKU, same RAM SKU, same BIOS, same
microcode — and they still separate at 100% on 290 features."

**On-stage evidence:** `dmidecode` panel showing identical SKUs. Source code
of `nonce_signature_v2.py` shown in a code panel, audience can see the signals.
Reference to `embodiment12/task_*` data.

---

## Attack 3 — "Personalities are hard-coded. You typed 'be warm' in K-2's system prompt."

**Sceptic line:** "The Twin Reveal is theatre. You wrote two system prompts."

**Response:** "Two answers. First, here is the system prompt on screen — it is
*identical* for both droids: 'Answer in five words.' Second, the per-chip
style differences come from a deterministic function of the chip's own
fingerprint vector: `style_template_idx = H(signature)[0:8] mod 32`. We do not
choose. The chip does. Reach into the codebase right now and check — there is
no `if host == ikaros` branch."

**On-stage evidence:** Code panel showing the hash-derived template selection.
Identical system prompt overlay. Live `git grep "host == "` returning nothing.

---

## Attack 4 — "This is just DRM. You're locking AI to chips so people can't escape vendors."

**Sceptic line:** "Whose interests does this serve? Sounds like Apple PCC for
laptops."

**Response:** "Fair concern. Three points. (a) Unlike Apple PCC, FabricCrypt
*does not require a vendor key* — you can run it on your own AMD laptop and
generate your own per-die identity, no enrollment with us or anyone. (b) The
primitive proves *who ran the inference*, not *what is allowed to run*. You can
install Linux freely; nothing about this design restricts what software boots.
(c) The use cases we care about are output attribution for AI liability and
sybil resistance in federated learning — both *protect* the user from
impersonation, they don't restrict you. We address this explicitly in
Section 6 of the paper."

**On-stage evidence:** Caveats panel (already in v2 video). Section 6 of the
paper. Open-source license on the verifier.

---

## Attack 5 — "N=2 is laughable. Come back with 1000 chips."

**Sceptic line:** "Two laptops is not a study. Where's your sample size?"

**Response:** "Agreed it is small, and we say so in Section 7. But N=2 is not
the *claim*; the claim is the *primitive works at all in a software-discoverable
way* with 100% LOO at 290 features, sub-ms latency, and replay/transplant rates
below 2%. The right next move is replication, not bigger N from us. The
artifact is released. We'd like you to bring a third chassis to the workshop;
we've published the protocol that lets you do it without coordination with us."

**On-stage evidence:** Section 7 caveats. Reproduce-script URL. Open call for
third-party chassis.

---

## Bonus — "Side channels can spoof your signals."

**Sceptic line:** "An attacker who controls scheduling, voltage, or HV cooling
could shape your signals to match a target chip."

**Response:** "Three layers. (a) Matched-governor — we re-ran with both chips
under `performance` governor and capability gains held within noise. (b) The
sampling *plan* is nonce-derived, so the attacker has to spoof a high-
dimensional signal across an unpredictable subset; we measured worst-case
spoof success at 2% for peer transplant. (c) We don't claim this resists a
state-level attacker with full SMU control; we claim it resists everything
short of that, on commodity hardware, software-discoverable. Section 5.5
covers the three adversary classes."

**On-stage evidence:** Section 4.4 matched-governor result. Section 5.5
adversary table.
