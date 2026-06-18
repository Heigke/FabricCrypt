"""Narration text for the FabricCrypt pedagogical explainer.

8 sections. Total target: 5 to 8 minutes at OpenAI tts-1-hd 'nova' pacing.
ELI12 tone. No banned words (die, kill, soul, feel, loyalty, sentient, alive).
"""

SECTIONS = [
    # 0: Cold open hook  ~25 s
    (
        "S0_hook",
        "Imagine an A.I. so deeply tied to its body that copying it doesn't work. "
        "Think about Star Wars. If you copied R2-D2's memory into another droid, "
        "you wouldn't get R2. You would get a knockoff. Today we'll show you something "
        "very similar, but for real computers. We call it FabricCrypt."
    ),
    # 1: Everyday problem ~55 s
    (
        "S1_problem",
        "Here is the problem. You build an A.I. model. It is worth real money. "
        "Someone copies the file. Now they have your A.I., and they can run it anywhere. "
        "There is no easy way to prove that the inference came from your specific computer "
        "and not a stolen copy somewhere else. "
        "Apple, NVIDIA, Intel, and AMD all have answers to this. Apple Private Cloud Compute. "
        "NVIDIA Confidential Compute. Intel T.D.X. AMD S.E.V. dash S.N.P. "
        "But all four of them share a quiet assumption. You must trust the vendor's signing key, "
        "and you must use their special hardware: a Secure Enclave, a DICE certificate, a T.P.M. chip. "
        "We asked a simple question: could we do this on ANY commodity computer, without trusting any vendor?"
    ),
    # 2: The fingerprints ~80 s
    (
        "S2_fingerprints",
        "Every chip is a little bit unique. Manufacturing leaves tiny variations, "
        "like fingerprints. You can't see them, but you can measure them. "
        "We measure fifteen signals in three groups. "
        "First, five HAL-bypass micro-architectural signals. "
        "T.S.C. offset: how long it takes a timing signal to cross the chip. "
        "Cacheline ping-pong: the dance of data moving between processor cores. "
        "D.R.A.M. refresh jitter: the heartbeat of memory, which never quite ticks the same way twice. "
        "Syscall p99.9 tails: the worst-case response time of the operating system. "
        "And N.V.M.e. queue tail: the storage drive's personal latency signature. "
        "Second, three cross-host K.S.-verified micro-architectural signals: "
        "G.P.U. clock jitter, multi-zone thermal spread, and Jacobian temporal dynamics. "
        "Third, seven board-level deterministic fingerprints: "
        "P.C.I., P.C.I.-e., U.S.B., D.M.I., U.C.S.I., amdgpu, and kernel-boot descriptors. "
        "These last seven are board-level, not micro-architectural, and we say so. "
        "Together, they form a four hundred and sixty-six dimensional fingerprint. "
        "On two physically identical laptops, leave-one-out classification is one hundred percent. "
        "The chip can be told apart from its twin, every single time."
    ),
    # 3: Cryptographic binding ~75 s
    (
        "S3_crypto",
        "Now, just measuring is not enough. If an attacker records your fingerprint once, "
        "they could try to replay it later. So we add a twist. "
        "Every verification, the verifier picks a fresh sixty-four bit random number called a nonce. "
        "The nonce decides the sampling plan itself: which processor cores to probe, which thermal zones to read, "
        "which timing pairs to measure, and how long to wait between each measurement. "
        "The attacker cannot pre-record an answer, because they did not know the question until it arrived. "
        "On top of that, we add a Reverse Fuzzy Extractor and a Controlled Physically Unclonable Function wrap. "
        "These let us derive a stable per-die key from a noisy physical measurement, "
        "without ever exposing the raw bits. "
        "We also bind inference outputs to that key with H.M.A.C. and Pedersen commitments, "
        "so the verifier can confirm in zero-knowledge: yes, this output came from THAT chip. "
        "We report empirical attack-cost, not a formal cryptographic proof: "
        "roughly ten to the twelfth modeling samples returning random-Hamming distance, "
        "against an attacker who has full source code access but no physical extraction. "
        "Ten out of ten protocol attack gates passed, including custom forgery."
    ),
    # 4: Personality experiment - honest limitations subsection ~45 s
    (
        "S4_personality",
        "Before we close, one honest aside. "
        "We additionally explored whether the A.I.'s writing style would change between machines. "
        "Same small language model architecture, same training code, same data, "
        "trained one copy on ikaros and another copy on daedalus. "
        "Then we asked them to generate text and tried to tell them apart. "
        "Result: sixty-six point four percent detectable. Above chance, but not as strong as we wanted. "
        "We pre-registered a strict gate at seventy-five percent. We did not pass it. "
        "Honest null on the hard test. Full details are in the paper, section seven, L6 supplement. "
        "The substrate leaves a stylistic footprint, but whether it is strong enough to matter is still open."
    ),
    # 5: Why it matters ~70 s
    (
        "S5_why",
        "Why does any of this matter? "
        "FabricCrypt is, at n equals two chassis, the first per-die A.I. attestation primitive that needs no vendor key, "
        "no Secure Enclave, no dedicated security chip. It runs on a commodity A.M.D. mini-P.C. "
        "you can buy today for about seven hundred dollars. "
        "Consider the use cases. "
        "Verifiable inference origin: a customer can prove an output came from a specific chip, "
        "not a stolen copy of the model. "
        "A.I. insurance: an insurer can verify that a claimed deployment is actually running where it says it is. "
        "Sybil-resistant federated learning: each participant proves they are a distinct physical device, "
        "without relying on Intel S.G.X. or T.D.X. "
        "Substrate-locked A.I.: a model that is bound, by physics, to one specific chassis. "
        "Move it, and the verifier rejects. "
        "End-to-end sign and verify latency: one point one two milliseconds at the median. "
        "Fast enough to run inline with every inference."
    ),
    # 6: Honest caveats ~40 s
    (
        "S6_honest",
        "Let us be honest about what we have not shown. "
        "We tested on two chassis. Two. We need many more before this is a robust population claim. "
        "We did not pass our top-line personality gate. We are reporting a null on that. "
        "The protocol proves who ran the inference, not what the inference was. "
        "It is a primitive, not a complete confidential compute system. "
        "If you have a Strix Halo A.P.U., please reproduce us. Push us. Break us if you can. "
        "Code and figures will be on GitHub. The paper is going to arXiv."
    ),
    # 7: Close ~20 s
    (
        "S7_close",
        "An A.I. coupled to its substrate. No vendor key. No special chip. Just physics, and a fresh challenge every time. "
        "FabricCrypt. Two identical computers. Only one can run our A.I."
    ),
]
