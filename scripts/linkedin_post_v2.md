# LinkedIn Post V2 — Draft

Last week I said every GPU is already a neuromorphic computer. 156 reactions and a lot of "no."

So I went deeper. Much deeper.

I systematically probed every layer of AMD's RDNA4 silicon — from the HIP API down through inline assembly, through MMIO register space, through the System Management Network, all the way to the Platform Security Processor's firmware boundary. 18 distinct hardware probes. Five independent protection layers mapped.

Here's what I found.

**Layer 3 — ISA-level control.** The GCN instruction `s_setreg` can modify the GPU's floating-point rounding mode from inside a running shader — in a single clock cycle. Four rounding modes produce four measurably different computational results from identical code. This isn't noise. It's a controllable nonlinear modulation primitive at the arithmetic unit level.

Same layer: `atomicAdd` functions as hardware synaptic integration (17/64 neurons active, 108 spikes). Wavefront shuffle instructions work as lateral neural connectivity. Instruction timing jitter (0–62 cycles for a single `sin()`) provides a genuine stochastic source. All from within a standard HIP kernel.

**Layer 2 — Real-time internal state observation.** The GPU's MMIO registers are readable from userspace at ~100 MHz while kernels execute. That means I can observe the GPU's internal pipeline dynamics — which blocks are busy, where the scheduler's program counter is, what the command processor is doing — at a bandwidth exceeding the kernel execution rate by orders of magnitude. No API exposes this. No documentation describes it. It's a new observation channel.

**Layer 0 — The wall.** Below the analog layers, AMD's PSP enforces five independent hardware protections. One PSP command drops the Data Fabric memory protection entirely — I wrote 110,000 words of patched firmware into the Trusted Memory Region with verified readback. Page table injection maps GPU virtual addresses to protected physical memory. The compute firmware sits in plaintext on disk.

None of it composes into code execution. The TMR content is AES-encrypted. The instruction cache base is hardware-locked. The five layers are independent — defeating one leaves the rest intact.

This is, to my knowledge, the first public offensive characterisation of RDNA3/4 PSP security architecture.

**The synthesis:** Commodity GPUs provide five distinct layers of neuromorphic computational access — standard API, ISA-level arithmetic control, hardware atomic neurons, real-time MMIO state observation, and analog sensor readout — all without firmware modification.

We didn't need to break the security. The computation was already there, four layers deep.

Updated preprint with full depth characterisation, security analysis, and 104 experiment groups (846 tests): [link]

Code: github.com/Heigke/feel-bridge

To the skeptics from last week: I heard you. So I went deeper. The data got stronger.
