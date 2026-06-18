"""LIF Conceptual Illustration of Pavlovian Conditioning (NOT NS-RAM substrate).

⚠️ HONESTY NOTE (per O18 oracle review, 2026-05-03):
This script is a CONCEPTUAL ILLUSTRATION. The cells are simplified
leaky-integrate-and-fire (LIF) surrogates with a hard-threshold snapback,
NOT the BSIM4 NS-RAM cell model used elsewhere in this project
(see `pyport/` and `scripts/z91*`/`z139*` for the real BSIM4-driven sims).

The demo shows the *coupling-structure potential* of a Hebbian-paired
network — i.e. that a recurrent graph plus a coincidence-update rule
yields conditioning. It does NOT demonstrate that the NS-RAM silicon's
intrinsic dynamics (BSIM4 + parasitic NPN charge pumping) implement
this learning rule. Whether the real cell exhibits this behaviour
under analog drive is an open question, pending transient validation
against Sebas's measured traces (see M3a-F harness).

Setup:
  - 2 input neurons:
      * cell #0 = "tone"   (CS, conditioned stimulus)
      * cell #1 = "reward" (US, unconditioned stimulus)
  - 1 output neuron:
      * cell #7 = "response" (the salivation analogue)
  - 5 hidden NS-RAM cells (#2..#6) form a small recurrent reservoir
    that mediates the association.
  - Couplings:
      * cell 1 → 7 (US→response) is FIXED strong
      * cell 0 → 7 (CS→response) starts WEAK and grows by Hebbian
        coincidence: a tone fires while the response cell is already
        depolarised by the reward → coupling strengthens.
      * 0/1 → hidden cells: random sparse, fixed
      * hidden ↔ hidden: random sparse, fixed
      * hidden → 7: small fixed positive weights

Dynamics: simplified leaky-integrate-and-fire with a snapback threshold
that captures NS-RAM body-voltage dynamics (charge, fire, reset). This
is illustrative — the *concept* (body voltage integrating, snapback
firing on threshold, body-charge persistence between events) is the
same physics our pyport simulator handles in full BSIM4 detail; the
demo runs at 100x real-time so we can show the Pavlovian arc in 30 s.

Output: figures/demos/pavlovian_conditioning.mp4 + .png preview
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures/demos"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(7)

# --- Network topology --------------------------------------------------------

N = 8
TONE = 0
REWARD = 1
RESPONSE = 7
HIDDEN = list(range(2, 7))  # 5 hidden cells

# Cell layout for visualization (x, y on a 2D plot)
positions = {
    0: (-3.5,  1.2),   # tone (top-left)
    1: (-3.5, -1.2),   # reward (bottom-left)
    2: (-0.8,  1.5),
    3: (-0.5,  0.5),
    4: ( 0.0, -0.5),
    5: ( 0.5, -1.5),
    6: ( 1.0,  0.8),
    7: ( 3.5,  0.0),   # response (right)
}

# Initial weight matrix W[from, to]
W = np.zeros((N, N))

# Inputs to hidden cells: sparse random
for src in [TONE, REWARD]:
    for h in HIDDEN:
        if rng.random() < 0.6:
            W[src, h] = rng.uniform(0.10, 0.25)

# Hidden to hidden: very sparse weak
for i in HIDDEN:
    for j in HIDDEN:
        if i != j and rng.random() < 0.25:
            W[i, j] = rng.uniform(-0.05, 0.10)

# Hidden to response: small positive
for h in HIDDEN:
    W[h, RESPONSE] = rng.uniform(0.05, 0.12)

# US (reward) → response: STRONG (innate reflex)
W[REWARD, RESPONSE] = 0.95

# CS (tone) → response: WEAK (will be learned)
W[TONE, RESPONSE] = 0.05

W_initial = W.copy()


# --- Cell dynamics: NS-RAM-style leaky integrator with snapback ------------

# Body voltage decay constant
TAU = 14.0          # body-RC, in steps (corresponds to 2 µs at 140 ns/step)
THRESHOLD = 1.0     # body voltage above this triggers snapback (spike)
RESET = -0.3        # body voltage after snapback
SPIKE_DURATION = 2  # how long the cell stays in "fired" state visually

# Hebbian rule for CS→response weight update:
# When the response cell fires AND the tone cell fired recently,
# strengthen W[TONE, RESPONSE] by an amount proportional to the
# coincidence window.
HEBB_RATE = 0.06
HEBB_WINDOW = 8     # tone-firing trace decays over this many steps


# --- Run simulation ----------------------------------------------------------

T = 600  # total timesteps
dt = 1.0  # arbitrary step

# State arrays
Vb = np.zeros((T, N))                 # body voltage per cell, per step
spike = np.zeros((T, N), dtype=bool)  # spike events
W_history = np.zeros((T, 2))          # track W[TONE→RESP] and W[REWARD→RESP]

# Stimulus pattern:
# Trial 1-3: tone alone — no learning yet, response shouldn't fire
# Trial 4-12: tone + reward (paired conditioning trials)
# Trial 13-15: tone alone — should now fire response (conditioned!)
TONE_AMP = 0.8
REWARD_AMP = 1.0
trial_starts = [40, 90, 140,                              # tone-alone (pre)
                200, 240, 280, 320, 360, 400, 440, 480,   # paired
                520, 555, 590]                            # tone-alone (post)
# Phase labels for the timeline annotation
phases = [(0,180,  "PRE: tone alone\n(no response)"),
          (180,490,"CONDITIONING: tone + reward paired"),
          (490,T,  "TEST: tone alone\n→ conditioned response!")]

stimuli = np.zeros((T, N))
PAIRED_RANGE = (180, 490)

def trial_kind(t_start):
    if t_start < PAIRED_RANGE[0]:
        return "tone_only"
    elif t_start < PAIRED_RANGE[1]:
        return "paired"
    else:
        return "tone_only"

for t_start in trial_starts:
    # Pulse shape: 6-step ramp then 4-step plateau (mimics presynaptic input train)
    pulse_len = 10
    kind = trial_kind(t_start)
    for k in range(pulse_len):
        amp = 0.7 + 0.3*np.sin(np.pi*k/pulse_len)
        if t_start + k < T:
            stimuli[t_start + k, TONE] += TONE_AMP * amp
            if kind == "paired":
                # Reward fires ~3 steps AFTER tone (classic delay conditioning)
                if t_start + k + 3 < T:
                    stimuli[t_start + k + 3, REWARD] += REWARD_AMP * amp

# Tone-firing trace for Hebbian rule
tone_trace = 0.0

for t in range(T):
    # Compute drive into each cell from prior-step spikes + external stimulus
    drive = stimuli[t].copy()
    if t > 0:
        drive += spike[t-1].astype(float) @ W   # row-vec @ W → contribution to each cell

    # Update body voltages
    Vb_prev = Vb[t-1] if t > 0 else np.zeros(N)
    Vb[t] = Vb_prev + drive - dt/TAU * Vb_prev

    # Snapback firing
    fired = Vb[t] > THRESHOLD
    spike[t] = fired
    Vb[t][fired] = RESET

    # Update tone-firing trace (decays exponentially)
    tone_trace *= np.exp(-1.0/HEBB_WINDOW)
    if spike[t, TONE]:
        tone_trace = 1.0

    # Hebbian update on CS→response coupling
    # Fires when response cell fires AND tone fired recently
    if spike[t, RESPONSE] and tone_trace > 0.1:
        W[TONE, RESPONSE] = min(W[TONE, RESPONSE] + HEBB_RATE * tone_trace, 1.0)

    W_history[t, 0] = W[TONE, RESPONSE]
    W_history[t, 1] = W[REWARD, RESPONSE]

print(f"Final W[TONE→RESP] = {W[TONE, RESPONSE]:.3f} (started at 0.05)")
print(f"Spikes per cell: {spike.sum(axis=0)}")
print(f"Response cell fires at steps: {np.where(spike[:, RESPONSE])[0][:20]}")


# --- Animation ---------------------------------------------------------------

fig = plt.figure(figsize=(11, 6.5), dpi=110)
gs = fig.add_gridspec(3, 2, width_ratios=[3.0, 1.6], height_ratios=[3.0, 1.0, 1.0],
                       wspace=0.20, hspace=0.45)
ax_net = fig.add_subplot(gs[0, 0])
ax_w = fig.add_subplot(gs[0, 1])
ax_stim = fig.add_subplot(gs[1, :])
ax_resp = fig.add_subplot(gs[2, :])

# --- Network panel layout (precompute static elements) ---
ax_net.set_xlim(-5, 5)
ax_net.set_ylim(-2.5, 2.5)
ax_net.set_aspect("equal")
ax_net.axis("off")

# Static labels
ax_net.text(-3.5, 2.0, "TONE", ha="center", fontsize=10, fontweight="bold", color="#2266aa")
ax_net.text(-3.5, -2.05, "REWARD", ha="center", fontsize=10, fontweight="bold", color="#c0392b")
ax_net.text(3.5, 0.85, "RESPONSE", ha="center", fontsize=10, fontweight="bold", color="#117a4a")
ax_net.text(0.0, 2.3, "5 hidden LIF cells (surrogate, not BSIM4)", ha="center", fontsize=9, style="italic", color="#666")
title = ax_net.text(0.0, -2.4, "", ha="center", fontsize=11, fontweight="bold")

# Edges (drawn dynamically — weight changes for CS→RESP)
edge_lines = {}
for i in range(N):
    for j in range(N):
        if abs(W[i, j]) > 0.01 and i != j:
            x0, y0 = positions[i]
            x1, y1 = positions[j]
            color = "#2266aa" if W[i, j] > 0 else "#c0392b"
            line, = ax_net.plot([x0, x1], [y0, y1], color=color,
                                  linewidth=0.5 + 4*abs(W[i, j]),
                                  alpha=0.5, zorder=1)
            edge_lines[(i, j)] = line

# Nodes
node_circles = []
for i in range(N):
    x, y = positions[i]
    if i == TONE: r = 0.45
    elif i == REWARD: r = 0.45
    elif i == RESPONSE: r = 0.50
    else: r = 0.30
    c = plt.Circle((x, y), r, facecolor="lightgray", edgecolor="black",
                    linewidth=1.2, zorder=3)
    ax_net.add_patch(c)
    node_circles.append(c)

# --- Weight evolution panel ---
ax_w.set_xlim(0, T)
ax_w.set_ylim(0, 1.05)
ax_w.set_xlabel("step", fontsize=8.5)
ax_w.set_ylabel("coupling strength", fontsize=8.5)
ax_w.set_title("learnt vs innate coupling", fontsize=9.5, fontweight="bold")
ax_w.tick_params(labelsize=7.5)
line_cs, = ax_w.plot([], [], color="#2266aa", linewidth=2.0, label="$W_{tone→resp}$ (learnt)")
line_us, = ax_w.plot([], [], color="#c0392b", linewidth=2.0, linestyle="--",
                      label="$W_{reward→resp}$ (innate)")
ax_w.legend(loc="lower right", fontsize=7.5, framealpha=0.92)
ax_w.grid(alpha=0.3, linewidth=0.4)
ax_w.spines[["top", "right"]].set_visible(False)

# Phase shading on weight axis
phase_colors = {"PRE": "#fff5e6", "CONDITIONING": "#e6f4ff", "TEST": "#e6ffe6"}
phase_labels = ["PRE", "CONDITIONING", "TEST"]
for (start, end, label_long), c_key in zip(phases, phase_labels):
    ax_w.axvspan(start, end, alpha=0.4, color=phase_colors[c_key], zorder=0)

# --- Stimulus and response time series ---
for ax, label, color in [(ax_stim, "stimuli", None),
                           (ax_resp, "response cell", "#117a4a")]:
    ax.set_xlim(0, T)
    ax.set_xlabel("step", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

ax_stim.plot(stimuli[:, TONE], color="#2266aa", linewidth=1.0, label="tone")
ax_stim.plot(stimuli[:, REWARD], color="#c0392b", linewidth=1.0, label="reward")
ax_stim.legend(loc="upper right", fontsize=7.5, framealpha=0.92)
ax_stim.set_ylabel("input", fontsize=8.5)
ax_stim.set_ylim(-0.05, 1.4)
for (start, end, _), c_key in zip(phases, phase_labels):
    ax_stim.axvspan(start, end, alpha=0.4, color=phase_colors[c_key], zorder=0)
    midpt = (start + end) / 2
    ax_stim.text(midpt, 1.30, c_key, ha="center", fontsize=8, fontweight="bold", color="#444")

ax_resp.plot(Vb[:, RESPONSE], color="#117a4a", linewidth=1.0)
# Mark spikes
spike_steps = np.where(spike[:, RESPONSE])[0]
ax_resp.plot(spike_steps, [THRESHOLD]*len(spike_steps), marker="^", linestyle="",
              color="#117a4a", markersize=8, label="response fires")
ax_resp.set_ylabel("$V_b$ (response)", fontsize=8.5)
ax_resp.set_ylim(-0.5, 1.3)
ax_resp.axhline(THRESHOLD, color="black", linewidth=0.6, linestyle=":")
ax_resp.text(T-5, THRESHOLD+0.05, "threshold", ha="right", fontsize=7.5, color="#666")
for (start, end, _), c_key in zip(phases, phase_labels):
    ax_resp.axvspan(start, end, alpha=0.4, color=phase_colors[c_key], zorder=0)
ax_resp.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

# Time cursor
cursor_w = ax_w.axvline(0, color="black", linewidth=0.8)
cursor_stim = ax_stim.axvline(0, color="black", linewidth=0.8)
cursor_resp = ax_resp.axvline(0, color="black", linewidth=0.8)


# --- Animation update ---
ANIM_STEP = 4  # advance 4 sim steps per animation frame for speed
frames_total = T // ANIM_STEP

fig.suptitle("LIF Conceptual Illustration of Pavlovian Conditioning (NOT NS-RAM substrate)\n"
             "demonstrates coupling-structure potential, not the silicon's intrinsic learning rule",
              fontsize=11, fontweight="bold", y=0.98)

def animate(frame_idx):
    t = min(frame_idx * ANIM_STEP, T - 1)

    # Update node colours by current Vb
    for i, c in enumerate(node_circles):
        v = Vb[t, i]
        # Spiking? colour bright
        if spike[t, i]:
            c.set_facecolor("yellow")
            c.set_edgecolor("orange")
            c.set_linewidth(2.5)
        else:
            # Activity gradient: gray (rest) → orange (charged) → red (near-spike)
            v_norm = np.clip(v / THRESHOLD, 0, 1) if v > 0 else 0
            if v_norm < 0.01:
                c.set_facecolor("lightgray")
            else:
                # interp gray → red
                rgb = (0.85 + 0.15*v_norm, 0.85*(1-v_norm) + 0.4*v_norm, 0.85*(1-v_norm) + 0.2*v_norm)
                c.set_facecolor(rgb)
            c.set_edgecolor("black")
            c.set_linewidth(1.2)

    # Update CS→RESP edge thickness/colour to show learning
    cs_resp_edge = edge_lines.get((TONE, RESPONSE))
    if cs_resp_edge is not None:
        w_now = W_history[t, 0]
        cs_resp_edge.set_linewidth(0.5 + 4*w_now)
        cs_resp_edge.set_alpha(0.4 + 0.6*w_now)

    # Update weight history lines
    line_cs.set_data(np.arange(t+1), W_history[:t+1, 0])
    line_us.set_data(np.arange(t+1), W_history[:t+1, 1])

    # Time cursors
    cursor_w.set_xdata([t, t])
    cursor_stim.set_xdata([t, t])
    cursor_resp.set_xdata([t, t])

    # Title text — current phase
    for start, end, lbl in phases:
        if start <= t < end:
            title.set_text(lbl)
            break

    return [*node_circles, *edge_lines.values(), line_cs, line_us,
            cursor_w, cursor_stim, cursor_resp, title]


# Render preview PNG at frame just past conditioning phase
frame_preview = int(0.93 * frames_total)
animate(frame_preview)
fig.savefig(OUT / "pavlovian_conditioning.png", bbox_inches="tight", dpi=180)
print(f"saved preview {OUT}/pavlovian_conditioning.png")

# Render mp4
print("rendering mp4 (this may take ~30s)...")
ani = animation.FuncAnimation(fig, animate, frames=frames_total, interval=40, blit=False)
writer = animation.FFMpegWriter(fps=25, bitrate=2000)
ani.save(OUT / "pavlovian_conditioning.mp4", writer=writer, dpi=140)
print(f"saved {OUT}/pavlovian_conditioning.mp4")
plt.close(fig)
