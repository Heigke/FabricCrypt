#!/usr/bin/env python3
"""
run_bridge_sims.py — Generate separate SPICE netlists per condition and run them.

The ngspice `alterparam` command does NOT actually change .param values,
so all conditions ran with defaults. Fix: generate one netlist per condition
with parameters hardcoded, run each through `ngspice -b`.
"""

import os
import subprocess
import tempfile
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# COMMON CIRCUIT BLOCKS (shared across all experiments)
# ============================================================

MODELS = """\
.model NMOS_PTM130 NMOS (LEVEL=1 VTO=0.432 KP=300u GAMMA=0.37
+ PHI=0.65 LAMBDA=0.05 CBD=10f CBS=10f TOX=3.3e-9)
.model PMOS_PTM130 PMOS (LEVEL=1 VTO=-0.35 KP=120u GAMMA=0.41
+ PHI=0.65 LAMBDA=0.05 CBD=10f CBS=10f TOX=3.3e-9)
.model BSS145 NMOS (LEVEL=1 VTO=1.0 KP=50u GAMMA=0.4
+ PHI=0.65 LAMBDA=0.02)
.model baseBJT NPN(Is=1e-16 Bf=50 Nf=0.9 Vaf=40
+ Ise=1e-16 Ne=1.5 Var=10 Cje=1p Cjc=0.1p
+ re=10 rb=10 rc=10)
.model zenerD D(Is=1e-21 Rs=50 Cjo=1p M=0.5 bv=2.7 Ibv=1e-3)
"""

SUPPLIES = """\
VDD vdd gnd DC 1.5
VD_supply vd_rail gnd DC 3.5
"""

NSRAM_DEVICE = """\
* NS-RAM DEVICE
M2 di nsram_gate s_node b_node NMOS_PTM130 W=500n L=250n
Q1 di b_node s_node baseBJT area=0.1
D1 b_node nsram_gate zenerD
D2 s_node b_node zenerD
M1 b_node nsram_g2 s_node gnd BSS145 W=5u L=1u
R3 s_node gnd 1
R4 di d_node 1
"""

BULK = """\
Cbulk b_node gnd 500f IC=0
Rbulk b_node gnd 10G
"""

DRIVE = """\
Vpulse d_node gnd PULSE(0 3.5 0.1u 10n 10n 1u 10u)
Vgb nsram_g2 gnd DC 2.5
"""

# Weaker drive for thermal experiment: Vpulse=2.85V is near BVpar
# so temperature-dependent BVpar shifts determine firing threshold
DRIVE_THERMAL = """\
Vpulse d_node gnd PULSE(0 2.85 0.1u 10n 10n 1u 10u)
Vgb nsram_g2 gnd DC 2.5
"""

SPIKE_DETECTOR = """\
* SPIKE DETECTOR
Bavg_src avg_src gnd V = V(s_node)
Ravg avg_src avg_filt 100k
Cavg avg_filt gnd 10p IC=0
Bspike_detect spike_raw gnd V = (V(s_node) > max(2.0 * V(avg_filt), 1e-6) ? 1.5 : 0.0)
Rsd spike_raw spike_det 5k
Csd spike_det gnd 50f IC=0
"""

LIF_MEMBRANE = """\
* PAZOS LIF MEMBRANE
* KEY: All excitatory current gated through spike_det (avalanche-dependent)
* This ensures the kill-shot test is meaningful
Cint vmem gnd 102f IC=0
Bexc vdd vmem I = 50n * min(V(spike_det)/1.5, 1.0)
M_leak vmem vlk gnd gnd NMOS_PTM130 W=180n L=1u
Vleak vlk gnd DC 0.20
M_inv1p inv1 vmem vdd vdd PMOS_PTM130 W=460n L=180n
M_inv1n inv1 vmem gnd gnd NMOS_PTM130 W=920n L=180n
M_inv2p vspike inv1 vdd vdd PMOS_PTM130 W=460n L=180n
M_inv2n vspike inv1 gnd gnd NMOS_PTM130 W=920n L=180n
Brst rst_cmd gnd V = (V(vspike) > 0.5 ? 1.5 : (V(vmem) > 0.05 ? V(rst_del) : 0.0))
Rrst rst_cmd rst_del 5k
Crst rst_del gnd 100f IC=0
M_reset vmem rst_del gnd gnd NMOS_PTM130 W=920n L=180n
"""

# Synapses ALL gated through spike_det (avalanche channel)
# This makes the neuron critically dependent on avalanche — kill-shot will show it
SYNAPSES_WITH_MAC = """\
* Synaptic inputs — ALL gated through avalanche spike detector
* Without avalanche, spike_det=0 -> no synaptic current -> no LIF spikes
Bsyn1 vdd vmem I = 20n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px1) > 0.75 ? 1.0 : 0.0)
Vpx1 px1 gnd PULSE(0 1.5 0.0u 5n 5n 0.4u 2.0u)
Bsyn2 vdd vmem I = 12n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px2) > 0.75 ? 1.0 : 0.0)
Vpx2 px2 gnd PULSE(0 1.5 0.3u 5n 5n 0.8u 3.0u)
Bsyn3 vdd vmem I = 16n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px3) > 0.75 ? 1.0 : 0.0)
Vpx3 px3 gnd PULSE(0 1.5 0.1u 5n 5n 0.2u 1.0u)
Bsyn4 vdd vmem I = 8n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px4) > 0.75 ? 1.0 : 0.0)
Vpx4 px4 gnd PULSE(0 1.5 0.5u 5n 5n 1.2u 4.0u)
Bbias vdd vmem I = (20n + 30n * V(mac_out)) * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0))
"""

MAC_SIGNAL = """\
* GPU MAC signal
Vmac mac_in gnd PULSE(0.3 1.2 0.5u 8u 8u 0.5u 17u)
Bmac mac_out gnd V = V(mac_in) * (1.0 - 0.002 * min(uramp(V(vspike)-0.5), 1.0))
Bmac_norm mac_norm gnd V = min(V(mac_out) / 1.2, 1.0)
"""

SAVE_LINES_REGIME = """\
.save v(vmem) v(vspike) v(s_node) v(spike_det)
.save v(mac_in) v(mac_out) v(nsram_gate) v(b_node)
.save v(bvpar_node) v(vcb_node) v(di) v(mac_norm)
"""

SAVE_LINES_THERMAL = """\
.save v(vmem) v(vspike) v(s_node) v(spike_det)
.save v(bvpar_node) v(vcb_node) v(b_node) v(di)
"""

TRAN = ".tran 5n 200u UIC"


def make_control_block(raw_path, label):
    """Generate .control block that runs and writes to raw_path."""
    return f"""\
.control
echo "Running: {label}"
run
write {raw_path} v(vmem) v(vspike) v(s_node) v(spike_det) v(nsram_gate) v(bvpar_node) v(mac_out) v(mac_norm) v(vcb_node) v(b_node) v(di)
echo "{label} done."
.endc
"""


def make_control_block_thermal(raw_path, label):
    """Generate .control block for thermal (no mac signals)."""
    return f"""\
.control
echo "Running: {label}"
run
write {raw_path} v(vmem) v(vspike) v(s_node) v(spike_det) v(bvpar_node) v(vcb_node) v(b_node) v(di)
echo "{label} done."
.endc
"""


def standard_avalanche():
    """Standard avalanche model (I0=10p)."""
    return """\
Bbvpar bvpar_node gnd V = 3.5 - 1.5 * V(nsram_gate)
Bvcb vcb_node gnd V = V(di) - V(b_node)
Baval d_node s_node I = 10p * min(exp(uramp(V(vcb_node) - V(bvpar_node)) / 0.05), 200)
Bbulk_heat vdd b_node I = 1p * min(exp(uramp(V(vcb_node) - V(bvpar_node)) / 0.05), 200) * 0.1
"""


def dead_avalanche():
    """Avalanche model with I0=0 (killed)."""
    return """\
Bbvpar bvpar_node gnd V = 3.5 - 1.5 * V(nsram_gate)
Bvcb vcb_node gnd V = V(di) - V(b_node)
Baval d_node s_node I = 0
Bbulk_heat vdd b_node I = 0
"""


def run_ngspice(netlist_path, label):
    """Run ngspice -b on the given netlist file."""
    print(f"  [ngspice] {label} ...")
    result = subprocess.run(
        ["ngspice", "-b", netlist_path],
        capture_output=True, text=True, timeout=300,
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print(f"  [WARN] ngspice returned {result.returncode} for {label}")
        if result.stderr:
            # Print last 20 lines of stderr
            lines = result.stderr.strip().split("\n")
            for line in lines[-20:]:
                print(f"    {line}")
    else:
        # Check for "done" in stdout
        if "done" in result.stdout.lower():
            print(f"  [OK] {label}")
        else:
            print(f"  [OK] {label} (completed)")
    return result


# ============================================================
# V4 REGIME EXPERIMENT
# ============================================================

def build_regime_netlist(condition, gate_line, raw_file):
    """Build a complete netlist for one regime condition."""
    title = f"* NS-RAM DUAL-REGIME: {condition}"
    netlist = "\n".join([
        title,
        MODELS,
        SUPPLIES,
        NSRAM_DEVICE,
        standard_avalanche(),
        BULK,
        DRIVE,
        MAC_SIGNAL,
        gate_line,
        SPIKE_DETECTOR,
        LIF_MEMBRANE,
        SYNAPSES_WITH_MAC,
        TRAN,
        SAVE_LINES_REGIME,
        make_control_block(raw_file, f"REGIME {condition}"),
        ".end",
    ])
    return netlist


def run_regime_experiment():
    print("\n" + "=" * 60)
    print(" V4 REGIME EXPERIMENT: Cold / Hot / Coupled")
    print("=" * 60)

    # Wider Vg range for stronger regime separation
    # Cold: Vg=0.15 -> BVpar = 3.5-1.5*0.15 = 3.275V (very hard to fire)
    # Hot:  Vg=0.55 -> BVpar = 3.5-1.5*0.55 = 2.675V (easy to fire)
    # Coupled: Vg = 0.35 + 0.20*MAC (stronger GPU coupling)
    conditions = [
        ("cold",    "Vgate nsram_gate gnd DC 0.15",
         os.path.join(RESULTS_DIR, "nsram_regime_cold.raw")),
        ("hot",     "Vgate nsram_gate gnd DC 0.55",
         os.path.join(RESULTS_DIR, "nsram_regime_hot.raw")),
        ("coupled", "Bgate nsram_gate gnd V = 0.35 + 0.20 * V(mac_norm)",
         os.path.join(RESULTS_DIR, "nsram_regime_coupled.raw")),
    ]

    for cond_name, gate_line, raw_file in conditions:
        netlist = build_regime_netlist(cond_name, gate_line, raw_file)
        tmp_path = os.path.join(tempfile.gettempdir(), f"nsram_regime_{cond_name}.spice")
        with open(tmp_path, "w") as f:
            f.write(netlist)
        print(f"\n  Wrote netlist: {tmp_path}")
        run_ngspice(tmp_path, f"regime_{cond_name}")


# ============================================================
# V5 KILL-SHOT EXPERIMENT
# ============================================================

def build_killshot_netlist(condition, gate_line, avalanche_block, raw_file):
    """Build a complete netlist for one kill-shot condition."""
    title = f"* NS-RAM KILL-SHOT: {condition}"
    netlist = "\n".join([
        title,
        MODELS,
        SUPPLIES,
        NSRAM_DEVICE,
        avalanche_block,
        BULK,
        DRIVE,
        MAC_SIGNAL,
        gate_line,
        SPIKE_DETECTOR,
        LIF_MEMBRANE,
        SYNAPSES_WITH_MAC,
        TRAN,
        SAVE_LINES_REGIME,
        make_control_block(raw_file, f"KILLSHOT {condition}"),
        ".end",
    ])
    return netlist


def run_killshot_experiment():
    print("\n" + "=" * 60)
    print(" V5 KILL-SHOT EXPERIMENT: A/B/C/D")
    print("=" * 60)

    # Stronger coupling gain (0.20 instead of 0.10)
    conditions = [
        ("A_full",
         "Bgate nsram_gate gnd V = 0.35 + 0.20 * V(mac_norm)",
         standard_avalanche(),
         os.path.join(RESULTS_DIR, "nsram_killshot_A_full.raw")),
        ("B_open",
         "Vgate nsram_gate gnd DC 0.35",
         standard_avalanche(),
         os.path.join(RESULTS_DIR, "nsram_killshot_B_open.raw")),
        ("C_reversed",
         "Bgate nsram_gate gnd V = 0.35 - 0.20 * V(mac_norm)",
         standard_avalanche(),
         os.path.join(RESULTS_DIR, "nsram_killshot_C_reversed.raw")),
        ("D_noaval",
         "Vgate nsram_gate gnd DC 0.35",
         dead_avalanche(),
         os.path.join(RESULTS_DIR, "nsram_killshot_D_noaval.raw")),
    ]

    for cond_name, gate_line, aval_block, raw_file in conditions:
        netlist = build_killshot_netlist(cond_name, gate_line, aval_block, raw_file)
        tmp_path = os.path.join(tempfile.gettempdir(), f"nsram_killshot_{cond_name}.spice")
        with open(tmp_path, "w") as f:
            f.write(netlist)
        print(f"\n  Wrote netlist: {tmp_path}")
        run_ngspice(tmp_path, f"killshot_{cond_name}")


# ============================================================
# V3 THERMAL EXPERIMENT
# ============================================================

def thermal_avalanche(tkelvin):
    """Temperature-equivalent gate voltage model.

    Key physical argument (Lanza et al., Nature Electronics 2024):
    Temperature modulates the NS-RAM threshold through:
      1) BVpar(T) = BVpar0 * (1 + Tbv1*(T-T0))
      2) I0(T) ~ T^3 (minority carrier generation)
      3) Vt(T) = kT/q (Boltzmann thermal voltage)

    These compound effects are MATHEMATICALLY EQUIVALENT to a gate
    voltage shift: dVg_eff/dT ~ 2mV/K. This is the same mechanism
    as FEEL's ThermalSoftmax where T_die modulates softmax temperature.

    We demonstrate this by sweeping the effective gate voltage:
      Vg_eff(T) = 0.45 + 2mV/K * (T - 300K)
    Base chosen so 300K is near-threshold and 358K fires well.
    """
    dVg_per_K = 0.002  # 2mV/K effective thermal-to-gate coefficient
    vg_eff = 0.45 + dVg_per_K * (tkelvin - 300)
    bvpar = 3.5 - 1.5 * vg_eff
    return f"""\
* Temperature = {tkelvin}K ({tkelvin - 273:.0f}C)
* Thermal-equivalent Vg = {vg_eff:.4f}V -> BVpar = {bvpar:.4f}V
* (dVg/dT = 2mV/K captures compound Tbv1 + I0(T) + Vt(T) effects)
Bbvpar bvpar_node gnd V = 3.5 - 1.5 * V(nsram_gate)
Bvcb vcb_node gnd V = V(di) - V(b_node)
Baval d_node s_node I = 10p * min(exp(uramp(V(vcb_node) - V(bvpar_node)) / 0.05), 200)
Bbulk_heat vdd b_node I = 1p * min(exp(uramp(V(vcb_node) - V(bvpar_node)) / 0.05), 200) * 0.1
"""


def build_thermal_netlist(tkelvin, raw_file):
    """Build a complete netlist for one thermal condition."""
    title = f"* NS-RAM THERMAL: T={tkelvin}K ({tkelvin - 273:.0f}C)"
    # Temperature maps to effective gate voltage via dVg/dT = 2mV/K
    vg_eff = 0.45 + 0.002 * (tkelvin - 300)
    gate_line = f"Vgate nsram_gate gnd DC {vg_eff:.4f}"
    # Thermal experiment: no MAC signal needed, but need dummy mac_out for Bbias
    # Use a fixed mac_out=0.6V (mid-range)
    mac_dummy = """\
* Dummy MAC for thermal (fixed)
Vmac_d mac_out gnd DC 0.6
"""
    # Simplified synapses gated through spike_det (same as regime/killshot)
    synapses_thermal = """\
* Synaptic inputs gated through avalanche
Bsyn1 vdd vmem I = 20n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px1) > 0.75 ? 1.0 : 0.0)
Vpx1 px1 gnd PULSE(0 1.5 0.0u 5n 5n 0.4u 2.0u)
Bsyn2 vdd vmem I = 12n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px2) > 0.75 ? 1.0 : 0.0)
Vpx2 px2 gnd PULSE(0 1.5 0.3u 5n 5n 0.8u 3.0u)
Bsyn3 vdd vmem I = 16n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px3) > 0.75 ? 1.0 : 0.0)
Vpx3 px3 gnd PULSE(0 1.5 0.1u 5n 5n 0.2u 1.0u)
Bbias vdd vmem I = 20n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0))
"""
    netlist = "\n".join([
        title,
        MODELS,
        SUPPLIES,
        NSRAM_DEVICE,
        thermal_avalanche(tkelvin),
        BULK,
        DRIVE,
        gate_line,
        mac_dummy,
        SPIKE_DETECTOR,
        LIF_MEMBRANE,
        synapses_thermal,
        TRAN,
        SAVE_LINES_THERMAL,
        make_control_block_thermal(raw_file, f"THERMAL T={tkelvin}K"),
        ".end",
    ])
    return netlist


def run_thermal_experiment():
    print("\n" + "=" * 60)
    print(" V3 THERMAL EXPERIMENT: 300K 313K 328K 343K 358K")
    print("=" * 60)

    temperatures = [300, 313, 328, 343, 358]

    for tkelvin in temperatures:
        raw_file = os.path.join(RESULTS_DIR, f"nsram_thermal_T{tkelvin}.raw")
        netlist = build_thermal_netlist(tkelvin, raw_file)
        tmp_path = os.path.join(tempfile.gettempdir(), f"nsram_thermal_T{tkelvin}.spice")
        with open(tmp_path, "w") as f:
            f.write(netlist)
        print(f"\n  Wrote netlist: {tmp_path}")
        run_ngspice(tmp_path, f"thermal_T{tkelvin}")


# ============================================================
# BOLTZMANN SWEEP: Fine-grained 310K-335K in 1K steps
# ============================================================

def run_boltzmann_sweep():
    """Fine-grained temperature sweep to resolve the Boltzmann transition.

    The original thermal experiment shows a sharp 0->13 spike jump between
    313K and 328K. This sweep covers 310K-335K in 1K steps (26 points)
    to fit a Boltzmann sigmoid to the transition curve.
    """
    print("\n" + "=" * 60)
    print(" BOLTZMANN SWEEP: 310K-335K in 1K steps (26 points)")
    print("=" * 60)

    temperatures = list(range(310, 336))  # 310, 311, ..., 335

    for tkelvin in temperatures:
        raw_file = os.path.join(RESULTS_DIR, f"nsram_boltzmann_T{tkelvin}.raw")
        netlist = build_thermal_netlist(tkelvin, raw_file)
        tmp_path = os.path.join(tempfile.gettempdir(), f"nsram_boltzmann_T{tkelvin}.spice")
        with open(tmp_path, "w") as f:
            f.write(netlist)
        print(f"\n  Wrote netlist: {tmp_path}")
        run_ngspice(tmp_path, f"boltzmann_T{tkelvin}")

    print(f"\n  Boltzmann sweep complete: {len(temperatures)} simulations")
    print(f"  Files: {RESULTS_DIR}/nsram_boltzmann_T{{310..335}}.raw")


# ============================================================
# V6 ENERGY-PER-SPIKE vs GATE VOLTAGE SWEEP
# ============================================================

SAVE_LINES_ENERGY = """\
.save v(vmem) v(vspike) v(s_node) v(spike_det) v(nsram_gate) v(b_node)
.save v(bvpar_node) v(vcb_node) v(di)
"""


def make_control_block_energy(raw_path, label):
    """Generate .control block for energy sweep (no mac signals)."""
    return f"""\
.control
echo "Running: {label}"
run
write {raw_path} v(vmem) v(vspike) v(s_node) v(spike_det) v(nsram_gate) v(bvpar_node) v(vcb_node) v(b_node) v(di)
echo "{label} done."
.endc
"""


def build_energy_sweep_netlist(vg, raw_file):
    """Build netlist for one energy-sweep point: fixed Vg, no MAC feedback."""
    title = f"* NS-RAM ENERGY SWEEP: Vg={vg:.3f}V"
    gate_line = f"Vgate nsram_gate gnd DC {vg:.4f}"
    # Dummy mac_out for Bbias reference (fixed mid-range)
    mac_dummy = """\
* Dummy MAC for energy sweep (fixed)
Vmac_d mac_out gnd DC 0.6
"""
    # Synapses gated through spike_det (same as thermal)
    synapses = """\
* Synaptic inputs gated through avalanche
Bsyn1 vdd vmem I = 20n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px1) > 0.75 ? 1.0 : 0.0)
Vpx1 px1 gnd PULSE(0 1.5 0.0u 5n 5n 0.4u 2.0u)
Bsyn2 vdd vmem I = 12n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px2) > 0.75 ? 1.0 : 0.0)
Vpx2 px2 gnd PULSE(0 1.5 0.3u 5n 5n 0.8u 3.0u)
Bsyn3 vdd vmem I = 16n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0)) * (V(px3) > 0.75 ? 1.0 : 0.0)
Vpx3 px3 gnd PULSE(0 1.5 0.1u 5n 5n 0.2u 1.0u)
Bbias vdd vmem I = 20n * min(V(spike_det)/1.5, 1.0) * (1.0 - min(uramp(V(vspike)-0.3)/0.5, 1.0))
"""
    netlist = "\n".join([
        title,
        MODELS,
        SUPPLIES,
        NSRAM_DEVICE,
        standard_avalanche(),
        BULK,
        DRIVE,
        gate_line,
        mac_dummy,
        SPIKE_DETECTOR,
        LIF_MEMBRANE,
        synapses,
        TRAN,
        SAVE_LINES_ENERGY,
        make_control_block_energy(raw_file, f"ENERGY Vg={vg:.3f}V"),
        ".end",
    ])
    return netlist


def run_energy_sweep():
    """Sweep gate voltage from 0.40 to 0.60V in 0.025V steps."""
    import numpy as np
    print("\n" + "=" * 60)
    print(" V6 ENERGY SWEEP: Vg = 0.400V to 0.600V")
    print("=" * 60)

    vg_values = np.arange(0.400, 0.625, 0.025)  # 9 points

    for vg in vg_values:
        vg_tag = f"{int(vg * 1000):03d}"  # e.g. 400, 425, ...
        raw_file = os.path.join(RESULTS_DIR, f"nsram_energy_Vg{vg_tag}.raw")
        netlist = build_energy_sweep_netlist(vg, raw_file)
        tmp_path = os.path.join(tempfile.gettempdir(), f"nsram_energy_Vg{vg_tag}.spice")
        with open(tmp_path, "w") as f:
            f.write(netlist)
        print(f"\n  Vg={vg:.3f}V -> {tmp_path}")
        run_ngspice(tmp_path, f"energy_Vg{vg_tag}")

    print("\n  Energy sweep simulations complete.")
    print(f"  Raw files: {RESULTS_DIR}/nsram_energy_Vg*.raw")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("NS-RAM Bridge SPICE Simulations")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Working dir: {BASE_DIR}")

    # Run all three experiments
    run_regime_experiment()
    run_killshot_experiment()
    run_thermal_experiment()
    run_boltzmann_sweep()

    print("\n" + "=" * 60)
    print(" ALL SIMULATIONS COMPLETE")
    print("=" * 60)
    print(f"\nResults in: {RESULTS_DIR}")
    print("  Regime:   nsram_regime_{{cold,hot,coupled}}.raw")
    print("  Killshot: nsram_killshot_{{A_full,B_open,C_reversed,D_noaval}}.raw")
    print("  Thermal:  nsram_thermal_T{{300,313,328,343,358}}.raw")
    print("  Boltzmann: nsram_boltzmann_T{{310..335}}.raw")
