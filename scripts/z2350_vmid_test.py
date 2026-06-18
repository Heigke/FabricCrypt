#!/usr/bin/env python3
"""z2350: Test VMID change in ME IC_BASE_CNTL.

ME uses VMID=15 (privileged firmware VMID). If we change to VMID=0,
the IC might fetch from the kernel's address space where we can
map our own firmware via BAR0.

Strategy:
1. Write NOP sled to VRAM via BAR0 at offset 0x0 (overwrite encrypted FW)
2. Change VMID in IC_BASE_CNTL from 15 to 0
3. Invalidate + prime IC
4. Observe: if ME reads from VMID=0's VA=0x0, it might get our NOPs

RISK: ME will almost certainly crash. GPU MODE2 reset required (~5s).
This is a destructive test — save all state first.

Must run as root.
"""
import struct, time, os, json, mmap

REGS = '/sys/kernel/debug/dri/0/amdgpu_regs'
BAR0 = '/sys/bus/pci/devices/0000:c3:00.0/resource0'

def read_reg(off):
    with open(REGS, 'rb') as f:
        f.seek(off * 4)
        return struct.unpack('<I', f.read(4))[0]

def write_reg(off, val):
    with open(REGS, 'r+b') as f:
        f.seek(off * 4)
        f.write(struct.pack('<I', val))
        f.flush()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

ME_IC_BASE_LO   = 0x5844
ME_IC_BASE_HI   = 0x5845
ME_IC_BASE_CNTL = 0x5846
ME_IC_OP_CNTL   = 0x5847

IC_INVALIDATE = 1 << 0
IC_PRIME      = 1 << 4

# RS64 RISC-V NOP = 0x00000013
RISCV_NOP = 0x00000013

def main():
    results = {}

    log("=== z2350: VMID Change + IC Prime Test ===")
    log("WARNING: This WILL likely crash the ME engine. GPU reset expected.")

    # Save ALL register state
    log("\n=== Pre-test state ===")
    pre_state = {}
    for name, off in [
        ('ME_IC_BASE_LO', ME_IC_BASE_LO),
        ('ME_IC_BASE_HI', ME_IC_BASE_HI),
        ('ME_IC_BASE_CNTL', ME_IC_BASE_CNTL),
        ('ME_IC_OP_CNTL', ME_IC_OP_CNTL),
    ]:
        val = read_reg(off)
        pre_state[name] = val
        log(f"  {name} = 0x{val:08X}")

    results['pre_state'] = {k: f"0x{v:08X}" for k, v in pre_state.items()}

    # Phase 1: Write NOP sled to VRAM offset 0x0 via BAR0
    log("\n=== Phase 1: Write RS64 NOP sled to VRAM offset 0x0 via BAR0 ===")

    # First backup the current VRAM content at offset 0
    fd = os.open(BAR0, os.O_RDWR)
    bar0_size = os.fstat(fd).st_size
    log(f"  BAR0 size: {bar0_size} bytes ({bar0_size/1024/1024:.0f}MB)")

    # Map first 1MB
    mm = mmap.mmap(fd, 0x100000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)

    # Backup first 4KB of VRAM
    backup = mm[0:4096]
    log(f"  Backed up first 4KB of VRAM")

    # Write NOP sled (RS64 RISC-V NOPs: 0x00000013)
    nop_bytes = struct.pack('<I', RISCV_NOP) * 1024  # 4KB of NOPs
    mm[0:4096] = nop_bytes
    mm.flush()

    # Verify write
    readback = struct.unpack('<I', mm[0:4])[0]
    log(f"  VRAM[0x0] = 0x{readback:08X} (expect 0x{RISCV_NOP:08X})")
    write_ok = (readback == RISCV_NOP)
    log(f"  NOP sled write: {'OK' if write_ok else 'FAILED'}")
    results['nop_sled_write'] = write_ok

    if not write_ok:
        log("  ABORTING: Could not write to VRAM")
        mm.close()
        os.close(fd)
        return

    # Phase 2: Change VMID from 15 to 0 in IC_BASE_CNTL
    log("\n=== Phase 2: Change VMID 15->0 in IC_BASE_CNTL ===")
    orig_cntl = read_reg(ME_IC_BASE_CNTL)
    new_cntl = (orig_cntl & ~0xF) | 0  # Clear VMID bits, set to 0
    log(f"  Original IC_BASE_CNTL = 0x{orig_cntl:08X}")
    log(f"  New IC_BASE_CNTL      = 0x{new_cntl:08X} (VMID=0)")

    write_reg(ME_IC_BASE_CNTL, new_cntl)
    check = read_reg(ME_IC_BASE_CNTL)
    log(f"  Readback              = 0x{check:08X}")
    results['vmid_change'] = {
        'original': f"0x{orig_cntl:08X}",
        'target': f"0x{new_cntl:08X}",
        'readback': f"0x{check:08X}",
        'vmid_changed': ((check & 0xF) == 0),
    }

    # Phase 3: Invalidate + Prime
    log("\n=== Phase 3: Invalidate + Prime ME IC ===")
    log("  Invalidating...")
    write_reg(ME_IC_OP_CNTL, IC_INVALIDATE)
    time.sleep(0.05)

    op1 = read_reg(ME_IC_OP_CNTL)
    log(f"  IC_OP_CNTL after inv = 0x{op1:08X}")

    log("  Priming...")
    write_reg(ME_IC_OP_CNTL, IC_PRIME)
    time.sleep(0.5)

    # Phase 4: Check if ME survived
    log("\n=== Phase 4: Check ME status ===")
    try:
        post_cntl = read_reg(ME_IC_BASE_CNTL)
        post_op = read_reg(ME_IC_OP_CNTL)
        log(f"  IC_BASE_CNTL = 0x{post_cntl:08X}")
        log(f"  IC_OP_CNTL   = 0x{post_op:08X}")
        me_alive = True
    except Exception as e:
        log(f"  Register read FAILED: {e}")
        me_alive = False

    if me_alive:
        try:
            with open('/sys/class/drm/card0/device/gpu_busy_percent', 'r') as f:
                busy = f.read().strip()
            log(f"  GPU busy: {busy}%")
        except:
            log("  GPU busy: read failed")

    results['post_prime'] = {
        'me_alive': me_alive,
        'ic_base_cntl': f"0x{post_cntl:08X}" if me_alive else "FAILED",
        'ic_op_cntl': f"0x{post_op:08X}" if me_alive else "FAILED",
    }

    # Phase 5: Restore original state
    log("\n=== Phase 5: Restore original state ===")
    try:
        # Restore VMID
        write_reg(ME_IC_BASE_CNTL, orig_cntl)
        time.sleep(0.05)

        # Restore VRAM
        mm[0:4096] = backup
        mm.flush()
        log("  VMID and VRAM restored")

        # Re-prime with original firmware
        write_reg(ME_IC_OP_CNTL, IC_INVALIDATE)
        time.sleep(0.05)
        write_reg(ME_IC_OP_CNTL, IC_PRIME)
        time.sleep(0.1)

        final_cntl = read_reg(ME_IC_BASE_CNTL)
        log(f"  Final IC_BASE_CNTL = 0x{final_cntl:08X} {'OK' if final_cntl == orig_cntl else 'MISMATCH'}")
        results['restore'] = 'OK'
    except Exception as e:
        log(f"  Restore FAILED: {e}")
        results['restore'] = str(e)

    mm.close()
    os.close(fd)

    # Save results
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2350_vmid_test.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {outpath}")

if __name__ == "__main__":
    main()
