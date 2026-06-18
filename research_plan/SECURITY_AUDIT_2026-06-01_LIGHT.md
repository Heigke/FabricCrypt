# Light Security Audit — 2026-06-01

Scope: ikaros (local laptop), daedalus (192.168.0.37/.40), zgx (192.168.0.41).
Method: lightweight checks only (no AIDE init, no lynis/rkhunter full scans, no debsums).
Pre-audit cleanup: `pkill` of any leftover aide/lynis/rkhunter/chkrootkit on all three hosts — **none were running** (the previous heavy-scan worry is resolved; no zombie aide processes were present on any machine).

---

### ikaros
- **Ports**: 11 listening. All bound to loopback except `sshd :22` (any) and `xrdp :3389` (any). Unusual: `xrdp :3389` exposed on all interfaces, `ollama :11434` (localhost only — OK).
- **Established conns**: claude (Anthropic API 160.79.104.10:443, 34.149.66.137:443), outbound ssh→daedalus, inbound ssh from zgx. All expected.
- **eBPF**: progs=0, maps=0, links=0, kprobes=0, uprobes=0.
- **Kernel**: lockdown=`[none]` (none active), module_sig_enforce=unset, tainted=**12292** = bits 2+12+13 (unsigned out-of-tree module + unsigned firmware = amdgpu). Benign for this hardware.
- **Deleted exes**: NONE.
- **Anonymous RWX maps procs**: 12 (normal — JIT runtimes: node, chrome, etc.).
- **LD_PRELOAD**: empty (file and env).
- **authorized_keys** (/home/ikaros): 2 keys — `daedalus@daedalus`, `naorw@zgx-5175`. Both expected. /root: empty.
- **Crontabs**: none (empty `/var/spool/cron/crontabs/`).
- **Recent system bin changes (30d)**: 20+ — all from routine apt upgrades (perf, rsync, dpkg-*, libpng, evince, bind utils). Benign.
- **New systemd services 30d**: snap mounts + `ollama.service`. Benign.
- **Failed ssh auths 7d**: 6 on 2026-05-30 from 192.168.0.2 (LAN, single burst — likely user typo/SSH client retry). No external bruteforce.
- **dmesg anomalies 24h**: 2× `whiptail` segfault in libnewt (cosmetic TUI crash, no security implication); standard amdgpu unsigned-module taint message; PCIe DPC capability notes (informational).
- **Verdict**: **CLEAN** (one item to note: xrdp :3389 listening on all interfaces).

### daedalus
- **Ports**: 8 listening. sshd :22 (any), postfix `master :25` (any), cupsd/systemd-resolve on loopback.
- **Established conns**: only inbound ssh from ikaros (192.168.0.35). No external.
- **eBPF**: progs=0, maps=0, links=0, kprobes=0, uprobes=0.
- **Kernel**: lockdown=`[none]`, module_sig_enforce=unset, tainted=**0**. Clean.
- **Deleted exes**: `/proc/463/exe -> /usr/sbin/plymouthd (deleted)` — plymouthd holding an unlinked file across a package upgrade. Benign; harmless restart will clear it (`systemctl restart plymouth-start.service` or ignore).
- **Anonymous RWX maps procs**: 5 (normal).
- **LD_PRELOAD**: empty.
- **authorized_keys** (/home/daedalus): 5 keys — `your_email@example.com` (default placeholder, ssh-keygen factory comment), `ericbergvall@MacBook-Air`, `minos@minos`, `daedalus@daedalus`, `naorw@zgx-5175`. The `your_email@example.com` comment is generic; verify it's the user's own (low risk on LAN-only host but **recommend confirming/removing if unknown**). /root: empty.
- **Crontabs**: none.
- **Recent system bin changes (30d)**: postfix toolchain only (recent postfix package upgrade). Benign.
- **New systemd services 30d**: snap mounts, `ollama.service.d/rocm.conf`, **`agent-triage.service`** (your project) — all expected.
- **Failed ssh auths 7d**: 1 — banner exchange format error from 192.168.0.35 (LAN, likely sshpass quirk). Not a bruteforce.
- **dmesg anomalies 24h**: only RAS init + hp_wmi cosmetic error. Clean.
- **Verdict**: **CLEAN** (verify the `your_email@example.com` ssh key is yours; remove if not recognized).

### zgx (zgx-5175)
- **Ports**: 9 listening. sshd :22 (any), postfix `master :25` (any), `dashboard-servi :11000` (localhost — your C2 dashboard), cupsd/resolve on loopback.
- **Established conns**: only inbound ssh from ikaros. No external.
- **eBPF**: many cgroup_device/cgroup_skb progs (named `sd_fw_egress`, `sd_fw_ingress`, `sd_devices`, `s_snapd_desktop`, `s_firefox_hook_*`, `s_thunderbird_h*`, `s_vivaldi_hook_*`, `s_firmware_upda*`). **These are all systemd cgroup device/network filters and snapd application hooks — normal kernel/snapd behavior, all owned by uid=0 or uid=1000.** maps=8 hash tables for the snap hooks. links=0, kprobes=0, uprobes=0. No suspicious unnamed progs.
- **Kernel**: lockdown=`[integrity]` (active — good!), module_sig_enforce=unset, tainted=**4096** = bit 12 only (out-of-tree module loaded at some point). Benign.
- **Deleted exes**: NONE.
- **Anonymous RWX maps procs**: 4 (normal).
- **LD_PRELOAD**: empty.
- **authorized_keys** (/home/naorw): 4 keys — `naorw@zgx-5175`, `ericbergvall@MacBook-Air`, `minos@minos`, `daedalus@daedalus`. All expected. /root: empty.
- **Crontabs**: none.
- **Recent system bin changes (30d)**: postfix toolchain only. Benign.
- **New systemd services 30d**: snap mounts, `c2-dashboard.service`, `agent-strategist.service`, `litellm.service` — all your project services. Expected.
- **Failed ssh auths 7d**: 0 (clean).
- **dmesg anomalies 24h**: 0 (clean).
- **Verdict**: **CLEAN**.

---

## TL;DR

All three machines are **CLEAN**. No rootkit indicators, no unsigned/unnamed eBPF programs, no kprobes/uprobes, no LD_PRELOAD hijacks, no deleted-binary processes (except a harmless plymouthd post-upgrade on daedalus), no unexplained crontabs, no external established connections, no external SSH bruteforce. Kernel taints are all benign (amdgpu unsigned firmware on ikaros, single out-of-tree bit on zgx, zero on daedalus). zgx has kernel `lockdown=integrity` active — best of the three.

The previously-feared `aide --init` heat issue was **not present** — no aide/lynis/rkhunter processes were running on any host when audit started.

## Actionable findings (low priority)

1. **ikaros**: `xrdp :3389` listens on all interfaces. If you don't actively need remote desktop from the LAN, mask it: `sudo systemctl disable --now xrdp xrdp-sesman`. Otherwise firewall-restrict to LAN subnet.
2. **daedalus**: `/home/daedalus/.ssh/authorized_keys` contains a key with the default comment `your_email@example.com`. Verify it's yours; if not recognized, remove that single line.
3. **daedalus**: stale `/proc/463/exe -> plymouthd (deleted)` from a package upgrade — purely cosmetic; will clear on next reboot.
4. **All hosts**: kernel `module_sig_enforce` is unset and lockdown is `none` on ikaros/daedalus. Not actionable on this hardware (amdgpu requires unsigned firmware on ikaros), but noted.
5. None of the failed SSH attempts are from external addresses — no public-internet attack surface visible.

No urgent action required. Time budget: ~2 min wall-clock, no CPU/thermal load.
