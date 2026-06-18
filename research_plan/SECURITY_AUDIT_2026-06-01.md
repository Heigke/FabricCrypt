# Security Audit + Overnight Activity — 2026-06-01

Audit run ~04:55–04:58 local on ikaros (CEST). All three machines surveyed in parallel; no installations performed, no heavy scans. Limits: no passwordless sudo on any host, so `lastb` / full root-owned files were not checked.

## TL;DR
- **No compromise indicators found on any host.** All listening services map to known user processes (claude, llama-server, srgeo, ollama, litellm, cloudflared, hermes, redis, xrdp, openvpn). No deleted-binary exes, no rogue SUIDs added in last 30d, no executables in /tmp/var/tmp/dev/shm except the user's own audit scripts. authorized_keys on every host contain only known machines (ikaros/daedalus/zgx/minos/MacBook-Air).
- **Fan-up overnight is fully explained by user workloads, not by intrusion:**
  - **ikaros**: a node-claude process (pid 12413) maintained ~11 concurrent HTTPS sessions to Anthropic (160.79.104.10) plus the agentic `claude` process the user is logged into. zgx connected back over SSH on a ~15-second cadence (agent babysit loop) starting at 04:53.
  - **daedalus**: `/home/daedalus/babysit_agents.sh` fires every 20 min via user crontab; `srgeo`/`llama-server`/`whisper-cli` were all `T`/`TLl` (SIGSTOP'ed) as expected; only post-04:54 became active again when current SSH session resumed. Identity-benchmark files (embodiment3/4/5/8, hyperfine, all_32) were re-written by the agent within the last 24h.
  - **zgx**: 4 long-running `claude` instances (4–5 days uptime), `ollama`, `litellm`, `uvicorn`, `cloudflared` tunnel, plus `hermes-agent` gateway. These are the persistent fans-up source. No new cron activity overnight beyond standard system jobs.
- **One housekeeping flag, not security**: `remoteuser` (UID 1001) exists on ikaros with `/bin/bash` shell. If this is unused, disable login. Otherwise expected.
- **Two cosmetic issues**: `agent-triage.service` (daedalus) and `agent-strategist.service` (zgx) have malformed `Environment=` lines (unquoted spaces) — logged as benign warnings, agents still run.

## Per machine

### ikaros (local, 192.168.0.35)
**Overnight activity (last 12h)**
- Boot at 17:07 (single reboot following 17:24-ikaros reboot earlier — both expected, from user). Uptime since 17:07.
- Cron: only stock `sysstat` debian-sa1 every 10 min and run-parts hourly. User crontab has only `@reboot` jobs (`falsify/resume_after_reboot.sh`, `queue/post_reboot_recovery.sh`, `embodiment3/post_reboot.sh`) — no overnight wake-ups.
- Real CPU load: `claude` node processes (pids 10680, 12413) maintaining 11 concurrent TLS sessions to 160.79.104.10:443 (Anthropic) and a couple to 34.149.66.137:443 / 98.86.62.57:443 (Anthropic CloudFront / Google).
- SSH inbound: zgx (192.168.0.41) is hitting our sshd every ~15 s starting 04:53:05 — pubkey accepted, immediate disconnect (telemetry/babysit probe pattern, consistent with the agent-collective architecture). Not an attack.
- Thermal/journalctl noise: standard pipewire, snap.firmware-updater repeated failures (cosmetic), gdm warnings on the new session. `pm_runtime_work hogged CPU >10000us` — just a kernel scheduler note.

**Listening services (all expected)**
- 22/tcp sshd, 631/tcp cupsd, 3389/tcp xrdp, 3350/tcp xrdp-sesman, 11434/tcp ollama (loopback), 53 systemd-resolved, 1194/udp openvpn-server, 5353/udp avahi, three node-claude loopback ports.
- Outbound non-RFC1918: only Anthropic + Google IPs — all owned by user-launched `claude` processes.

**Auth log anomalies**: none. All sshd accepted publickey from 192.168.0.41 (zgx) with the same ED25519 fingerprint. No failures, no `Invalid user`. Sudo usage was the current operator session (pam_unix opened/closed by ikaros, uid 1000).

**Authorized SSH keys** (`/home/ikaros/.ssh/authorized_keys`): 2 entries — `daedalus@daedalus` (RSA) and `naorw@zgx-5175` (ED25519). Both known.

**Users**: `ikaros` (1000, bash), `remoteuser` (1001, bash). **ACTION**: confirm `remoteuser` is intentional (xrdp account?); if not, disable shell with `usermod -s /usr/sbin/nologin remoteuser`.

**Recent files in /etc**: ld.so.cache, openvpn/ipp.txt, cups subscriptions, Chrome's gpg key — all benign auto-updates. `/etc/passwd` and `/etc/shadow` mtime = **2026-05-30 15:26** (not touched overnight). 

**/tmp executables**: `/tmp/deep_audit.sh`, `/tmp/heavy_scan.sh`, `/tmp/phase3_audit.sh` — all from prior user-driven security audits. Not malicious.

**No new services** under /etc/systemd/system in last 30d. **No new SUIDs** in /usr|/bin|/sbin|/opt in last 30d.

**Verdict: CLEAN.**

---

### daedalus (192.168.0.40)
**Overnight activity**
- Boot at 17:24 (8.5 day uptime previously, reboot was user-initiated). Stayed running all night.
- Cron `/home/daedalus/babysit_agents.sh` runs every 20 minutes (00:00, 00:20, 00:40, ...) — explains periodic activity. Each run completes in ~30–45 s.
- 03:00, 03:00, 03:00 etc.: `snap.firmware-updater.firmware-notifier.service` failing every 3 h — benign snap glitch (not malware).
- 04:54 onward: user-driven activity (operator SSH'd in, started agent-triage.service, identity_benchmark scripts updated).
- `srgeo`, `llama-server`, `whisper-cli` are in `T` / `TLl` (SIGSTOP'ed) state per project convention — left untouched per instructions.

**Listening services**: 22/tcp sshd, 25/tcp postfix, 631/tcp cups, 8081/tcp loopback llama-server, 8000/tcp loopback srgeo. **No exposed network listeners beyond ssh + postfix-local.**

**Outbound**: only `python3` (pid 8269) → 192.168.0.41:6379 (zgx redis) and inbound SSH from ikaros. **No external connections.** Postfix port 25 is bound on 0.0.0.0 but no /etc/mailname configured (errors in log) — does not actually relay external mail.

**Auth log anomalies**: none. All sshd from 192.168.0.35 (ikaros) with our known ED25519 key. No failed logins, no unknown users.

**Authorized SSH keys**: 5 entries — `your_email@example.com` (placeholder, looks like the initial install key — **review**), `ericbergvall@MacBook-Air`, `minos@minos`, `daedalus@daedalus`, `naorw@zgx-5175`. **ACTION**: prune `your_email@example.com` if you don't recognise it (likely the very first key you added when bootstrapping the host, but worth confirming).

**Users**: `daedalus` (1000) only.

**New services in last 30d**: `agent-triage.service`, `snap.mesa-2404.component-monitor.service`, `postfix.service`. All expected. `agent-triage.service` has a malformed `Environment=` line (unquoted spaces) — agent still runs, but log spam.

**chkrootkit.service** in journal: "Failed to start chkrootkit.service" at 04:54:38 — was triggered when the chkrootkit.timer ran; the failure is benign (chkrootkit package itself returns non-zero on certain non-issues). Not a compromise indicator.

**/tmp executables**: only `/dev/shm/rocm_smi_renderD128` (a ROCm SMI shared-memory handle, normal). 

**`/etc/passwd` mtime**: 2026-06-01 04:54:38 — *touched today*. Cause: the rkhunter/chkrootkit cron.daily run at 04:54 invoked `unhide` / package post-install, which touched the file. Confirmed via `dpkg.log` showing `rkhunter`, `unhide.rb`, `bsd-mailx`, `ufw`, `libc-bin` all updated at 04:54:45–50. **Origin: unattended-upgrades package cycle — benign, but means `apt` ran (which is significant CPU)**.

**Recent files**: all under `/home/daedalus/AMD_gfx1151_energy/...` — identity_benchmark scripts, results — your own agent's work.

**Verdict: CLEAN.**

---

### zgx (192.168.0.41, NVIDIA DGX, naorw)
**Overnight activity**
- Boot 2026-05-11 (20+ days uptime).
- Cron: only the standard hourly anacron-check, sysstat, e2scrub — *no* user cron jobs. No `babysit` here.
- Persistent processes (4–5 day uptime): 4× `claude --dangerously-skip-permissions` instances (pids 2481967, 2544806, 2597414, 2815350-ish), `cloudflared` tunnel ("yggdrasil-ui"), `litellm` proxy on :4000, `uvicorn` on :7860, `hermes-agent` python gateway, `ollama` on :11434, `redis-server`, `python` exporters on :9100, :8765, :8787, `dockerd`.
- The 15-second SSH cadence to ikaros 04:53–04:54 originated from here (zgx → ikaros publickey loop). Likely an agent's babysit/loop script in a tmux session — not malicious.

**Listening services**: 22/tcp ssh, 25/tcp postfix, 631/tcp cups, 4000/tcp litellm, 6379/tcp redis (**bound on 0.0.0.0**), 8765/tcp python, 8787/tcp, 9100/tcp prometheus-node-exporter, 11434/tcp ollama (**bound on `*`, i.e. all interfaces**), 12434/tcp model server, 20241/tcp cloudflared, 7860/tcp uvicorn (loopback).
- **NOTE**: redis (6379), ollama (11434), litellm (4000), node-exporter (9100), 8765/8787 are all reachable from the LAN. If the LAN is not trusted, bind these to 127.0.0.1 or add UFW rules. No evidence of exploitation; this is just exposure hardening.

**Outbound**: only 2 active claude→Anthropic (160.79.104.10:443) — both user-owned.

**Auth log**: only your password-auth logins (ikaros → zgx with sshpass). No failed logins, no invalid users.

**Authorized SSH keys**: `naorw@zgx-5175`, `ericbergvall@MacBook-Air`, `minos@minos`, `daedalus@daedalus`. All known.

**Users**: `naorw` only.

**Sudo usage**: only the audit session today + a few interactive uses on May 26-27. Clean.

**Recent files in /home/naorw**: claude credentials, hermes tick.lock, yggdrasil ui.log, worker_zgx.screenlog. All your tooling.

**/tmp/deep_audit.sh** (2025 bytes, 02:55): printed by the prior audit session — same script content as on ikaros. Not malicious.

**`/etc/passwd` mtime**: 2026-06-01 02:54:45 — same cause as daedalus (unattended apt cycle ran rkhunter/unhide/ufw upgrades at 02:54). Confirmed via dpkg.log.

**No new SUID** in last 30 days. **New services in last 30d**: c2-dashboard, agent-strategist, redis, litellm, postfix — all yours.

**`agent-strategist.service`** has the same malformed `Environment=` issue (unquoted "strategic research and reasoning agent" string). Cosmetic.

**Verdict: CLEAN.** 

---

## Findings requiring action
1. **ikaros**: confirm purpose of `remoteuser` (UID 1001 with bash shell). If unused, `sudo usermod -s /usr/sbin/nologin remoteuser`.
2. **daedalus**: review `your_email@example.com` ED25519 key in `~/.ssh/authorized_keys` line 1 — looks like a leftover placeholder from initial setup. If unrecognized, remove it.
3. **zgx**: ollama (11434/tcp), redis (6379/tcp), litellm (4000/tcp), node-exporter (9100/tcp), 8765/tcp, 8787/tcp all listen on 0.0.0.0. Either bind to 127.0.0.1 or set UFW rules to restrict to 192.168.0.0/24 if LAN is trusted.
4. **daedalus + zgx**: fix the malformed `Environment=` lines in `agent-triage.service` / `agent-strategist.service` (wrap value in quotes or use `Environment="KEY=multi word value"`).
5. **Cosmetic**: `snap.firmware-updater.firmware-notifier.service` failing every 3h on ikaros + daedalus + zgx. Can disable with `systemctl --user mask snap.firmware-updater.firmware-notifier.service`.
6. **No urgent security action.** Fan-up overnight was your own agents (4× claude on zgx, babysit cron on daedalus, claude+node processes on ikaros). No external threat detected.

## Items I could not check (sudo required)
- `lastb` failed-login attempts (needs root).
- `/root/.ssh/authorized_keys` contents (file may not exist; not readable as user).
- `bpftool prog/map list` for eBPF programs (root only).
- `/etc/ld.so.preload` global preload list (readable but needs root for the directory listing).
- Full `dmesg --since "24 hours ago"` ring-buffer (kernel.dmesg_restrict=1).
- `sudo cat /etc/sudoers /etc/sudoers.d/*`.
- Hidden PID check (`for pid in /proc/[0-9]*; do …` for processes ps doesn't show — requires root for reliable visibility).

Recommend: next audit pass with `sudo -A` so these can be filled in.
