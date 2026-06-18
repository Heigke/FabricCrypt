"""Phase 14D — governor control + verification (LOCAL ikaros only).

Sets/verifies the CPU governor on this host. sudo password from env ($SUDO_PASS)
or fallback 'ikaros'. Returns previous governor so the caller can restore.
"""
import os, subprocess, time

GOV_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"


def list_governors():
    import glob
    govs = []
    for p in sorted(glob.glob(GOV_GLOB)):
        try:
            with open(p) as f:
                govs.append((p, f.read().strip()))
        except Exception as e:
            govs.append((p, f"ERR:{e}"))
    return govs


def current_governor():
    govs = list_governors()
    if not govs:
        return None
    # Return the first one — they should all agree
    return govs[0][1]


def all_agree(target):
    return all(g == target for _p, g in list_governors())


def set_governor(target, sudo_pass=None):
    sudo_pass = sudo_pass or os.environ.get("SUDO_PASS", "ikaros")
    cmd = ["sudo", "-S", "cpupower", "frequency-set", "-g", target]
    p = subprocess.run(cmd, input=(sudo_pass + "\n").encode(),
                       capture_output=True, timeout=30)
    out = p.stdout.decode() + p.stderr.decode()
    if p.returncode != 0:
        # fallback: write directly via tee
        import glob
        for path in glob.glob(GOV_GLOB):
            p2 = subprocess.run(
                ["sudo", "-S", "tee", path],
                input=(sudo_pass + "\n" + target + "\n").encode(),
                capture_output=True, timeout=10,
            )
            if p2.returncode != 0:
                out += "\nTEE_FAIL " + path + ": " + p2.stderr.decode()
    time.sleep(2)
    return all_agree(target), out


if __name__ == "__main__":
    import sys, json
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "show":
        print(json.dumps({"governors": list_governors(),
                          "agree_performance": all_agree("performance"),
                          "agree_powersave": all_agree("powersave")}, indent=2))
    elif cmd == "set":
        target = sys.argv[2]
        ok, out = set_governor(target)
        print(json.dumps({"target": target, "ok": ok,
                          "now": list_governors(), "raw": out}, indent=2))
    else:
        print("usage: governor_ctl.py [show | set <governor>]")
