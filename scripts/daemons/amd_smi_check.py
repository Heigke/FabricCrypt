#!/usr/bin/env python3
"""
AMD SMI Actuator Check - Verify power cap support on AMD GPUs

Checks if AMD SMI is available and what actuator capabilities exist.
Run on daedalus to verify power cap support.

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
import subprocess
import json
from typing import Dict, Any, Optional, List

def run_cmd(cmd: List[str], check: bool = False) -> tuple:
    """Run command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), -1

def check_rocm_smi() -> Dict[str, Any]:
    """Check rocm-smi availability and capabilities."""
    results = {
        "available": False,
        "version": None,
        "devices": [],
        "power_cap_support": False,
        "profile_support": False,
        "dpm_support": False,
    }

    # Check if rocm-smi exists
    stdout, stderr, rc = run_cmd(["which", "rocm-smi"])
    if rc != 0:
        results["error"] = "rocm-smi not found in PATH"
        return results

    results["available"] = True
    results["path"] = stdout.strip()

    # Get version
    stdout, stderr, rc = run_cmd(["rocm-smi", "--version"])
    if rc == 0:
        results["version"] = stdout.strip()

    # List devices
    stdout, stderr, rc = run_cmd(["rocm-smi", "--showid"])
    if rc == 0:
        results["device_list"] = stdout.strip()

    # Check power cap support
    stdout, stderr, rc = run_cmd(["rocm-smi", "--showmaxpower"])
    if rc == 0 and "N/A" not in stdout:
        results["power_cap_support"] = True
        results["max_power_output"] = stdout.strip()

    # Check current power
    stdout, stderr, rc = run_cmd(["rocm-smi", "--showpower"])
    if rc == 0:
        results["current_power"] = stdout.strip()

    # Check profile support
    stdout, stderr, rc = run_cmd(["rocm-smi", "--showprofile"])
    if rc == 0:
        results["profile_support"] = True
        results["profiles"] = stdout.strip()

    # Check DPM support
    stdout, stderr, rc = run_cmd(["rocm-smi", "--showdpm"])
    if rc == 0:
        results["dpm_support"] = True
        results["dpm_levels"] = stdout.strip()

    return results

def check_amd_smi() -> Dict[str, Any]:
    """Check amdsmi Python library availability."""
    results = {
        "available": False,
        "version": None,
    }

    try:
        import amdsmi
        results["available"] = True

        # Initialize
        amdsmi.amdsmi_init()

        # Get devices
        devices = amdsmi.amdsmi_get_processor_handles()
        results["device_count"] = len(devices)

        if devices:
            device = devices[0]

            # Get device info
            try:
                info = amdsmi.amdsmi_get_gpu_device_bdf(device)
                results["device_bdf"] = str(info)
            except:
                pass

            # Check power cap
            try:
                power_info = amdsmi.amdsmi_get_power_cap_info(device)
                results["power_cap_info"] = str(power_info)
                results["power_cap_support"] = True
            except Exception as e:
                results["power_cap_error"] = str(e)
                results["power_cap_support"] = False

            # Check current power
            try:
                power = amdsmi.amdsmi_get_power_info(device)
                results["current_power"] = str(power)
            except Exception as e:
                results["power_error"] = str(e)

        amdsmi.amdsmi_shut_down()

    except ImportError:
        results["error"] = "amdsmi library not installed"
    except Exception as e:
        results["error"] = str(e)

    return results

def check_sysfs_actuators() -> Dict[str, Any]:
    """Check sysfs-based actuator paths."""
    results = {
        "drm_devices": [],
        "hwmon_devices": [],
    }

    # Check DRM devices
    drm_path = "/sys/class/drm"
    if os.path.exists(drm_path):
        for entry in os.listdir(drm_path):
            if entry.startswith("card") and not entry.startswith("card0-"):
                card_path = os.path.join(drm_path, entry, "device")
                if os.path.exists(card_path):
                    device_info = {"name": entry, "path": card_path}

                    # Check power_dpm_force_performance_level
                    dpm_path = os.path.join(card_path, "power_dpm_force_performance_level")
                    if os.path.exists(dpm_path):
                        device_info["dpm_path"] = dpm_path
                        try:
                            with open(dpm_path, 'r') as f:
                                device_info["dpm_current"] = f.read().strip()
                        except:
                            pass

                    # Check pp_power_profile_mode
                    profile_path = os.path.join(card_path, "pp_power_profile_mode")
                    if os.path.exists(profile_path):
                        device_info["profile_path"] = profile_path
                        try:
                            with open(profile_path, 'r') as f:
                                device_info["profiles_available"] = f.read().strip()[:200]
                        except:
                            pass

                    results["drm_devices"].append(device_info)

    # Check hwmon devices
    hwmon_path = "/sys/class/hwmon"
    if os.path.exists(hwmon_path):
        for entry in os.listdir(hwmon_path):
            hw_path = os.path.join(hwmon_path, entry)
            name_path = os.path.join(hw_path, "name")
            if os.path.exists(name_path):
                try:
                    with open(name_path, 'r') as f:
                        name = f.read().strip()
                    if "amdgpu" in name:
                        device_info = {"name": name, "path": hw_path}

                        # Check power cap
                        power1_cap = os.path.join(hw_path, "power1_cap")
                        if os.path.exists(power1_cap):
                            device_info["power_cap_path"] = power1_cap
                            try:
                                with open(power1_cap, 'r') as f:
                                    device_info["power_cap_uw"] = int(f.read().strip())
                                    device_info["power_cap_w"] = device_info["power_cap_uw"] / 1000000
                            except:
                                pass

                        # Check power cap max
                        power1_cap_max = os.path.join(hw_path, "power1_cap_max")
                        if os.path.exists(power1_cap_max):
                            try:
                                with open(power1_cap_max, 'r') as f:
                                    device_info["power_cap_max_uw"] = int(f.read().strip())
                                    device_info["power_cap_max_w"] = device_info["power_cap_max_uw"] / 1000000
                            except:
                                pass

                        results["hwmon_devices"].append(device_info)
                except:
                    pass

    return results

def main():
    """Run all checks and report."""
    print("=" * 70)
    print("AMD SMI ACTUATOR CAPABILITY CHECK")
    print("=" * 70)

    # rocm-smi check
    print("\n1. ROCM-SMI CHECK")
    print("-" * 40)
    rocm_results = check_rocm_smi()
    print(f"  Available: {rocm_results['available']}")
    if rocm_results['available']:
        print(f"  Path: {rocm_results.get('path', 'N/A')}")
        print(f"  Power Cap Support: {rocm_results['power_cap_support']}")
        print(f"  Profile Support: {rocm_results['profile_support']}")
        print(f"  DPM Support: {rocm_results['dpm_support']}")
    else:
        print(f"  Error: {rocm_results.get('error', 'Unknown')}")

    # amdsmi library check
    print("\n2. AMDSMI LIBRARY CHECK")
    print("-" * 40)
    amd_results = check_amd_smi()
    print(f"  Available: {amd_results['available']}")
    if amd_results['available']:
        print(f"  Device Count: {amd_results.get('device_count', 'N/A')}")
        print(f"  Power Cap Support: {amd_results.get('power_cap_support', False)}")
    else:
        print(f"  Error: {amd_results.get('error', 'Unknown')}")

    # sysfs check
    print("\n3. SYSFS ACTUATOR PATHS")
    print("-" * 40)
    sysfs_results = check_sysfs_actuators()

    print(f"  DRM Devices: {len(sysfs_results['drm_devices'])}")
    for dev in sysfs_results['drm_devices']:
        print(f"    {dev['name']}: {dev['path']}")
        if 'dpm_path' in dev:
            print(f"      DPM: {dev.get('dpm_current', 'N/A')}")
        if 'profile_path' in dev:
            print(f"      Profiles: Available")

    print(f"\n  HWMON Devices: {len(sysfs_results['hwmon_devices'])}")
    for dev in sysfs_results['hwmon_devices']:
        print(f"    {dev['name']}: {dev['path']}")
        if 'power_cap_w' in dev:
            print(f"      Power Cap: {dev['power_cap_w']:.1f}W")
        if 'power_cap_max_w' in dev:
            print(f"      Power Cap Max: {dev['power_cap_max_w']:.1f}W")

    # Summary
    print("\n" + "=" * 70)
    print("ACTUATOR CAPABILITY SUMMARY")
    print("=" * 70)

    capabilities = []

    # Best actuator path
    if sysfs_results['hwmon_devices'] and any('power_cap_path' in d for d in sysfs_results['hwmon_devices']):
        capabilities.append("HWMON Power Cap (sysfs)")
    if sysfs_results['drm_devices'] and any('dpm_path' in d for d in sysfs_results['drm_devices']):
        capabilities.append("DPM Level (sysfs)")
    if sysfs_results['drm_devices'] and any('profile_path' in d for d in sysfs_results['drm_devices']):
        capabilities.append("Power Profile (sysfs)")
    if rocm_results.get('power_cap_support'):
        capabilities.append("Power Cap (rocm-smi)")
    if amd_results.get('power_cap_support'):
        capabilities.append("Power Cap (amdsmi)")

    if capabilities:
        print("\nAvailable Actuators:")
        for cap in capabilities:
            print(f"  ✓ {cap}")

        print("\nRecommended Actuator Priority:")
        print("  1. HWMON sysfs power cap (most direct)")
        print("  2. DPM sysfs level (reliable)")
        print("  3. rocm-smi (if HWMON not available)")
    else:
        print("\n⚠ No actuator capabilities detected!")
        print("  This may be a permissions issue or unsupported GPU.")

    # Save full results
    all_results = {
        "rocm_smi": rocm_results,
        "amdsmi": amd_results,
        "sysfs": sysfs_results,
        "capabilities": capabilities,
    }

    output_file = "amd_smi_check_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
