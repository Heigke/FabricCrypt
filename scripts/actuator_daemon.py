#!/usr/bin/env python3
"""
Actuator Daemon - Privileged Hardware Control Service
======================================================

This daemon runs with elevated privileges (root/admin) and exposes a
narrow, safe API for hardware power/performance control.

Design Goals:
1. SAFETY: Only expose safe, bounded operations
2. ISOLATION: Run as separate process with minimal permissions
3. SIMPLICITY: Unix socket, JSON protocol
4. AUDITABILITY: Log all operations

Installation (systemd):
-----------------------
1. Copy this file to /usr/local/bin/feel-actuator-daemon.py
2. Create systemd service:

    [Unit]
    Description=FEEL Actuator Daemon
    After=network.target

    [Service]
    Type=simple
    ExecStart=/usr/bin/python3 /usr/local/bin/feel-actuator-daemon.py
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target

3. Enable: sudo systemctl enable feel-actuator.service
4. Start: sudo systemctl start feel-actuator.service

Client Usage:
-------------
    from actuator_client import ActuatorClient
    client = ActuatorClient()
    client.set_power_limit(150)  # Watts
    client.reset_to_default()

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
import json
import socket
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Socket path (must be accessible by non-root users)
SOCKET_PATH = "/var/run/feel-actuator.sock"
LOG_PATH = "/var/log/feel-actuator.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH) if os.path.exists(os.path.dirname(LOG_PATH)) else logging.StreamHandler(),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PowerLimits:
    """Power limit bounds for safety."""
    min_watts: int
    max_watts: int
    default_watts: int


class NVIDIAActuatorDaemon:
    """
    Privileged NVIDIA GPU actuator daemon.

    Provides safe, bounded power control via nvidia-smi.
    """

    # Safety bounds (absolute limits)
    ABSOLUTE_MIN_POWER_W = 50   # Never go below this
    ABSOLUTE_MAX_POWER_W = 500  # Never exceed this

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._verify_nvidia_smi()
        self._limits = self._get_power_limits()
        logger.info(f"NVIDIA Actuator initialized: device={device_id}, limits={self._limits}")

    def _verify_nvidia_smi(self) -> None:
        """Verify nvidia-smi is available."""
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"nvidia-smi not available: {e}")

    def _get_power_limits(self) -> PowerLimits:
        """Query actual GPU power limits."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=power.min_limit,power.max_limit,power.default_limit",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = result.stdout.strip().split(",")
            min_w = int(float(parts[0].strip()))
            max_w = int(float(parts[1].strip()))
            default_w = int(float(parts[2].strip()))

            # Apply absolute safety bounds
            min_w = max(self.ABSOLUTE_MIN_POWER_W, min_w)
            max_w = min(self.ABSOLUTE_MAX_POWER_W, max_w)

            return PowerLimits(min_watts=min_w, max_watts=max_w, default_watts=default_w)
        except Exception as e:
            logger.warning(f"Could not query power limits: {e}")
            return PowerLimits(min_watts=100, max_watts=300, default_watts=200)

    def set_power_limit(self, watts: int) -> Dict[str, Any]:
        """
        Set GPU power limit.

        Args:
            watts: Target power limit in watts (bounded to safe range)

        Returns:
            Result dict with success status and actual value set
        """
        # Bound to safe range
        bounded_watts = max(self._limits.min_watts, min(self._limits.max_watts, watts))

        logger.info(f"Setting power limit: requested={watts}W, bounded={bounded_watts}W")

        try:
            subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    f"--power-limit={bounded_watts}",
                ],
                capture_output=True,
                check=True,
            )
            return {
                "success": True,
                "requested": watts,
                "actual": bounded_watts,
                "limits": asdict(self._limits),
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set power limit: {e}")
            return {
                "success": False,
                "error": str(e),
                "requested": watts,
            }

    def reset_to_default(self) -> Dict[str, Any]:
        """Reset power limit to default."""
        return self.set_power_limit(self._limits.default_watts)

    def get_status(self) -> Dict[str, Any]:
        """Get current GPU status."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=power.draw,power.limit,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "success": True,
                "power_draw_w": float(parts[0]),
                "power_limit_w": float(parts[1]),
                "temp_c": float(parts[2]),
                "util_pct": float(parts[3]),
                "limits": asdict(self._limits),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def lock_clocks(self, gpu_mhz: Optional[int] = None, mem_mhz: Optional[int] = None) -> Dict[str, Any]:
        """Lock GPU clocks to specific values."""
        try:
            if gpu_mhz is not None:
                subprocess.run(
                    ["nvidia-smi", f"--id={self.device_id}", f"--lock-gpu-clocks={gpu_mhz}"],
                    capture_output=True,
                    check=True,
                )
            if mem_mhz is not None:
                subprocess.run(
                    ["nvidia-smi", f"--id={self.device_id}", f"--lock-memory-clocks={mem_mhz}"],
                    capture_output=True,
                    check=True,
                )
            return {"success": True, "gpu_mhz": gpu_mhz, "mem_mhz": mem_mhz}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": str(e)}

    def unlock_clocks(self) -> Dict[str, Any]:
        """Unlock GPU clocks."""
        try:
            subprocess.run(
                ["nvidia-smi", f"--id={self.device_id}", "--reset-gpu-clocks"],
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["nvidia-smi", f"--id={self.device_id}", "--reset-memory-clocks"],
                capture_output=True,
                check=True,
            )
            return {"success": True}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": str(e)}


class AMDActuatorDaemon:
    """
    Privileged AMD GPU actuator daemon.

    Provides safe DPM level control via sysfs.
    """

    VALID_DPM_LEVELS = ['auto', 'low', 'high', 'manual', 'profile_standard',
                        'profile_min_sclk', 'profile_min_mclk', 'profile_peak']

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._sysfs_path = self._find_device_path()
        logger.info(f"AMD Actuator initialized: {self._sysfs_path}")

    def _find_device_path(self) -> Path:
        """Find sysfs path for device."""
        for i in range(10):
            path = Path(f"/sys/class/drm/card{i}/device")
            if path.exists():
                dpm_file = path / "power_dpm_force_performance_level"
                if dpm_file.exists():
                    return path
        raise RuntimeError("No AMD GPU found with DPM control")

    def set_dpm_level(self, level: str) -> Dict[str, Any]:
        """Set DPM performance level."""
        if level not in self.VALID_DPM_LEVELS:
            return {"success": False, "error": f"Invalid level: {level}"}

        dpm_file = self._sysfs_path / "power_dpm_force_performance_level"
        logger.info(f"Setting DPM level: {level}")

        try:
            with open(dpm_file, 'w') as f:
                f.write(level)
            return {"success": True, "level": level}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reset_to_default(self) -> Dict[str, Any]:
        """Reset to auto DPM."""
        return self.set_dpm_level("auto")

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        try:
            dpm_file = self._sysfs_path / "power_dpm_force_performance_level"
            with open(dpm_file, 'r') as f:
                current_level = f.read().strip()
            return {"success": True, "level": current_level}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ActuatorDaemon:
    """
    Main daemon server that handles client requests.
    """

    ALLOWED_COMMANDS = {
        'set_power_limit',
        'reset_to_default',
        'get_status',
        'lock_clocks',
        'unlock_clocks',
        'set_dpm_level',
        'ping',
    }

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self._detect_and_init_actuator()

    def _detect_and_init_actuator(self):
        """Detect GPU type and initialize appropriate actuator."""
        # Try NVIDIA first
        try:
            self.actuator = NVIDIAActuatorDaemon()
            self.vendor = "nvidia"
            return
        except Exception:
            pass

        # Try AMD
        try:
            self.actuator = AMDActuatorDaemon()
            self.vendor = "amd"
            return
        except Exception:
            pass

        raise RuntimeError("No supported GPU found")

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a client request."""
        cmd = request.get('command')
        args = request.get('args', {})

        if cmd not in self.ALLOWED_COMMANDS:
            return {"success": False, "error": f"Unknown command: {cmd}"}

        if cmd == 'ping':
            return {"success": True, "vendor": self.vendor}

        # Dispatch to actuator
        method = getattr(self.actuator, cmd, None)
        if method is None:
            return {"success": False, "error": f"Command not supported: {cmd}"}

        return method(**args)

    def run(self):
        """Run the daemon server."""
        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o666)  # Allow non-root access
        server.listen(5)

        logger.info(f"Actuator daemon listening on {self.socket_path}")

        try:
            while True:
                conn, _ = server.accept()
                try:
                    data = conn.recv(4096).decode('utf-8')
                    request = json.loads(data)
                    logger.info(f"Request: {request}")

                    response = self.handle_request(request)
                    logger.info(f"Response: {response}")

                    conn.sendall(json.dumps(response).encode('utf-8'))
                except Exception as e:
                    logger.error(f"Request error: {e}")
                    conn.sendall(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
                finally:
                    conn.close()
        finally:
            server.close()
            os.unlink(self.socket_path)


class ActuatorClient:
    """
    Client for communicating with the actuator daemon.

    This is what non-privileged code should use.
    """

    def __init__(self, socket_path: str = SOCKET_PATH, timeout: float = 5.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def _send_request(self, command: str, **args) -> Dict[str, Any]:
        """Send request to daemon."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect(self.socket_path)
            request = {"command": command, "args": args}
            sock.sendall(json.dumps(request).encode('utf-8'))
            response = sock.recv(4096).decode('utf-8')
            return json.loads(response)
        finally:
            sock.close()

    def ping(self) -> bool:
        """Check if daemon is running."""
        try:
            result = self._send_request('ping')
            return result.get('success', False)
        except Exception:
            return False

    def set_power_limit(self, watts: int) -> Dict[str, Any]:
        """Set power limit."""
        return self._send_request('set_power_limit', watts=watts)

    def reset_to_default(self) -> Dict[str, Any]:
        """Reset to default."""
        return self._send_request('reset_to_default')

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return self._send_request('get_status')

    def set_dpm_level(self, level: str) -> Dict[str, Any]:
        """Set AMD DPM level."""
        return self._send_request('set_dpm_level', level=level)

    def lock_clocks(self, gpu_mhz: Optional[int] = None, mem_mhz: Optional[int] = None) -> Dict[str, Any]:
        """Lock clocks."""
        return self._send_request('lock_clocks', gpu_mhz=gpu_mhz, mem_mhz=mem_mhz)

    def unlock_clocks(self) -> Dict[str, Any]:
        """Unlock clocks."""
        return self._send_request('unlock_clocks')


def main():
    parser = argparse.ArgumentParser(description='FEEL Actuator Daemon')
    parser.add_argument('--socket', default=SOCKET_PATH, help='Socket path')
    parser.add_argument('--client', action='store_true', help='Run as client (test)')
    args = parser.parse_args()

    if args.client:
        # Client mode for testing
        client = ActuatorClient(args.socket)
        if client.ping():
            print("Daemon is running")
            print(f"Status: {client.get_status()}")
        else:
            print("Daemon is not running")
        return

    # Check if running as root
    if os.geteuid() != 0:
        logger.warning("Not running as root - may have limited functionality")

    # Run daemon
    daemon = ActuatorDaemon(args.socket)
    daemon.run()


if __name__ == "__main__":
    main()
