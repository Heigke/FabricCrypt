#!/usr/bin/env python3
"""
NVIDIA Actuator Daemon - Privileged NVML Control Service

Runs as root/admin to provide safe NVML state-changing operations.
Exposes a Unix socket API for non-privileged clients.

Installation:
    sudo cp nvidia_actuator_daemon.py /usr/local/bin/
    sudo cp nvidia-actuator.service /etc/systemd/system/
    sudo systemctl enable nvidia-actuator
    sudo systemctl start nvidia-actuator

API Commands (JSON over Unix socket):
    {"cmd": "get_state"}
    {"cmd": "set_power_limit", "watts": 150}
    {"cmd": "set_gpu_clocks", "min_mhz": 300, "max_mhz": 1500}
    {"cmd": "reset_clocks"}
    {"cmd": "set_persistence_mode", "enabled": true}

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
import json
import socket
import logging
import signal
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

# NVML imports
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("WARNING: pynvml not available. Install with: pip install pynvml")

SOCKET_PATH = "/var/run/nvidia-actuator.sock"
LOG_FILE = "/var/log/nvidia-actuator.log"
PID_FILE = "/var/run/nvidia-actuator.pid"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE) if os.path.exists(os.path.dirname(LOG_FILE)) else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class NVMLState:
    """Current NVML device state."""
    device_id: int = 0
    name: str = ""
    power_watts: float = 0.0
    power_limit_watts: float = 0.0
    power_limit_min_watts: float = 0.0
    power_limit_max_watts: float = 0.0
    temp_c: float = 0.0
    temp_threshold_c: float = 0.0
    gpu_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    gpu_clock_max_mhz: int = 0
    mem_clock_max_mhz: int = 0
    utilization_gpu: int = 0
    utilization_mem: int = 0
    persistence_mode: bool = False
    error: Optional[str] = None


class NVIDIAActuatorDaemon:
    """
    Privileged NVIDIA actuator daemon.

    Requires root/admin for NVML state-changing operations.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.handle = None
        self.running = False
        self._lock = threading.Lock()

        if not NVML_AVAILABLE:
            raise RuntimeError("pynvml not available")

        # Initialize NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        logger.info(f"NVIDIA Actuator initialized: {name}")

    def get_state(self) -> NVMLState:
        """Get current device state."""
        try:
            with self._lock:
                name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
                power_limits = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)

                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_threshold = pynvml.nvmlDeviceGetTemperatureThreshold(
                    self.handle, pynvml.NVML_TEMPERATURE_THRESHOLD_GPU_MAX
                )

                clocks = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clocks = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                max_clocks = pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
                max_mem_clocks = pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)

                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

                try:
                    persist = pynvml.nvmlDeviceGetPersistenceMode(self.handle)
                except:
                    persist = 0

                return NVMLState(
                    device_id=self.device_id,
                    name=name,
                    power_watts=power,
                    power_limit_watts=power_limit,
                    power_limit_min_watts=power_limits[0] / 1000.0,
                    power_limit_max_watts=power_limits[1] / 1000.0,
                    temp_c=temp,
                    temp_threshold_c=temp_threshold,
                    gpu_clock_mhz=clocks,
                    mem_clock_mhz=mem_clocks,
                    gpu_clock_max_mhz=max_clocks,
                    mem_clock_max_mhz=max_mem_clocks,
                    utilization_gpu=util.gpu,
                    utilization_mem=util.memory,
                    persistence_mode=bool(persist),
                )
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return NVMLState(error=str(e))

    def set_power_limit(self, watts: float) -> Dict[str, Any]:
        """Set power limit in watts."""
        try:
            with self._lock:
                milliwatts = int(watts * 1000)
                pynvml.nvmlDeviceSetPowerManagementLimit(self.handle, milliwatts)
                logger.info(f"Power limit set to {watts}W")
                return {"success": True, "power_limit_watts": watts}
        except pynvml.NVMLError as e:
            error = f"NVML error setting power limit: {e}"
            logger.error(error)
            return {"success": False, "error": error}

    def set_gpu_clocks(self, min_mhz: int, max_mhz: int) -> Dict[str, Any]:
        """Set GPU clock range (locked clocks)."""
        try:
            with self._lock:
                pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, min_mhz, max_mhz)
                logger.info(f"GPU clocks locked to {min_mhz}-{max_mhz} MHz")
                return {"success": True, "min_mhz": min_mhz, "max_mhz": max_mhz}
        except pynvml.NVMLError as e:
            error = f"NVML error setting clocks: {e}"
            logger.error(error)
            return {"success": False, "error": error}

    def reset_clocks(self) -> Dict[str, Any]:
        """Reset GPU clocks to default."""
        try:
            with self._lock:
                pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
                logger.info("GPU clocks reset to default")
                return {"success": True}
        except pynvml.NVMLError as e:
            error = f"NVML error resetting clocks: {e}"
            logger.error(error)
            return {"success": False, "error": error}

    def set_persistence_mode(self, enabled: bool) -> Dict[str, Any]:
        """Set persistence mode."""
        try:
            with self._lock:
                mode = pynvml.NVML_FEATURE_ENABLED if enabled else pynvml.NVML_FEATURE_DISABLED
                pynvml.nvmlDeviceSetPersistenceMode(self.handle, mode)
                logger.info(f"Persistence mode set to {enabled}")
                return {"success": True, "persistence_mode": enabled}
        except pynvml.NVMLError as e:
            error = f"NVML error setting persistence mode: {e}"
            logger.error(error)
            return {"success": False, "error": error}

    def handle_command(self, cmd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a command from client."""
        cmd = cmd_data.get("cmd", "")

        if cmd == "get_state":
            state = self.get_state()
            return asdict(state)

        elif cmd == "set_power_limit":
            watts = cmd_data.get("watts")
            if watts is None:
                return {"error": "Missing 'watts' parameter"}
            return self.set_power_limit(float(watts))

        elif cmd == "set_gpu_clocks":
            min_mhz = cmd_data.get("min_mhz")
            max_mhz = cmd_data.get("max_mhz")
            if min_mhz is None or max_mhz is None:
                return {"error": "Missing 'min_mhz' or 'max_mhz' parameter"}
            return self.set_gpu_clocks(int(min_mhz), int(max_mhz))

        elif cmd == "reset_clocks":
            return self.reset_clocks()

        elif cmd == "set_persistence_mode":
            enabled = cmd_data.get("enabled", True)
            return self.set_persistence_mode(bool(enabled))

        else:
            return {"error": f"Unknown command: {cmd}"}

    def run_server(self):
        """Run the Unix socket server."""
        # Remove existing socket
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o666)  # Allow non-root clients
        server.listen(5)

        self.running = True
        logger.info(f"NVIDIA Actuator daemon listening on {SOCKET_PATH}")

        while self.running:
            try:
                server.settimeout(1.0)
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue

                try:
                    data = conn.recv(4096).decode('utf-8')
                    cmd_data = json.loads(data)
                    result = self.handle_command(cmd_data)
                    conn.send(json.dumps(result).encode('utf-8'))
                except Exception as e:
                    conn.send(json.dumps({"error": str(e)}).encode('utf-8'))
                finally:
                    conn.close()

            except Exception as e:
                logger.error(f"Server error: {e}")

        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

    def stop(self):
        """Stop the daemon."""
        self.running = False

    def cleanup(self):
        """Cleanup NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


class NVIDIAActuatorClient:
    """Client for communicating with NVIDIA actuator daemon."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path

    def _send_command(self, cmd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to daemon."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
            sock.send(json.dumps(cmd_data).encode('utf-8'))
            response = sock.recv(4096).decode('utf-8')
            return json.loads(response)
        finally:
            sock.close()

    def get_state(self) -> Dict[str, Any]:
        """Get current GPU state."""
        return self._send_command({"cmd": "get_state"})

    def set_power_limit(self, watts: float) -> Dict[str, Any]:
        """Set power limit."""
        return self._send_command({"cmd": "set_power_limit", "watts": watts})

    def set_gpu_clocks(self, min_mhz: int, max_mhz: int) -> Dict[str, Any]:
        """Set GPU clock range."""
        return self._send_command({"cmd": "set_gpu_clocks", "min_mhz": min_mhz, "max_mhz": max_mhz})

    def reset_clocks(self) -> Dict[str, Any]:
        """Reset clocks to default."""
        return self._send_command({"cmd": "reset_clocks"})


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NVIDIA Actuator Daemon")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("ERROR: This daemon requires root privileges for NVML state changes.")
        print("Run with: sudo python nvidia_actuator_daemon.py")
        sys.exit(1)

    daemon = NVIDIAActuatorDaemon(device_id=args.device)

    def signal_handler(signum, frame):
        logger.info("Shutting down...")
        daemon.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        daemon.run_server()
    finally:
        daemon.cleanup()


if __name__ == "__main__":
    main()
