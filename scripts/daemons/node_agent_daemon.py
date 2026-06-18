#!/usr/bin/env python3
"""
Node Agent Daemon - Reports body state to Hypothalamus

Runs on each node (ikaros, daedalus, minos) and:
1. Collects local sensor data (power, temp, utilization)
2. Reports to central Hypothalamus server
3. Receives actuator commands (power cap, profile)

Usage:
    python node_agent_daemon.py --node-id ikaros --hypothalamus 192.168.0.1:8765

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
import json
import time
import socket
import argparse
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

# Add project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LocalState:
    """Local node state to report."""
    node_id: str
    hostname: str
    vendor: str

    # Hardware metrics
    power_watts: float = 0.0
    temp_c: float = 0.0
    utilization: float = 0.0
    clock_mhz: float = 0.0

    # Inference metrics
    j_per_token: float = 0.0
    tbt_ms: float = 0.0
    throughput_tps: float = 0.0

    # Body latent
    strain: float = 0.5
    urgency: float = 0.0
    debt: float = 0.0
    margin: float = 1.0
    stability: float = 1.0

    # Counters
    requests_served: int = 0
    tokens_generated: int = 0
    slo_violations: int = 0

    # Actuator state
    power_cap_watts: float = 0.0
    current_profile: str = "auto"

    timestamp_ns: int = 0


class SensorCollector:
    """Collects local sensor data."""

    def __init__(self, vendor: str = "auto"):
        self.vendor = self._detect_vendor() if vendor == "auto" else vendor
        self._sensor = None
        self._init_sensor()

    def _detect_vendor(self) -> str:
        """Detect GPU vendor."""
        # Check NVIDIA first
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return "NVIDIA"
        except:
            pass

        # Check AMD
        amd_paths = ["/sys/class/drm/card0/device", "/sys/class/drm/card1/device"]
        for path in amd_paths:
            if os.path.exists(f"{path}/vendor"):
                try:
                    with open(f"{path}/vendor") as f:
                        if "0x1002" in f.read():
                            return "AMD"
                except:
                    pass

        return "unknown"

    def _init_sensor(self) -> None:
        """Initialize appropriate sensor."""
        if self.vendor == "NVIDIA":
            try:
                from src.atom.nvml_sensor import NVMLSensor
                self._sensor = NVMLSensor()
                logger.info(f"NVML sensor initialized: {self._sensor.device_name}")
            except Exception as e:
                logger.warning(f"Failed to init NVML sensor: {e}")

        elif self.vendor == "AMD":
            try:
                from src.atom.amd_sensor import AMDSensor
                self._sensor = AMDSensor()
                logger.info(f"AMD sensor initialized: {self._sensor.device_name}")
            except Exception as e:
                logger.warning(f"Failed to init AMD sensor: {e}")

    def read(self) -> Dict[str, float]:
        """Read current sensor values."""
        if self._sensor is None:
            return {
                'power_watts': 0.0,
                'temp_c': 0.0,
                'utilization': 0.0,
                'clock_mhz': 0.0,
            }

        try:
            reading = self._sensor.read()
            return {
                'power_watts': reading.power_watts,
                'temp_c': reading.temp_c,
                'utilization': reading.utilization,
                'clock_mhz': getattr(reading, 'clock_mhz', 0.0),
            }
        except Exception as e:
            logger.warning(f"Sensor read error: {e}")
            return {
                'power_watts': 0.0,
                'temp_c': 0.0,
                'utilization': 0.0,
                'clock_mhz': 0.0,
            }


class NodeAgentDaemon:
    """
    Node agent daemon that reports to Hypothalamus.
    """

    def __init__(
        self,
        node_id: str,
        hypothalamus_url: str,
        report_interval_ms: int = 500,
        listen_port: int = 8766,
    ):
        self.node_id = node_id
        self.hypothalamus_url = hypothalamus_url.rstrip('/')
        self.report_interval_ms = report_interval_ms
        self.listen_port = listen_port

        # Initialize sensor
        self.sensor = SensorCollector()

        # State
        self.state = LocalState(
            node_id=node_id,
            hostname=socket.gethostname(),
            vendor=self.sensor.vendor,
        )

        # For receiving body latent updates from TokenMetabolismStep
        self._body_latent = {
            'strain': 0.5,
            'urgency': 0.0,
            'debt': 0.0,
            'margin': 1.0,
            'stability': 1.0,
        }
        self._inference_metrics = {
            'j_per_token': 0.0,
            'tbt_ms': 0.0,
            'throughput_tps': 0.0,
        }

        # Threads
        self._running = False
        self._reporter_thread = None
        self._server_thread = None
        self._http_server = None

        # Lock
        self._state_lock = threading.Lock()

        logger.info(f"NodeAgent initialized: {node_id} ({self.sensor.vendor})")

    def start(self) -> None:
        """Start daemon."""
        self._running = True

        # Start HTTP server for local updates
        self._start_http_server()

        # Start reporter thread
        self._reporter_thread = threading.Thread(target=self._report_loop, daemon=True)
        self._reporter_thread.start()

        logger.info(f"NodeAgent started, reporting to {self.hypothalamus_url}")

    def stop(self) -> None:
        """Stop daemon."""
        self._running = False

        if self._http_server:
            self._http_server.shutdown()

        if self._reporter_thread:
            self._reporter_thread.join(timeout=2.0)

    def _start_http_server(self) -> None:
        """Start HTTP server for receiving local updates."""
        daemon = self

        class UpdateHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging

            def do_POST(self):
                if self.path == '/update':
                    content_length = int(self.headers['Content-Length'])
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode('utf-8'))

                    with daemon._state_lock:
                        # Update body latent
                        if 'body_latent' in data:
                            daemon._body_latent.update(data['body_latent'])

                        # Update inference metrics
                        if 'inference' in data:
                            daemon._inference_metrics.update(data['inference'])

                        # Update counters
                        if 'requests_served' in data:
                            daemon.state.requests_served = data['requests_served']
                        if 'tokens_generated' in data:
                            daemon.state.tokens_generated = data['tokens_generated']
                        if 'slo_violations' in data:
                            daemon.state.slo_violations = data['slo_violations']

                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'OK')

                elif self.path == '/command':
                    # Receive actuator commands from Hypothalamus
                    content_length = int(self.headers['Content-Length'])
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode('utf-8'))

                    daemon._handle_command(data)

                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'OK')
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self):
                if self.path == '/status':
                    with daemon._state_lock:
                        response = asdict(daemon.state)

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()

        self._http_server = HTTPServer(('0.0.0.0', self.listen_port), UpdateHandler)
        self._server_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info(f"HTTP server listening on port {self.listen_port}")

    def _handle_command(self, command: Dict[str, Any]) -> None:
        """Handle actuator command from Hypothalamus."""
        cmd_type = command.get('type')

        if cmd_type == 'set_power_cap':
            watts = command.get('watts')
            logger.info(f"Setting power cap to {watts}W")
            # TODO: Integrate with actual actuator

        elif cmd_type == 'set_profile':
            profile = command.get('profile')
            logger.info(f"Setting profile to {profile}")
            # TODO: Integrate with actual actuator

        else:
            logger.warning(f"Unknown command: {cmd_type}")

    def _collect_state(self) -> LocalState:
        """Collect current state."""
        # Read sensors
        sensor_data = self.sensor.read()

        with self._state_lock:
            self.state.power_watts = sensor_data['power_watts']
            self.state.temp_c = sensor_data['temp_c']
            self.state.utilization = sensor_data['utilization']
            self.state.clock_mhz = sensor_data['clock_mhz']

            # Body latent
            self.state.strain = self._body_latent['strain']
            self.state.urgency = self._body_latent['urgency']
            self.state.debt = self._body_latent['debt']
            self.state.margin = self._body_latent['margin']
            self.state.stability = self._body_latent['stability']

            # Inference metrics
            self.state.j_per_token = self._inference_metrics['j_per_token']
            self.state.tbt_ms = self._inference_metrics['tbt_ms']
            self.state.throughput_tps = self._inference_metrics['throughput_tps']

            self.state.timestamp_ns = time.time_ns()

            return self.state

    def _report_loop(self) -> None:
        """Background loop to report state to Hypothalamus."""
        while self._running:
            try:
                state = self._collect_state()
                self._send_report(state)
            except Exception as e:
                logger.warning(f"Report error: {e}")

            time.sleep(self.report_interval_ms / 1000.0)

    def _send_report(self, state: LocalState) -> None:
        """Send state report to Hypothalamus."""
        url = f"{self.hypothalamus_url}/node/{self.node_id}/state"
        data = json.dumps(asdict(state)).encode('utf-8')

        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status != 200:
                    logger.warning(f"Hypothalamus returned {response.status}")
        except urllib.error.URLError as e:
            # Hypothalamus may not be running yet
            pass
        except Exception as e:
            logger.debug(f"Send report error: {e}")

    def update_from_metabolism(
        self,
        body_latent: Dict[str, float],
        inference_metrics: Dict[str, float],
        counters: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Update state from TokenMetabolismStep.

        Called by the inference code to report body state.
        """
        with self._state_lock:
            self._body_latent.update(body_latent)
            self._inference_metrics.update(inference_metrics)

            if counters:
                if 'requests_served' in counters:
                    self.state.requests_served = counters['requests_served']
                if 'tokens_generated' in counters:
                    self.state.tokens_generated = counters['tokens_generated']
                if 'slo_violations' in counters:
                    self.state.slo_violations = counters['slo_violations']


def main():
    parser = argparse.ArgumentParser(description='FEEL Node Agent Daemon')
    parser.add_argument('--node-id', required=True, help='Node identifier (e.g., ikaros)')
    parser.add_argument('--hypothalamus', default='http://192.168.0.1:8765',
                       help='Hypothalamus URL')
    parser.add_argument('--port', type=int, default=8766,
                       help='Local HTTP port for updates')
    parser.add_argument('--interval', type=int, default=500,
                       help='Report interval (ms)')

    args = parser.parse_args()

    daemon = NodeAgentDaemon(
        node_id=args.node_id,
        hypothalamus_url=args.hypothalamus,
        listen_port=args.port,
        report_interval_ms=args.interval,
    )

    daemon.start()

    try:
        logger.info(f"Node agent {args.node_id} running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        daemon.stop()


if __name__ == "__main__":
    main()
