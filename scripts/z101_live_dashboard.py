#!/usr/bin/env python3
"""
z101_live_dashboard.py - Live FEEL Demo Dashboard

This is the demo HP/AMD executives will care about.
Shows in real-time:
  - mJ/token
  - $ / 1M tokens
  - TTFT/TPOT p95
  - profile/power cap
  - "body mode" (BALANCED/RECOVER/EXPLORE)
  - Toggle: Baseline vs FEEL

Usage:
    # Start daemon and vLLM first, then:
    python z101_live_dashboard.py --daemon-host localhost --daemon-port 9877 --vllm-port 8000

Requirements:
    - PrivilegedDaemon v2 running with NVML
    - vLLM server running
    - Terminal with ANSI color support
"""

import argparse
import json
import os
import sys
import time
import threading
import signal
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import statistics

import requests

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.organism.hypothalamus import (
    Hypothalamus, HypothalamusConfig, BodyMode,
    NodeTelemetry, VLLMMetrics
)


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_power(watts: float) -> str:
    """Format power with color based on level."""
    if watts < 150:
        return f"{Colors.GREEN}{watts:.0f}W{Colors.RESET}"
    elif watts < 250:
        return f"{Colors.YELLOW}{watts:.0f}W{Colors.RESET}"
    else:
        return f"{Colors.RED}{watts:.0f}W{Colors.RESET}"


def format_temp(temp_c: float) -> str:
    """Format temperature with color."""
    if temp_c < 70:
        return f"{Colors.GREEN}{temp_c:.0f}°C{Colors.RESET}"
    elif temp_c < 80:
        return f"{Colors.YELLOW}{temp_c:.0f}°C{Colors.RESET}"
    else:
        return f"{Colors.RED}{temp_c:.0f}°C{Colors.RESET}"


def format_latency(ms: float, slo_ms: float) -> str:
    """Format latency with SLO comparison."""
    ratio = ms / slo_ms
    if ratio < 0.8:
        return f"{Colors.GREEN}{ms:.0f}ms{Colors.RESET}"
    elif ratio < 1.0:
        return f"{Colors.YELLOW}{ms:.0f}ms{Colors.RESET}"
    else:
        return f"{Colors.RED}{ms:.0f}ms{Colors.RESET} (>{slo_ms:.0f})"


def format_mode(mode: str) -> str:
    """Format body mode with color."""
    mode_colors = {
        "balanced": f"{Colors.BLUE}● BALANCED{Colors.RESET}",
        "recover": f"{Colors.GREEN}● RECOVER{Colors.RESET}",
        "performance": f"{Colors.RED}● PERFORMANCE{Colors.RESET}",
        "explore": f"{Colors.MAGENTA}● EXPLORE{Colors.RESET}",
    }
    return mode_colors.get(mode, mode)


@dataclass
class DemoMetrics:
    """Aggregated metrics for dashboard."""
    timestamp: float
    # Energy (truth)
    mj_per_token: float
    cost_per_m_tokens: float
    avg_power_w: float
    # Performance
    tokens_per_second: float
    ttft_ms_p95: float
    tpot_ms_p95: float
    # State
    mode: str
    profile: str
    power_limit_w: float
    temp_c: float
    # Meta
    feel_enabled: bool
    slo_violations: int


class DemoDaemon:
    """Simple daemon client for dashboard."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._prev_energy = 0
        self._prev_time = 0.0
        self._prev_tokens = 0

    def get_telemetry(self) -> Dict:
        """Get telemetry with derived power."""
        try:
            telem = requests.get(f"{self.base_url}/telemetry", timeout=1).json()
            energy = requests.get(f"{self.base_url}/energy", timeout=1).json()

            now = time.time()
            current_energy = energy.get('energy_mj', 0)

            # CRITICAL: Derive avg_power from ΔE/Δt
            if self._prev_energy > 0 and self._prev_time > 0:
                energy_delta = current_energy - self._prev_energy
                time_delta = now - self._prev_time
                derived_power = (energy_delta / 1000) / max(time_delta, 0.001)
            else:
                derived_power = telem.get('power_watts', 0)

            self._prev_energy = current_energy
            self._prev_time = now

            return {
                'power_w': derived_power,  # DERIVED
                'instant_power_w': telem.get('power_watts', 0),  # Detail
                'temp_c': telem.get('temp_c', 0),
                'profile': telem.get('profile', 'unknown'),
                'power_limit_w': telem.get('power_limit_w', 0),
                'utilization': telem.get('utilization', 0),
            }
        except:
            return {}

    def set_profile(self, profile: str) -> bool:
        """Set energy profile."""
        try:
            resp = requests.post(
                f"{self.base_url}/profile",
                json={"profile": profile},
                timeout=2
            )
            return resp.status_code == 200
        except:
            return False


class DemoVLLM:
    """Simple vLLM client for dashboard."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._prev_tokens = 0
        self._prev_time = 0.0

    def get_metrics(self) -> Dict:
        """Get vLLM metrics."""
        try:
            # Get Prometheus metrics
            resp = requests.get(f"{self.base_url}/metrics", timeout=2)
            if resp.status_code != 200:
                return {}

            # Parse key metrics
            metrics = {}
            for line in resp.text.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue

                # Parse histogram quantiles
                if 'vllm:e2e_request_latency_seconds' in line and 'quantile="0.95"' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics['ttft_ms_p95'] = float(parts[1]) * 1000

                if 'vllm:time_per_output_token_seconds' in line and 'quantile="0.95"' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics['tpot_ms_p95'] = float(parts[1]) * 1000

                if 'vllm:generation_tokens_total' in line and '{' not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        current = float(parts[1])
                        now = time.time()
                        if self._prev_time > 0:
                            dt = now - self._prev_time
                            metrics['tokens_per_second'] = (current - self._prev_tokens) / max(dt, 0.001)
                        self._prev_tokens = current
                        self._prev_time = now

            return metrics
        except Exception as e:
            return {}

    def generate_load(self, prompt: str, max_tokens: int = 64) -> Optional[Dict]:
        """Generate tokens to create load."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None


class LiveDashboard:
    """Live dashboard for FEEL demo."""

    def __init__(self, daemon: DemoDaemon, vllm: DemoVLLM,
                 hypothalamus: Optional[Hypothalamus] = None):
        self.daemon = daemon
        self.vllm = vllm
        self.hypothalamus = hypothalamus

        # State
        self.feel_enabled = True
        self.baseline_profile = "performance"
        self._running = False
        self._metrics_history: List[DemoMetrics] = []

        # SLO config
        self.ttft_slo_ms = 500
        self.tpot_slo_ms = 50
        self._slo_violations = 0

        # Comparison data
        self._baseline_data: List[DemoMetrics] = []
        self._feel_data: List[DemoMetrics] = []

    def toggle_feel(self):
        """Toggle between FEEL and baseline."""
        self.feel_enabled = not self.feel_enabled
        if not self.feel_enabled:
            # Switch to baseline (fixed performance profile)
            self.daemon.set_profile(self.baseline_profile)
        # If FEEL enabled, hypothalamus will take control

    def collect_metrics(self) -> Optional[DemoMetrics]:
        """Collect current metrics."""
        telem = self.daemon.get_telemetry()
        vllm_metrics = self.vllm.get_metrics()

        if not telem:
            return None

        power_w = telem.get('power_w', 0)
        tps = vllm_metrics.get('tokens_per_second', 0)

        # mJ/token = (power_W * 1000) / tokens_per_sec
        mj_per_token = (power_w * 1000) / max(tps, 0.1)

        # Cost: kWh per million tokens * $/kWh
        kwh_per_m = (mj_per_token / 1000 / 3600) * 1e6
        cost_per_m = kwh_per_m * 0.12

        # Check SLO
        ttft = vllm_metrics.get('ttft_ms_p95', 0)
        tpot = vllm_metrics.get('tpot_ms_p95', 0)
        if ttft > self.ttft_slo_ms or tpot > self.tpot_slo_ms:
            self._slo_violations += 1

        # Get mode from hypothalamus or infer from profile
        mode = "balanced"
        if self.hypothalamus:
            status = self.hypothalamus.get_status()
            mode = status.get('mode', 'balanced')
        elif telem.get('profile') == 'eco':
            mode = "recover"
        elif telem.get('profile') == 'performance':
            mode = "performance"

        return DemoMetrics(
            timestamp=time.time(),
            mj_per_token=mj_per_token,
            cost_per_m_tokens=cost_per_m,
            avg_power_w=power_w,
            tokens_per_second=tps,
            ttft_ms_p95=ttft,
            tpot_ms_p95=tpot,
            mode=mode,
            profile=telem.get('profile', 'unknown'),
            power_limit_w=telem.get('power_limit_w', 0),
            temp_c=telem.get('temp_c', 0),
            feel_enabled=self.feel_enabled,
            slo_violations=self._slo_violations,
        )

    def render(self, metrics: DemoMetrics):
        """Render dashboard to terminal."""
        clear_screen()

        # Header
        mode_str = f"{Colors.BG_GREEN}" if self.feel_enabled else f"{Colors.BG_RED}"
        toggle_str = "FEEL ENABLED" if self.feel_enabled else "BASELINE (Fixed Performance)"

        print(f"""
{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════╗
║                    FEEL Energy-Aware LLM Inference                    ║
║                         Live Demo Dashboard                           ║
╠══════════════════════════════════════════════════════════════════════╣{Colors.RESET}
║  {mode_str} {toggle_str:^20} {Colors.RESET}  │  Mode: {format_mode(metrics.mode):^30}  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  {Colors.BOLD}PRIMARY METRICS{Colors.RESET}                                                   ║
║  ┌────────────────────┬────────────────────┬────────────────────┐   ║
║  │  mJ/token          │  $/1M tokens       │  tokens/sec        │   ║
║  │  {Colors.CYAN}{metrics.mj_per_token:>12.1f}{Colors.RESET}      │  {Colors.GREEN}${metrics.cost_per_m_tokens:>11.4f}{Colors.RESET}     │  {Colors.BLUE}{metrics.tokens_per_second:>12.1f}{Colors.RESET}    │   ║
║  └────────────────────┴────────────────────┴────────────────────┘   ║
║                                                                      ║
║  {Colors.BOLD}LATENCY (SLO: TTFT<{self.ttft_slo_ms}ms, TPOT<{self.tpot_slo_ms}ms){Colors.RESET}                             ║
║  ┌────────────────────┬────────────────────┬────────────────────┐   ║
║  │  TTFT p95          │  TPOT p95          │  SLO Violations    │   ║
║  │  {format_latency(metrics.ttft_ms_p95, self.ttft_slo_ms):>18} │  {format_latency(metrics.tpot_ms_p95, self.tpot_slo_ms):>18} │  {Colors.RED if metrics.slo_violations > 0 else Colors.GREEN}{metrics.slo_violations:>12}{Colors.RESET}      │   ║
║  └────────────────────┴────────────────────┴────────────────────┘   ║
║                                                                      ║
║  {Colors.BOLD}HARDWARE STATUS{Colors.RESET}                                                   ║
║  ┌────────────────────┬────────────────────┬────────────────────┐   ║
║  │  Avg Power         │  Temperature       │  Power Limit       │   ║
║  │  {format_power(metrics.avg_power_w):>18} │  {format_temp(metrics.temp_c):>18} │  {metrics.power_limit_w:>12.0f}W     │   ║
║  └────────────────────┴────────────────────┴────────────────────┘   ║
║  Profile: {Colors.BOLD}{metrics.profile:^12}{Colors.RESET}                                              ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  {Colors.BOLD}COMPARISON (Rolling 30s){Colors.RESET}                                          ║""")

        # Calculate comparison stats
        if self._baseline_data and self._feel_data:
            baseline_mj = statistics.mean([m.mj_per_token for m in self._baseline_data[-30:]])
            feel_mj = statistics.mean([m.mj_per_token for m in self._feel_data[-30:]])
            savings_pct = ((baseline_mj - feel_mj) / baseline_mj) * 100 if baseline_mj > 0 else 0

            baseline_tps = statistics.mean([m.tokens_per_second for m in self._baseline_data[-30:]])
            feel_tps = statistics.mean([m.tokens_per_second for m in self._feel_data[-30:]])
            tps_diff = ((feel_tps - baseline_tps) / baseline_tps) * 100 if baseline_tps > 0 else 0

            print(f"""║  ┌────────────────────┬────────────────────┬────────────────────┐   ║
║  │                    │  Baseline          │  FEEL              │   ║
║  │  mJ/token          │  {baseline_mj:>12.1f}      │  {feel_mj:>12.1f}      │   ║
║  │  tokens/sec        │  {baseline_tps:>12.1f}      │  {feel_tps:>12.1f}      │   ║
║  └────────────────────┴────────────────────┴────────────────────┘   ║
║                                                                      ║
║  {Colors.BOLD}Energy Savings: {Colors.GREEN}{savings_pct:+.1f}%{Colors.RESET}    Throughput: {Colors.BLUE}{tps_diff:+.1f}%{Colors.RESET}                    ║""")
        else:
            print(f"""║  Collecting data... (toggle with 't' to compare)                     ║""")

        print(f"""╠══════════════════════════════════════════════════════════════════════╣
║  {Colors.YELLOW}Controls: [t] Toggle FEEL  [q] Quit  [r] Reset stats{Colors.RESET}                ║
╚══════════════════════════════════════════════════════════════════════╝
""")

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  Last update: {ts}")

    def run(self, update_interval: float = 1.0):
        """Run the dashboard."""
        self._running = True

        # Start load generator in background
        load_thread = threading.Thread(target=self._generate_load, daemon=True)
        load_thread.start()

        # Input handling
        import select
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while self._running:
                # Check for input
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q':
                        break
                    elif key == 't':
                        self.toggle_feel()
                    elif key == 'r':
                        self._baseline_data = []
                        self._feel_data = []
                        self._slo_violations = 0

                # Collect and display metrics
                metrics = self.collect_metrics()
                if metrics:
                    self._metrics_history.append(metrics)

                    # Track for comparison
                    if metrics.feel_enabled:
                        self._feel_data.append(metrics)
                    else:
                        self._baseline_data.append(metrics)

                    self.render(metrics)

                time.sleep(update_interval)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self._running = False

    def _generate_load(self):
        """Background load generator."""
        prompts = [
            "Explain quantum computing in simple terms.",
            "Write a haiku about artificial intelligence.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does machine learning work?",
        ]

        while self._running:
            for prompt in prompts:
                if not self._running:
                    break
                self.vllm.generate_load(prompt, max_tokens=64)
                time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="FEEL Live Dashboard Demo")
    parser.add_argument("--daemon-host", default="localhost")
    parser.add_argument("--daemon-port", type=int, default=9877)
    parser.add_argument("--vllm-host", default="localhost")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--use-hypothalamus", action="store_true",
                       help="Use Hypothalamus for control (vs simple profile switching)")
    parser.add_argument("--update-interval", type=float, default=1.0)
    args = parser.parse_args()

    # Create clients
    daemon = DemoDaemon(args.daemon_host, args.daemon_port)
    vllm = DemoVLLM(args.vllm_host, args.vllm_port)

    # Optional: use Hypothalamus for control
    hypothalamus = None
    if args.use_hypothalamus:
        config = HypothalamusConfig()
        hypothalamus = Hypothalamus(config)
        hypothalamus.add_node(
            "local",
            args.daemon_host, args.daemon_port,
            args.vllm_host, args.vllm_port
        )
        hypothalamus.start()

    # Create and run dashboard
    dashboard = LiveDashboard(daemon, vllm, hypothalamus)

    try:
        dashboard.run(args.update_interval)
    except KeyboardInterrupt:
        pass
    finally:
        if hypothalamus:
            hypothalamus.stop()

    print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
