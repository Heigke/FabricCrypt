#!/usr/bin/env python3
"""
Z87: FEEL Live Dashboard

Real-time dashboard showing:
- J/token and $/1M tokens
- TBT p95 and SLO violation rate
- Temperature + thermal margin
- BodyLatent + mode
- Actuator state

Run with: python scripts/z87_dashboard.py

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import threading
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import curses

# Try to import hardware modules
try:
    from src.atom import create_sensor, AtomConfig, BodyStateTracker
    from src.atom.interoception import (
        EmbodiedExpressionController, MachineCalibration, ExpressionMode
    )
    ATOM_AVAILABLE = True
except ImportError:
    ATOM_AVAILABLE = False
    print("Warning: atom modules not available, using mock data")


@dataclass
class DashboardState:
    """Current dashboard state."""
    # Hardware
    power_watts: float = 0.0
    power_cap_watts: float = 0.0
    temp_c: float = 0.0
    temp_limit_c: float = 85.0
    utilization: float = 0.0

    # Energy
    j_per_token: float = 0.0
    tokens_per_second: float = 0.0
    energy_cost_per_1m: float = 0.0  # $/1M tokens at $0.10/kWh

    # Latency
    tbt_p50_ms: float = 0.0
    tbt_p95_ms: float = 0.0
    slo_target_ms: float = 50.0
    slo_violation_rate: float = 0.0

    # BodyLatent
    strain: float = 0.5
    urgency: float = 0.0
    debt: float = 0.0
    margin: float = 1.0
    stability: float = 1.0

    # Expression
    mode: str = "BALANCED"
    temperature: float = 1.0
    top_k: int = 50

    # Actuator
    actuator_state: str = "auto"
    profile: str = "balanced"

    # Metadata
    timestamp: str = ""
    uptime_s: float = 0.0


class Dashboard:
    """
    Curses-based live dashboard for FEEL monitoring.
    """

    def __init__(self, update_interval_ms: int = 500):
        self.update_interval_ms = update_interval_ms
        self.state = DashboardState()
        self._running = False
        self._start_time = time.time()

        # Initialize sensor if available
        self.sensor = None
        self.body_tracker = None
        self.expression_controller = None

        if ATOM_AVAILABLE:
            try:
                self.sensor = create_sensor(device_id=0)
                self.body_tracker = BodyStateTracker(config=AtomConfig())
                calibration = MachineCalibration.for_amd_apu()
                self.expression_controller = EmbodiedExpressionController(
                    calibration=calibration,
                    enabled=True,
                    modulation_strength=0.5,
                )
            except Exception as e:
                print(f"Warning: Could not initialize hardware: {e}")

        # History for sparklines
        self._power_history: List[float] = []
        self._jpt_history: List[float] = []
        self._temp_history: List[float] = []

    def update_state(self) -> None:
        """Update dashboard state from hardware."""
        if self.sensor:
            try:
                snap = self.sensor.read()

                self.state.power_watts = snap.power_watts
                self.state.power_cap_watts = snap.power_cap_watts
                self.state.temp_c = snap.temp_c
                self.state.temp_limit_c = snap.temp_limit_c
                self.state.utilization = snap.util_gfx_percent

                # Update body tracker
                if self.body_tracker:
                    from src.atom.schema import InferencePhase
                    body_state = self.body_tracker.update(
                        snapshot=snap,
                        tokens_generated=1,
                        phase=InferencePhase.DECODE,
                        latency_ms=self.state.tbt_p50_ms or 10.0,
                    )
                    self.state.j_per_token = body_state.j_per_token

                # Update expression
                if self.expression_controller and self.body_tracker:
                    params = self.expression_controller.step(body_state)
                    latent = self.expression_controller.get_current_latent()

                    if latent:
                        self.state.strain = latent.strain
                        self.state.urgency = latent.urgency
                        self.state.debt = latent.debt
                        self.state.margin = latent.margin
                        self.state.stability = latent.stability
                        self.state.mode = latent.get_expression_mode().name

                    if params:
                        self.state.temperature = params.temperature
                        self.state.top_k = params.top_k

                # Calculate energy cost (assuming $0.10/kWh)
                if self.state.j_per_token > 0:
                    kwh_per_1m = (self.state.j_per_token * 1_000_000) / 3_600_000
                    self.state.energy_cost_per_1m = kwh_per_1m * 0.10

            except Exception as e:
                pass  # Silently handle errors

        self.state.timestamp = datetime.now().strftime('%H:%M:%S')
        self.state.uptime_s = time.time() - self._start_time

        # Update history
        self._power_history.append(self.state.power_watts)
        self._jpt_history.append(self.state.j_per_token)
        self._temp_history.append(self.state.temp_c)

        # Keep last 60 samples
        for hist in [self._power_history, self._jpt_history, self._temp_history]:
            if len(hist) > 60:
                hist.pop(0)

    def sparkline(self, values: List[float], width: int = 20) -> str:
        """Generate a sparkline string."""
        if not values:
            return " " * width

        chars = "▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1

        # Sample values to fit width
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values + [values[-1]] * (width - len(values))

        result = ""
        for v in sampled:
            idx = int((v - min_val) / range_val * (len(chars) - 1))
            result += chars[idx]

        return result

    def draw(self, stdscr) -> None:
        """Draw the dashboard."""
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)

        GREEN = curses.color_pair(1)
        YELLOW = curses.color_pair(2)
        RED = curses.color_pair(3)
        CYAN = curses.color_pair(4)
        WHITE = curses.color_pair(5)

        row = 0

        # Header
        header = "═══════════════════════════════════════════════════════════════════════════════"
        stdscr.addstr(row, 0, header[:width-1], CYAN | curses.A_BOLD)
        row += 1

        title = "  FEEL - Felt Energy-aware Expression for LLMs  "
        stdscr.addstr(row, (width - len(title)) // 2, title, CYAN | curses.A_BOLD)
        row += 1

        stdscr.addstr(row, 0, header[:width-1], CYAN | curses.A_BOLD)
        row += 2

        # Section: Hardware
        stdscr.addstr(row, 0, "╔═ HARDWARE ═══════════════════════════════════════════╗", WHITE | curses.A_BOLD)
        row += 1

        # Power
        power_color = GREEN if self.state.power_watts < 50 else (YELLOW if self.state.power_watts < 80 else RED)
        stdscr.addstr(row, 0, f"║ Power:   {self.state.power_watts:6.1f}W / {self.state.power_cap_watts:.0f}W  ", WHITE)
        stdscr.addstr(self.sparkline(self._power_history, 20), power_color)
        stdscr.addstr(" ║", WHITE)
        row += 1

        # Temperature
        temp_color = GREEN if self.state.temp_c < 60 else (YELLOW if self.state.temp_c < 75 else RED)
        stdscr.addstr(row, 0, f"║ Temp:    {self.state.temp_c:6.1f}°C / {self.state.temp_limit_c:.0f}°C  ", WHITE)
        stdscr.addstr(self.sparkline(self._temp_history, 20), temp_color)
        stdscr.addstr(" ║", WHITE)
        row += 1

        # Utilization
        util_color = GREEN if self.state.utilization < 70 else (YELLOW if self.state.utilization < 90 else RED)
        stdscr.addstr(row, 0, f"║ GPU Util: {self.state.utilization:5.1f}%                                     ║", WHITE)
        row += 1

        stdscr.addstr(row, 0, "╚════════════════════════════════════════════════════════╝", WHITE | curses.A_BOLD)
        row += 2

        # Section: Energy
        stdscr.addstr(row, 0, "╔═ ENERGY ════════════════════════════════════════════════╗", WHITE | curses.A_BOLD)
        row += 1

        jpt_color = GREEN if self.state.j_per_token < 1.0 else (YELLOW if self.state.j_per_token < 2.0 else RED)
        stdscr.addstr(row, 0, f"║ J/token:  {self.state.j_per_token:6.3f}  ", WHITE)
        stdscr.addstr(self.sparkline(self._jpt_history, 20), jpt_color)
        stdscr.addstr("       ║", WHITE)
        row += 1

        cost_str = f"${self.state.energy_cost_per_1m:.4f}" if self.state.energy_cost_per_1m > 0 else "N/A"
        stdscr.addstr(row, 0, f"║ Cost/1M tokens: {cost_str:<40} ║", WHITE)
        row += 1

        stdscr.addstr(row, 0, "╚════════════════════════════════════════════════════════╝", WHITE | curses.A_BOLD)
        row += 2

        # Section: Body Latent
        stdscr.addstr(row, 0, "╔═ BODY LATENT ═══════════════════════════════════════════╗", WHITE | curses.A_BOLD)
        row += 1

        # Strain bar
        strain_bar = "█" * int(self.state.strain * 20) + "░" * (20 - int(self.state.strain * 20))
        strain_color = GREEN if self.state.strain < 0.5 else (YELLOW if self.state.strain < 0.7 else RED)
        stdscr.addstr(row, 0, f"║ Strain:  [{strain_bar}] {self.state.strain:4.2f}        ║", strain_color)
        row += 1

        # Urgency bar
        urgency_bar = "█" * int(self.state.urgency * 20) + "░" * (20 - int(self.state.urgency * 20))
        urgency_color = GREEN if self.state.urgency < 0.5 else (YELLOW if self.state.urgency < 0.7 else RED)
        stdscr.addstr(row, 0, f"║ Urgency: [{urgency_bar}] {self.state.urgency:4.2f}        ║", urgency_color)
        row += 1

        # Debt bar (can be negative)
        debt_display = (self.state.debt + 1) / 2  # Map [-1,1] to [0,1]
        debt_bar = "█" * int(debt_display * 20) + "░" * (20 - int(debt_display * 20))
        debt_color = GREEN if self.state.debt < 0 else (YELLOW if self.state.debt < 0.5 else RED)
        debt_sign = "+" if self.state.debt > 0 else ""
        stdscr.addstr(row, 0, f"║ Debt:    [{debt_bar}] {debt_sign}{self.state.debt:4.2f}       ║", debt_color)
        row += 1

        # Margin bar
        margin_bar = "█" * int(self.state.margin * 20) + "░" * (20 - int(self.state.margin * 20))
        margin_color = RED if self.state.margin < 0.3 else (YELLOW if self.state.margin < 0.5 else GREEN)
        stdscr.addstr(row, 0, f"║ Margin:  [{margin_bar}] {self.state.margin:4.2f}        ║", margin_color)
        row += 1

        stdscr.addstr(row, 0, "╚════════════════════════════════════════════════════════╝", WHITE | curses.A_BOLD)
        row += 2

        # Section: Expression
        stdscr.addstr(row, 0, "╔═ EXPRESSION ════════════════════════════════════════════╗", WHITE | curses.A_BOLD)
        row += 1

        mode_colors = {
            'BALANCED': GREEN,
            'CONSERVE': YELLOW,
            'RECOVER': YELLOW,
            'URGENT': RED,
            'EXPLORE': CYAN,
        }
        mode_color = mode_colors.get(self.state.mode, WHITE)
        stdscr.addstr(row, 0, f"║ Mode:        ", WHITE)
        stdscr.addstr(f"{self.state.mode:<12}", mode_color | curses.A_BOLD)
        stdscr.addstr(f"                            ║", WHITE)
        row += 1

        stdscr.addstr(row, 0, f"║ Temperature: {self.state.temperature:5.2f}   Top-K: {self.state.top_k:<4}                 ║", WHITE)
        row += 1

        stdscr.addstr(row, 0, "╚════════════════════════════════════════════════════════╝", WHITE | curses.A_BOLD)
        row += 2

        # Footer
        stdscr.addstr(row, 0, f"Time: {self.state.timestamp}  Uptime: {self.state.uptime_s:.0f}s  [q] quit", CYAN)

        stdscr.refresh()

    def run(self, stdscr) -> None:
        """Main dashboard loop."""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input

        self._running = True
        while self._running:
            self.update_state()
            self.draw(stdscr)

            # Check for quit
            try:
                key = stdscr.getch()
                if key == ord('q'):
                    self._running = False
            except:
                pass

            time.sleep(self.update_interval_ms / 1000.0)


def main():
    dashboard = Dashboard(update_interval_ms=500)

    try:
        curses.wrapper(dashboard.run)
    except KeyboardInterrupt:
        pass

    print("Dashboard closed.")


if __name__ == "__main__":
    main()
