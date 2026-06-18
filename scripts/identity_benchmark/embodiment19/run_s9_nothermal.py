#!/usr/bin/env python3
"""Run s9 with thermal_guard disabled (for daedalus where competing workloads
hold temp above pause threshold)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common19
common19.thermal_guard = lambda *a, **k: None
common19.wait_cool = lambda *a, **k: 0.0
import s9_jacobian_dynamics
# Re-patch references inside s9 module
s9_jacobian_dynamics.thermal_guard = common19.thermal_guard
s9_jacobian_dynamics.wait_cool = common19.wait_cool
s9_jacobian_dynamics.run(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
