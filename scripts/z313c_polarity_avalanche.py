"""z313c — bisection variant C: polarity fix + drain-end avalanche.

No R_body table (uses default 1e9 for all V_G1).
Drain-end avalanche M(V_bc) with Vbr=3.0V, N=4.
Output: results/z313_bisection/c_summary.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z313_bisection_common import run_variant

if __name__ == "__main__":
    s = run_variant(label="c", rbody_table=None, enable_avalanche=True)
    sys.exit(0)
