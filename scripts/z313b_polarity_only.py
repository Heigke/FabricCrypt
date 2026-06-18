"""z313b — bisection variant B: polarity fix ONLY.

No R_body table (uses default 1e9 for all V_G1).
No drain-end avalanche.
Output: results/z313_bisection/b_summary.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z313_bisection_common import run_variant

if __name__ == "__main__":
    s = run_variant(label="b", rbody_table=None, enable_avalanche=False)
    sys.exit(0)
