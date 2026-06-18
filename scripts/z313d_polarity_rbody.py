"""z313d — bisection variant D: polarity fix + original R_body table.

R_body[0.2]=1e10, R_body[0.4]=1e9, R_body[0.6]=1e8.
No drain-end avalanche.
Output: results/z313_bisection/d_summary.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z313_bisection_common import run_variant, RBODY_Z313

if __name__ == "__main__":
    s = run_variant(label="d", rbody_table=RBODY_Z313, enable_avalanche=False)
    sys.exit(0)
