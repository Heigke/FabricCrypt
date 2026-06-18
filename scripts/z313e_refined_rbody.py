"""z313e — bisection variant E: polarity + avalanche + REFINED R_body table.

R_body[0.2]=1e8 (was 1e10 — too restrictive),
R_body[0.4]=1e9 (unchanged),
R_body[0.6]=1e10 (was 1e8 — too leaky).
Reversed gradient relative to z313 original.
Drain-end avalanche M(V_bc) ON (Vbr=3.0V, N=4).
Output: results/z313_bisection/e_summary.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z313_bisection_common import run_variant, RBODY_REFINED

if __name__ == "__main__":
    s = run_variant(label="e", rbody_table=RBODY_REFINED, enable_avalanche=True)
    sys.exit(0)
