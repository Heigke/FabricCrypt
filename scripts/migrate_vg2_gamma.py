"""migrate_vg2_gamma — convert old-convention fitted gamma_VG2 → new convention.

Per GPT review 2026-04-29: the cell wrapper (nsram/bsim4_port/nsram_cell.py) uses
    vth0_eff = vth0 + gamma_VG2 * VG2
while the old fit scripts used a minus sign. Fitted gammas from old runs must be
NEGATED to be portable to the wrapper convention. Run this script to print the
converted values for any fitted summary.json under results/.
"""
import json
from pathlib import Path

results = [
    'z76_bsim4_port_fit_p7v2',
    'z77_bsim4_port_fit_p7v3',
    'z78_bsim4_port_fit_p7v4',
    'z79_bsim4_port_fit_p7v5',
    'z79_bsim4_port_fit_p7v5_stagewise',
    'z80_bsim4_port_fit_p7v5b_merged',
]

for r in results:
    f = Path('results') / r / 'summary.json'
    if not f.exists():
        print(f'{r}: missing')
        continue
    d = json.load(open(f))
    fp = d.get('fitted_params', d.get('params', {}))
    if 'gamma_VG2' not in fp:
        print(f'{r}: no gamma_VG2 in summary')
        continue
    g_old = fp['gamma_VG2']
    print(f'{r}: gamma_VG2 (old fit-script convention) = {g_old:+.4f}')
    print(f'        gamma_VG2 (wrapper convention)    = {-g_old:+.4f}')
