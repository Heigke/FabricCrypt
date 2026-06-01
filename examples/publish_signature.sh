#!/usr/bin/env bash
# Example: publish your signature to the community dataset (N >= 6 chip goal).
#
# The .npz contains only numerical measurements + your chosen host label.
# It does NOT contain personal data, usernames, paths, or filesystem content.
# Inspect before sharing:
#   python -c "import numpy as np; d=np.load('data/$(hostname)_sig_v2.npz'); \
#              print(list(d.keys())); print('host:', d['host']); \
#              print('shape:', d['vec'].shape)"
#
# Then open an issue at https://github.com/Heigke/FabricCrypt/issues
# titled "dataset contribution: <chassis model>" and attach the .npz.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="$(cat /etc/hostname 2>/dev/null || hostname)"
NPZ="data/${HOST}_sig_v2.npz"
[ -f "$NPZ" ] || { echo "missing $NPZ — run 01_collect_signature.sh first"; exit 2; }

echo "Will share: $NPZ"
python3 -c "import numpy as np; d=np.load('$NPZ'); \
            print('  keys :', list(d.keys())); \
            print('  host :', d['host']); \
            print('  shape:', d['vec'].shape); \
            print('  dim  :', int(d['dim']))"
echo
echo "OK to attach this file to a GitHub issue or send to the maintainer."
