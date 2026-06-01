#!/usr/bin/env bash
# Example: collect signatures on TWO machines, then classify cross-chassis.
#
# Run on machine A:
#   ./scripts/01_collect_signature.sh --host alice --reps 10
#   scp data/alice_sig_v2.npz user@machineB:~/FabricCrypt/data/
#
# Then on machine B:
#   ./scripts/01_collect_signature.sh --host bob --reps 10
#   ./scripts/02_classify.sh data/alice_sig_v2.npz data/bob_sig_v2.npz
#
# Expected output: LOO accuracy > 0.95 (gate passes).
#
# This file is documentation; do not run directly. Edit the hostnames
# and SCP target then run the commands by hand.
echo "See comments at top of this file."
