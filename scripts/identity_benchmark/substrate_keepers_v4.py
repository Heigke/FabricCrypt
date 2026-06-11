"""SubstrateStateV4 — keeper subset from 2026-06-10 cross-host channel audit.

Wraps V3 and exposes the 5 robust embodiment-bearing channels.

KEEPER set (passes ≥3/4 criteria on BOTH ikaros and daedalus):
  idx_in_v3=2  C20_lat_x     closed-loop ✓ spoof ✓ dynamics ✓
  idx_in_v3=4  C11_drift     closed-loop ✓ spoof ✓ dynamics ✓
  idx_in_v3=5  C05_e0_rt     closed-loop ✓ spoof ✓ dynamics ✓ (4/4 perfect)
  idx_in_v3=6  C06_fast      closed-loop ✗ (load_d≈0.1) — identity-only passenger
  idx_in_v3=9  C20_lat_e     closed-loop ✓ spoof ✓ dynamics ~

SMC_INDICES are channels with verified compute-load feedback (the closed loop
Buhrmann/Di Paolo SMC requires). Use these for compute-action conditioning
losses; C06_fast (index 3 in keeper space) is an identity-carrier only.

See results/IDENTITY_H7_2026-06-09/CHANNEL_AUDIT_SYNTHESIS_2026-06-10.md
"""
from __future__ import annotations
import numpy as np
from substrate_realtime_v3 import SubstrateStateV3

KEEPER_V3_INDICES = (2, 4, 5, 6, 9)
KEEPER_NAMES = ("C20_lat_x", "C11_drift", "C05_e0_rt", "C06_fast", "C20_lat_e")
SMC_KEEPER_INDICES = (0, 1, 2, 4)  # indices into keeper space — closed-loop channels only
IDENTITY_ONLY_INDICES = (3,)        # C06_fast — identity carrier, no SMC


class SubstrateStateV4(SubstrateStateV3):
    """V3 ring buffer; latest_window() returns only the 5 keeper channels."""
    N_KEEPER = 5

    def latest_window(self, length: int = 256) -> np.ndarray:
        w = super().latest_window(length=length)
        return w[:, list(KEEPER_V3_INDICES)]

    @staticmethod
    def smc_channels(window: np.ndarray) -> np.ndarray:
        """Closed-loop channels only (drop C06_fast). Shape (T, 4)."""
        return window[:, list(SMC_KEEPER_INDICES)]
