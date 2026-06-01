"""FabricCrypt signature module.

5-signal HAL-bypass per-die fingerprint extractor.
Produces a 290-dimensional vector per capture from:
  - Block 1: TSC inter-core offsets (35 dims)
  - Block 2: Cacheline ping-pong RTT matrix (35 dims)
  - Block 3: DRAM-refresh-aligned latency histogram (200 dims)
  - Block 4: Syscall p99.9 tail percentiles (10 dims)
  - Block 5: NVMe queue-tail percentiles (10 dims)
"""
from .signature_v2 import extract_one, TOTAL_DIM, BLOCK_STARTS, DIMS

__all__ = ["extract_one", "TOTAL_DIM", "BLOCK_STARTS", "DIMS"]
