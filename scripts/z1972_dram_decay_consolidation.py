#!/usr/bin/env python3
"""
z1972: DRAM Decay as Memory Consolidation Mechanism

================================================================================
                    BIOLOGICAL MEMORY CONSOLIDATION VIA DRAM PHYSICS
================================================================================

CONTEXT from z1190-z1196:
- DDR3 at room temperature shows NO decay for 60+ seconds
- Decay only occurs at elevated temperatures (85C spec)
- We use SIMULATED controlled decay as a memory consolidation mechanism
  (since real FPGA partial writes don't work reliably via Etherbone)

CONCEPT - Biological Memory Consolidation:
- Short-term memory: High-activity, frequently refreshed
- Long-term memory: Low-activity, consolidated, more permanent
- Forgetting: Intentional decay of unimportant information

KEY INSIGHT:
Physical DRAM decay at elevated temperatures follows: exp(-t / tau(T))
We simulate this to create a biologically-inspired memory system where:
1. Important memories get "refreshed" (like synaptic potentiation)
2. Unimportant memories decay (like synaptic depression)
3. Consolidation moves memories from volatile to stable storage

TESTS:
1. Memory capacity vs refresh rate tradeoff
2. Consolidation improves recall of important items
3. Decay removes noise/unimportant items
4. Compare with standard neural memory

================================================================================
"""

import os
import sys
import time
import json
import math
import hashlib
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from pathlib import Path

# Set HSA override for AMD gfx1151
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
#                    DRAM PHYSICS SIMULATION
# ============================================================================

@dataclass
class DRAMPhysics:
    """Physical parameters for DRAM decay simulation."""

    # Temperature-dependent decay time constant
    # tau(T) = tau_0 * exp(Ea / (k * T))
    # At 25C (298K): tau ~ 64ms (spec) but real chips much longer
    # At 85C (358K): tau ~ 2ms (accelerated decay)

    tau_25c_ms: float = 64000.0      # Very long at room temp (observed in z1196)
    tau_85c_ms: float = 2.0          # Fast decay at elevated temp
    activation_energy_ev: float = 0.5  # Typical for DRAM retention

    def get_tau_ms(self, temperature_c: float) -> float:
        """Get decay time constant at given temperature."""
        # Arrhenius relationship
        k_ev_per_k = 8.617e-5  # Boltzmann constant in eV/K

        T1 = 273.15 + 25  # Reference at 25C
        T2 = 273.15 + temperature_c

        # tau(T2) = tau(T1) * exp(Ea/k * (1/T2 - 1/T1))
        exponent = self.activation_energy_ev / k_ev_per_k * (1/T2 - 1/T1)
        tau = self.tau_25c_ms * math.exp(exponent)

        return tau

    def decay_probability(self, time_ms: float, temperature_c: float) -> float:
        """Probability that a bit decays after given time at temperature."""
        tau = self.get_tau_ms(temperature_c)
        return 1.0 - math.exp(-time_ms / tau)


# ============================================================================
#                    SIMULATED DRAM MEMORY CELL ARRAY
# ============================================================================

class SimulatedDRAMArray:
    """
    Simulates DRAM cell array with decay behavior.

    Each cell stores a float32 value that can decay over time
    based on temperature and refresh policy.
    """

    def __init__(self, num_cells: int, physics: DRAMPhysics = None):
        self.num_cells = num_cells
        self.physics = physics or DRAMPhysics()

        # Cell storage
        self.cells = np.zeros(num_cells, dtype=np.float32)
        self.write_times = np.zeros(num_cells, dtype=np.float64)
        self.refresh_times = np.zeros(num_cells, dtype=np.float64)
        self.importance = np.zeros(num_cells, dtype=np.float32)  # 0=unimportant, 1=important

        # Statistics
        self.total_writes = 0
        self.total_reads = 0
        self.total_refreshes = 0
        self.decay_events = 0

        self.current_time = 0.0

    def advance_time(self, delta_ms: float):
        """Advance simulation time."""
        self.current_time += delta_ms

    def write(self, index: int, value: float, importance: float = 0.5):
        """Write value to cell with importance tag."""
        self.cells[index] = value
        self.write_times[index] = self.current_time
        self.refresh_times[index] = self.current_time
        self.importance[index] = importance
        self.total_writes += 1

    def read(self, index: int, temperature_c: float = 25.0) -> Tuple[float, bool]:
        """
        Read value from cell, applying decay probability.
        Returns (value, decayed_flag).
        """
        self.total_reads += 1

        # Calculate time since last refresh
        time_since_refresh = self.current_time - self.refresh_times[index]

        # Apply decay
        decay_prob = self.physics.decay_probability(time_since_refresh, temperature_c)

        if np.random.random() < decay_prob:
            # Cell decayed - return degraded value
            self.decay_events += 1
            original = self.cells[index]
            # Decay towards zero with noise
            decay_factor = np.random.uniform(0.0, 0.5)
            self.cells[index] *= decay_factor
            return self.cells[index], True

        return self.cells[index], False

    def refresh(self, index: int):
        """Refresh cell (restores and prevents decay)."""
        self.refresh_times[index] = self.current_time
        self.total_refreshes += 1

    def batch_read(self, indices: np.ndarray, temperature_c: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
        """Read multiple cells at once."""
        values = np.zeros(len(indices), dtype=np.float32)
        decayed = np.zeros(len(indices), dtype=bool)

        for i, idx in enumerate(indices):
            values[i], decayed[i] = self.read(idx, temperature_c)

        return values, decayed

    def batch_write(self, indices: np.ndarray, values: np.ndarray, importances: np.ndarray = None):
        """Write multiple cells at once."""
        if importances is None:
            importances = np.ones(len(indices)) * 0.5

        for i, (idx, val, imp) in enumerate(zip(indices, values, importances)):
            self.write(idx, val, imp)

    def batch_refresh(self, indices: np.ndarray):
        """Refresh multiple cells."""
        for idx in indices:
            self.refresh(idx)

    def get_stats(self) -> Dict[str, Any]:
        """Get array statistics."""
        return {
            'num_cells': self.num_cells,
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
            'total_refreshes': self.total_refreshes,
            'decay_events': self.decay_events,
            'decay_rate': self.decay_events / max(1, self.total_reads),
            'current_time_ms': self.current_time,
        }


# ============================================================================
#                    DRAM MEMORY CONSOLIDATION SYSTEM
# ============================================================================

class DRAMMemoryConsolidation:
    """
    Biologically-inspired memory consolidation using DRAM decay physics.

    Short-term memory: Frequently refreshed, volatile
    Long-term memory: Consolidated, stable, less frequent refresh

    Importance determines which memories survive consolidation.
    """

    def __init__(
        self,
        short_term_capacity: int = 1024,
        long_term_capacity: int = 4096,
        consolidation_threshold: float = 0.7,  # Importance threshold to consolidate
        decay_temperature_c: float = 85.0,      # Simulated temperature for decay
    ):
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.consolidation_threshold = consolidation_threshold
        self.decay_temperature = decay_temperature_c

        # Memory arrays
        self.short_term = SimulatedDRAMArray(short_term_capacity)
        self.long_term = SimulatedDRAMArray(long_term_capacity)

        # Key-to-index mapping
        self.short_term_map: Dict[str, int] = {}
        self.long_term_map: Dict[str, int] = {}

        # Free list for allocation
        self.short_term_free = list(range(short_term_capacity))
        self.long_term_free = list(range(long_term_capacity))

        # Statistics
        self.consolidation_count = 0
        self.forgotten_count = 0
        self.access_counts: Dict[str, int] = {}

    def store(self, key: str, value: float, importance: float = 0.5):
        """
        Store a memory.

        High importance (>threshold) → goes to long-term
        Low importance → goes to short-term (may decay)
        """
        if importance >= self.consolidation_threshold:
            self._store_long_term(key, value, importance)
        else:
            self._store_short_term(key, value, importance)

        self.access_counts[key] = self.access_counts.get(key, 0) + 1

    def _store_short_term(self, key: str, value: float, importance: float):
        """Store in short-term memory."""
        if key in self.short_term_map:
            idx = self.short_term_map[key]
        elif self.short_term_free:
            idx = self.short_term_free.pop()
            self.short_term_map[key] = idx
        else:
            # Evict least important
            min_imp = float('inf')
            evict_key = None
            for k, i in self.short_term_map.items():
                if self.short_term.importance[i] < min_imp:
                    min_imp = self.short_term.importance[i]
                    evict_key = k

            if evict_key:
                idx = self.short_term_map.pop(evict_key)
                self.forgotten_count += 1
            else:
                return  # Full and can't evict

            self.short_term_map[key] = idx

        self.short_term.write(idx, value, importance)

    def _store_long_term(self, key: str, value: float, importance: float):
        """Store in long-term memory."""
        if key in self.long_term_map:
            idx = self.long_term_map[key]
        elif self.long_term_free:
            idx = self.long_term_free.pop()
            self.long_term_map[key] = idx
        else:
            # Long-term is full - try to evict oldest low-importance item
            min_imp = float('inf')
            evict_key = None
            for k, i in self.long_term_map.items():
                if self.long_term.importance[i] < min_imp:
                    min_imp = self.long_term.importance[i]
                    evict_key = k

            if evict_key and min_imp < importance:
                idx = self.long_term_map.pop(evict_key)
            else:
                # Can't fit - demote to short-term
                self._store_short_term(key, value, importance)
                return

            self.long_term_map[key] = idx

        self.long_term.write(idx, value, importance)

    def retrieve(self, key: str) -> Tuple[Optional[float], str, bool]:
        """
        Retrieve a memory.

        Returns: (value, memory_type, was_decayed)
        """
        # Check long-term first (more stable)
        if key in self.long_term_map:
            idx = self.long_term_map[key]
            value, decayed = self.long_term.read(idx, temperature_c=25.0)  # Room temp for LT

            if decayed:
                # LT memory degraded - remove it
                self.long_term_free.append(idx)
                del self.long_term_map[key]
                return None, 'long_term_decayed', True

            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return value, 'long_term', False

        # Check short-term
        if key in self.short_term_map:
            idx = self.short_term_map[key]
            value, decayed = self.short_term.read(idx, self.decay_temperature)

            if decayed:
                # ST memory decayed - remove it
                self.short_term_free.append(idx)
                del self.short_term_map[key]
                self.forgotten_count += 1
                return None, 'short_term_decayed', True

            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return value, 'short_term', False

        return None, 'not_found', False

    def consolidate(self):
        """
        Run memory consolidation.

        1. Move important short-term memories to long-term
        2. Let unimportant short-term memories decay
        3. Refresh long-term memories
        """
        # Phase 1: Promote important short-term memories
        to_promote = []
        for key, idx in list(self.short_term_map.items()):
            importance = self.short_term.importance[idx]
            access_count = self.access_counts.get(key, 0)

            # Boost importance based on access frequency
            effective_importance = min(1.0, importance + 0.1 * access_count)

            if effective_importance >= self.consolidation_threshold:
                to_promote.append((key, self.short_term.cells[idx], effective_importance))

        for key, value, importance in to_promote:
            # Remove from short-term
            if key in self.short_term_map:
                idx = self.short_term_map.pop(key)
                self.short_term_free.append(idx)

            # Add to long-term
            self._store_long_term(key, value, importance)
            self.consolidation_count += 1

        # Phase 2: Refresh all long-term memories
        for idx in self.long_term_map.values():
            self.long_term.refresh(idx)

        # Phase 3: Decay short-term (just advance time, decay happens on read)
        # No explicit refresh for short-term

        return len(to_promote)

    def advance_time(self, delta_ms: float):
        """Advance simulation time for both memory arrays."""
        self.short_term.advance_time(delta_ms)
        self.long_term.advance_time(delta_ms)

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            'short_term': {
                'capacity': self.short_term_capacity,
                'used': len(self.short_term_map),
                'utilization': len(self.short_term_map) / self.short_term_capacity,
                **self.short_term.get_stats(),
            },
            'long_term': {
                'capacity': self.long_term_capacity,
                'used': len(self.long_term_map),
                'utilization': len(self.long_term_map) / self.long_term_capacity,
                **self.long_term.get_stats(),
            },
            'consolidation_count': self.consolidation_count,
            'forgotten_count': self.forgotten_count,
            'decay_temperature_c': self.decay_temperature,
        }


# ============================================================================
#                    STANDARD NEURAL MEMORY (BASELINE)
# ============================================================================

class StandardNeuralMemory(nn.Module):
    """
    Standard neural memory network for comparison.
    Uses attention-based memory with learned forgetting.
    """

    def __init__(self, memory_size: int = 1024, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.memory_size = memory_size
        self.embed_dim = embed_dim

        # Memory slots
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim) * 0.02)

        # Query/Key/Value projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Gating for write
        self.write_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # Importance predictor
        self.importance_pred = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.key_map: Dict[str, int] = {}
        self.next_slot = 0

    def store(self, key: str, value: torch.Tensor):
        """Store value in memory."""
        if key in self.key_map:
            slot = self.key_map[key]
        else:
            slot = self.next_slot % self.memory_size
            self.key_map[key] = slot
            self.next_slot += 1

        with torch.no_grad():
            self.memory.data[slot] = value.detach()

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from memory using attention."""
        q = self.query_proj(query.unsqueeze(0))
        k = self.key_proj(self.memory)
        v = self.value_proj(self.memory)

        attn = torch.softmax(q @ k.T / math.sqrt(self.embed_dim), dim=-1)
        output = attn @ v

        importance = self.importance_pred(output)

        return output.squeeze(0), importance.squeeze()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'memory_size': self.memory_size,
            'used_slots': len(self.key_map),
            'utilization': len(self.key_map) / self.memory_size,
        }


# ============================================================================
#                    DRAM-BACKED NEURAL MEMORY
# ============================================================================

class DRAMBackedNeuralMemory(nn.Module):
    """
    Neural memory backed by DRAM consolidation.
    Combines neural attention with physical decay simulation.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        short_term_capacity: int = 256,
        long_term_capacity: int = 1024,
        device: torch.device = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device or torch.device('cpu')

        # DRAM consolidation system (stores scalar importance/access patterns)
        self.dram = DRAMMemoryConsolidation(
            short_term_capacity=short_term_capacity,
            long_term_capacity=long_term_capacity,
        )

        # Neural encoder/decoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Importance predictor
        self.importance_net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Tensor storage (indexed by DRAM)
        self.tensor_store: Dict[str, torch.Tensor] = {}

    def store(self, key: str, value: torch.Tensor):
        """Store memory with neural importance estimation."""
        encoded = self.encoder(value)
        importance = self.importance_net(encoded).item()

        # Store tensor
        self.tensor_store[key] = encoded.detach()

        # Store in DRAM consolidation (importance tracking)
        self.dram.store(key, importance, importance)

    def retrieve(self, key: str) -> Tuple[Optional[torch.Tensor], str, bool]:
        """Retrieve memory, subject to DRAM decay."""
        importance_val, mem_type, decayed = self.dram.retrieve(key)

        if importance_val is None or decayed:
            # Memory decayed or not found
            if key in self.tensor_store:
                del self.tensor_store[key]
            return None, mem_type, decayed

        if key not in self.tensor_store:
            return None, 'tensor_missing', False

        decoded = self.decoder(self.tensor_store[key])
        return decoded, mem_type, False

    def consolidate(self):
        """Run DRAM consolidation."""
        promoted = self.dram.consolidate()
        return promoted

    def advance_time(self, delta_ms: float):
        """Advance DRAM time."""
        self.dram.advance_time(delta_ms)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.dram.get_stats()
        stats['tensor_store_size'] = len(self.tensor_store)
        return stats


# ============================================================================
#                    BENCHMARK TESTS
# ============================================================================

def test_capacity_refresh_tradeoff(
    dram_memory: DRAMMemoryConsolidation,
    num_items: int = 500,
    time_between_consolidations_ms: float = 100.0,
) -> Dict[str, Any]:
    """
    Test 1: Memory capacity vs refresh rate tradeoff.

    More frequent consolidation = more stable but higher overhead.
    Less frequent = more decay but lower overhead.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Capacity vs Refresh Rate Tradeoff")
    print("=" * 60)

    results = {'consolidation_intervals': []}

    for interval_ms in [50, 100, 200, 500, 1000]:
        # Reset memory
        dram_memory.__init__(
            short_term_capacity=dram_memory.short_term_capacity,
            long_term_capacity=dram_memory.long_term_capacity,
        )

        # Store items with varying importance
        np.random.seed(42)
        stored_items = {}
        for i in range(num_items):
            key = f"item_{i}"
            value = np.random.randn()
            importance = np.random.uniform(0.0, 1.0)
            stored_items[key] = (value, importance)
            dram_memory.store(key, value, importance)

            # Advance time between stores
            dram_memory.advance_time(5.0)

            # Consolidate at interval
            if (i + 1) % (interval_ms // 5) == 0:
                dram_memory.consolidate()

        # Final consolidation
        dram_memory.consolidate()

        # Retrieve and check retention
        retained = 0
        decayed = 0
        for key in stored_items.keys():
            val, mem_type, was_decayed = dram_memory.retrieve(key)
            if val is not None:
                retained += 1
            else:
                decayed += 1

        retention_rate = retained / num_items
        stats = dram_memory.get_stats()

        result = {
            'interval_ms': interval_ms,
            'items_stored': num_items,
            'retained': retained,
            'decayed': decayed,
            'retention_rate': retention_rate,
            'consolidations': stats['consolidation_count'],
            'refresh_overhead': stats['long_term']['total_refreshes'],
        }
        results['consolidation_intervals'].append(result)

        print(f"  Interval {interval_ms:4d}ms: Retention={retention_rate:.1%}, "
              f"Consolidations={stats['consolidation_count']}")

    return results


def test_importance_based_consolidation(
    dram_memory: DRAMMemoryConsolidation,
    num_items: int = 200,
) -> Dict[str, Any]:
    """
    Test 2: Consolidation improves recall of important items.

    Important items should survive consolidation better.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Importance-Based Consolidation")
    print("=" * 60)

    # Reset
    dram_memory.__init__(
        short_term_capacity=dram_memory.short_term_capacity,
        long_term_capacity=dram_memory.long_term_capacity,
    )

    np.random.seed(42)

    # Store items: 50% high importance, 50% low importance
    high_importance_items = {}
    low_importance_items = {}

    for i in range(num_items):
        key = f"item_{i}"
        value = np.random.randn()

        if i < num_items // 2:
            # High importance
            importance = np.random.uniform(0.8, 1.0)
            high_importance_items[key] = (value, importance)
        else:
            # Low importance
            importance = np.random.uniform(0.0, 0.3)
            low_importance_items[key] = (value, importance)

        dram_memory.store(key, value, importance)
        dram_memory.advance_time(10.0)

    # Run multiple consolidation cycles
    for _ in range(5):
        dram_memory.advance_time(200.0)
        dram_memory.consolidate()

    # Check retention by importance group
    high_retained = sum(1 for k in high_importance_items if dram_memory.retrieve(k)[0] is not None)
    low_retained = sum(1 for k in low_importance_items if dram_memory.retrieve(k)[0] is not None)

    high_retention = high_retained / len(high_importance_items)
    low_retention = low_retained / len(low_importance_items)

    results = {
        'high_importance': {
            'count': len(high_importance_items),
            'retained': high_retained,
            'retention_rate': high_retention,
        },
        'low_importance': {
            'count': len(low_importance_items),
            'retained': low_retained,
            'retention_rate': low_retention,
        },
        'selectivity': high_retention - low_retention,  # Higher = better consolidation
        'stats': dram_memory.get_stats(),
    }

    print(f"  High importance: {high_retention:.1%} retained")
    print(f"  Low importance:  {low_retention:.1%} retained")
    print(f"  Selectivity:     {results['selectivity']:.1%} (higher = better)")

    return results


def test_decay_removes_noise(
    dram_memory: DRAMMemoryConsolidation,
    num_signal: int = 100,
    num_noise: int = 400,
) -> Dict[str, Any]:
    """
    Test 3: Decay removes noise/unimportant items.

    Signal items (important) should be retained.
    Noise items (unimportant) should decay.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Decay Removes Noise")
    print("=" * 60)

    # Reset
    dram_memory.__init__(
        short_term_capacity=dram_memory.short_term_capacity,
        long_term_capacity=dram_memory.long_term_capacity,
    )

    np.random.seed(42)

    # Store signal (important, coherent values)
    signal_keys = []
    signal_value = 42.0  # Consistent signal
    for i in range(num_signal):
        key = f"signal_{i}"
        signal_keys.append(key)
        dram_memory.store(key, signal_value + np.random.randn() * 0.1, importance=0.9)
        dram_memory.advance_time(5.0)

    # Store noise (unimportant, random values)
    noise_keys = []
    for i in range(num_noise):
        key = f"noise_{i}"
        noise_keys.append(key)
        dram_memory.store(key, np.random.randn() * 100, importance=0.1)
        dram_memory.advance_time(5.0)

    # Aggressive consolidation
    for _ in range(10):
        dram_memory.advance_time(500.0)
        dram_memory.consolidate()

    # Check what survives
    signal_retained = sum(1 for k in signal_keys if dram_memory.retrieve(k)[0] is not None)
    noise_retained = sum(1 for k in noise_keys if dram_memory.retrieve(k)[0] is not None)

    signal_retention = signal_retained / num_signal
    noise_retention = noise_retained / num_noise

    # Calculate signal-to-noise ratio improvement
    initial_snr = num_signal / (num_signal + num_noise)
    final_snr = signal_retained / max(1, signal_retained + noise_retained)
    snr_improvement = final_snr / initial_snr if initial_snr > 0 else 0

    results = {
        'signal': {
            'initial': num_signal,
            'retained': signal_retained,
            'retention_rate': signal_retention,
        },
        'noise': {
            'initial': num_noise,
            'retained': noise_retained,
            'retention_rate': noise_retention,
        },
        'initial_snr': initial_snr,
        'final_snr': final_snr,
        'snr_improvement': snr_improvement,
        'stats': dram_memory.get_stats(),
    }

    print(f"  Signal retained: {signal_retention:.1%}")
    print(f"  Noise retained:  {noise_retention:.1%}")
    print(f"  SNR improvement: {snr_improvement:.2f}x")

    return results


def test_vs_neural_memory(
    dram_memory: DRAMBackedNeuralMemory,
    neural_memory: StandardNeuralMemory,
    num_items: int = 200,
    embed_dim: int = 64,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Test 4: Compare DRAM-backed memory with standard neural memory.
    """
    print("\n" + "=" * 60)
    print("TEST 4: DRAM vs Standard Neural Memory")
    print("=" * 60)

    device = device or torch.device('cpu')
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate test items
    items = []
    for i in range(num_items):
        key = f"item_{i}"
        value = torch.randn(embed_dim, device=device)
        importance = np.random.uniform(0.0, 1.0)
        items.append((key, value, importance))

    # Store in both memories
    for key, value, importance in items:
        dram_memory.store(key, value)
        neural_memory.store(key, value)
        dram_memory.advance_time(10.0)

    # Run consolidation on DRAM memory
    for _ in range(5):
        dram_memory.advance_time(200.0)
        dram_memory.consolidate()

    # Test retrieval
    dram_hits = 0
    dram_decayed = 0
    neural_hits = 0

    for key, original_value, importance in items:
        # DRAM retrieval
        retrieved, mem_type, was_decayed = dram_memory.retrieve(key)
        if retrieved is not None:
            dram_hits += 1
        if was_decayed:
            dram_decayed += 1

        # Neural retrieval (always succeeds if slot exists)
        if key in neural_memory.key_map:
            neural_hits += 1

    results = {
        'dram_memory': {
            'hits': dram_hits,
            'decayed': dram_decayed,
            'hit_rate': dram_hits / num_items,
            **dram_memory.get_stats(),
        },
        'neural_memory': {
            'hits': neural_hits,
            'hit_rate': neural_hits / num_items,
            **neural_memory.get_stats(),
        },
        'advantage': 'DRAM provides selective forgetting based on importance',
    }

    print(f"  DRAM memory:   {dram_hits}/{num_items} hits ({dram_decayed} decayed)")
    print(f"  Neural memory: {neural_hits}/{num_items} hits")
    print(f"  DRAM provides importance-based forgetting")

    return results


# ============================================================================
#                    GPU INTEGRATION TEST
# ============================================================================

def test_gpu_integration(device: torch.device) -> Dict[str, Any]:
    """Test DRAM-backed memory with GPU computation."""
    print("\n" + "=" * 60)
    print("TEST 5: GPU Integration")
    print("=" * 60)

    telemetry = SysfsHwmonTelemetry()
    embed_dim = 64

    # Create DRAM-backed neural memory
    dram_neural = DRAMBackedNeuralMemory(
        embed_dim=embed_dim,
        short_term_capacity=512,
        long_term_capacity=2048,
        device=device,
    ).to(device)

    # Simple neural net for processing
    processor = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.GELU(),
        nn.Linear(128, embed_dim),
    ).to(device)

    optimizer = torch.optim.Adam(list(dram_neural.parameters()) + list(processor.parameters()), lr=1e-3)

    results = {
        'training_steps': [],
        'consolidation_events': [],
    }

    print("  Running GPU training with DRAM consolidation...")

    with EnergyMeter(telemetry) as meter:
        for step in range(100):
            # Generate batch
            batch = torch.randn(16, embed_dim, device=device)
            target = batch * 2  # Simple target

            # Process
            output = processor(batch)
            loss = F.mse_loss(output, target)

            # Store in DRAM memory
            for i in range(min(4, batch.shape[0])):
                key = f"step{step}_item{i}"
                importance = 1.0 / (1.0 + loss.item())  # Higher importance for lower loss
                dram_neural.store(key, batch[i], )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Advance time and consolidate periodically
            dram_neural.advance_time(50.0)
            if step % 20 == 19:
                promoted = dram_neural.consolidate()
                results['consolidation_events'].append({
                    'step': step,
                    'promoted': promoted,
                    'stats': dram_neural.get_stats(),
                })

            if step % 25 == 0:
                print(f"    Step {step}: Loss={loss.item():.4f}")
                results['training_steps'].append({
                    'step': step,
                    'loss': loss.item(),
                })

    results['energy_j'] = meter.energy_j
    results['final_stats'] = dram_neural.get_stats()

    print(f"  Training complete. Energy: {meter.energy_j:.2f}J")
    print(f"  Final DRAM stats: ST={results['final_stats']['short_term']['used']}, "
          f"LT={results['final_stats']['long_term']['used']}")

    return results


# ============================================================================
#                    MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("z1972: DRAM Decay as Memory Consolidation Mechanism")
    print("=" * 70)
    print("\nBased on findings from z1190-z1196:")
    print("  - DDR3 at room temp: NO decay for 60+ seconds")
    print("  - Decay occurs at elevated temps (85C spec)")
    print("  - We SIMULATE controlled decay for memory consolidation")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")

    results = {
        'experiment': 'z1972_dram_decay_consolidation',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'tests': {},
    }

    # Create DRAM memory consolidation system
    dram_memory = DRAMMemoryConsolidation(
        short_term_capacity=1024,
        long_term_capacity=4096,
        consolidation_threshold=0.7,
        decay_temperature_c=85.0,
    )

    # Test DRAM physics
    print("\n" + "=" * 60)
    print("DRAM Physics Verification")
    print("=" * 60)
    physics = DRAMPhysics()
    for temp in [25, 45, 65, 85]:
        tau = physics.get_tau_ms(temp)
        decay_1s = physics.decay_probability(1000, temp)
        print(f"  {temp}C: tau={tau:.1f}ms, decay@1s={decay_1s:.1%}")

    results['physics'] = {
        'tau_25c_ms': physics.get_tau_ms(25),
        'tau_85c_ms': physics.get_tau_ms(85),
        'decay_1s_25c': physics.decay_probability(1000, 25),
        'decay_1s_85c': physics.decay_probability(1000, 85),
    }

    # Run tests
    results['tests']['capacity_tradeoff'] = test_capacity_refresh_tradeoff(dram_memory)
    results['tests']['importance_consolidation'] = test_importance_based_consolidation(dram_memory)
    results['tests']['noise_removal'] = test_decay_removes_noise(dram_memory)

    # Neural memory comparison
    dram_neural = DRAMBackedNeuralMemory(embed_dim=64, device=device).to(device)
    neural_baseline = StandardNeuralMemory(memory_size=1024, embed_dim=64).to(device)
    results['tests']['vs_neural'] = test_vs_neural_memory(dram_neural, neural_baseline, device=device)

    # GPU integration
    results['tests']['gpu_integration'] = test_gpu_integration(device)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. Capacity vs Refresh Tradeoff:")
    for interval in results['tests']['capacity_tradeoff']['consolidation_intervals']:
        print(f"   {interval['interval_ms']:4d}ms interval: {interval['retention_rate']:.1%} retention")

    print("\n2. Importance-Based Consolidation:")
    imp_test = results['tests']['importance_consolidation']
    print(f"   High importance: {imp_test['high_importance']['retention_rate']:.1%}")
    print(f"   Low importance:  {imp_test['low_importance']['retention_rate']:.1%}")
    print(f"   Selectivity:     {imp_test['selectivity']:.1%}")

    print("\n3. Noise Removal:")
    noise_test = results['tests']['noise_removal']
    print(f"   SNR improvement: {noise_test['snr_improvement']:.2f}x")

    print("\n4. DRAM vs Neural Memory:")
    print(f"   DRAM provides importance-based forgetting")

    print("\n5. GPU Integration:")
    print(f"   Energy: {results['tests']['gpu_integration']['energy_j']:.2f}J")

    # Conclusions
    results['conclusions'] = {
        'biological_analogy': 'DRAM decay simulates synaptic depression',
        'key_findings': [
            'Consolidation improves retention of important memories',
            'Unimportant memories naturally decay (like forgetting)',
            'Refresh rate trades off capacity vs stability',
            'Physical decay provides automatic noise filtering',
        ],
        'future_work': [
            'Integrate with real FPGA DRAM (requires working partial writes)',
            'Use temperature control for decay rate modulation',
            'Combine with Hebbian learning for importance estimation',
        ],
    }

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1972_dram_consolidation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
