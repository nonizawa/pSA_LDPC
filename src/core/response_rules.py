"""Vector response rules used by the lightweight validation suite.

The equations match the update branches in ``src/ldpc/ldpc_pbit.py``.  This
small module exposes them without requiring a full decoding experiment.
"""

from __future__ import annotations

import numpy as np


def apply_response_rule(
    current: np.ndarray,
    previous: np.ndarray,
    noise: np.ndarray,
    rule: str,
    *,
    coefficient: float = 0.0,
    source_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the stochastic discriminant and stored response for one update."""
    current = np.asarray(current, dtype=float)
    previous = np.asarray(previous, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if current.shape != previous.shape or current.shape != noise.shape:
        raise ValueError("current, previous, and noise must have identical shapes")
    if rule == "pSA":
        stored = current
        discriminant = current + noise
    elif rule == "additive":
        stored = current
        discriminant = current + coefficient * previous + noise
    elif rule == "normalized":
        stored = current
        discriminant = (current + coefficient * previous) / (1.0 + coefficient) + noise
    elif rule == "gain_only":
        stored = current
        discriminant = (1.0 + coefficient) * current + noise
    elif rule == "finite_response":
        stored = coefficient * previous + (1.0 - coefficient) * current
        discriminant = stored + noise
    elif rule == "shuffled":
        if source_index is None:
            raise ValueError("shuffled rule requires source_index")
        source_index = np.asarray(source_index, dtype=int)
        if source_index.shape != (current.size,):
            raise ValueError("source_index must contain one source for each bit")
        stored = current
        discriminant = current + coefficient * previous[source_index] + noise
    else:
        raise ValueError(f"unknown response rule: {rule}")
    return discriminant, stored


def shuffled_sources(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Construct a derangement that exactly preserves response-vector moments."""
    if n_bits < 2:
        raise ValueError("a derangement requires at least two bits")
    relabeling = rng.permutation(n_bits)
    shift = int(rng.integers(1, n_bits))
    source = np.empty(n_bits, dtype=int)
    source[relabeling] = np.roll(relabeling, shift)
    return source
