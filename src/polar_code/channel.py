"""Binary symmetric channel helpers."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from .utils import ensure_binary_vector, validate_crossover_probability


def bsc_transmit(
    codeword: Sequence[int] | Iterable[int],
    crossover_probability: float,
    rng: random.Random | None = None,
) -> list[int]:
    """Transmit a binary codeword through a BSC."""
    validate_crossover_probability(crossover_probability)
    bits = ensure_binary_vector(codeword, label="codeword")
    source = rng if rng is not None else random.Random()
    return [bit ^ int(source.random() < crossover_probability) for bit in bits]


def bsc_llr(received_bits: Sequence[int] | Iterable[int], crossover_probability: float) -> list[float]:
    """Convert received BSC outputs into channel LLRs."""
    validate_crossover_probability(crossover_probability)
    bits = ensure_binary_vector(received_bits, label="received bits")

    if crossover_probability == 0.0:
        magnitude = 1_000_000.0
    else:
        magnitude = math.log((1.0 - crossover_probability) / crossover_probability)

    return [magnitude if bit == 0 else -magnitude for bit in bits]

