"""Polar code construction utilities for a BSC channel."""

from __future__ import annotations

import math

from .utils import (
    validate_block_length,
    validate_crossover_probability,
    validate_message_length,
)


def bhattacharyya_bounds_bsc(block_length: int, crossover_probability: float) -> list[float]:
    """Return Bhattacharyya upper bounds for every polarized bit-channel."""
    validate_block_length(block_length)
    validate_crossover_probability(crossover_probability)

    if crossover_probability == 0.0:
        base = 0.0
    else:
        base = 2.0 * math.sqrt(crossover_probability * (1.0 - crossover_probability))

    reliabilities = [base]
    while len(reliabilities) < block_length:
        next_stage: list[float] = []
        for z_value in reliabilities:
            worse = min(1.0, 2.0 * z_value - z_value * z_value)
            better = max(0.0, z_value * z_value)
            next_stage.extend((worse, better))
        reliabilities = next_stage
    return reliabilities


def select_information_bits(
    block_length: int,
    message_length: int,
    crossover_probability: float,
) -> tuple[int, ...]:
    """Pick the K most reliable bit-channels."""
    validate_block_length(block_length)
    validate_message_length(message_length, block_length)
    reliabilities = bhattacharyya_bounds_bsc(block_length, crossover_probability)
    ranked = sorted(range(block_length), key=lambda index: (reliabilities[index], index))
    return tuple(sorted(ranked[:message_length]))

