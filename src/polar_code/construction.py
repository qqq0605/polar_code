"""Polar code construction utilities for a BSC channel."""

from __future__ import annotations

from functools import lru_cache
import math

from .Bhatta import estimate_bhattacharyya_parameters_bsc
from .utils import (
    validate_block_length,
    validate_crossover_probability,
    validate_message_length,
)

DEFAULT_CONSTRUCTION_SAMPLES = 1_000
DEFAULT_CONSTRUCTION_SEED = 0


def bhattacharyya_bounds_bsc(block_length: int, crossover_probability: float) -> list[float]:
    """Return analytic Bhattacharyya upper bounds for every polarized bit-channel."""
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


@lru_cache(maxsize=None)
def _cached_sampled_bhattacharyya_parameters_bsc(
    block_length: int,
    crossover_probability: float,
    samples: int,
    seed: int | None,
) -> tuple[float, ...]:
    return estimate_bhattacharyya_parameters_bsc(
        block_length=block_length,
        crossover_probability=crossover_probability,
        samples=samples,
        seed=seed,
    )


def sampled_bhattacharyya_parameters_bsc(
    block_length: int,
    crossover_probability: float,
    samples: int = DEFAULT_CONSTRUCTION_SAMPLES,
    seed: int | None = DEFAULT_CONSTRUCTION_SEED,
) -> tuple[float, ...]:
    """Return sampled Bhattacharyya estimates used by the default construction."""
    validate_block_length(block_length)
    validate_crossover_probability(crossover_probability)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")
    return _cached_sampled_bhattacharyya_parameters_bsc(
        block_length=block_length,
        crossover_probability=crossover_probability,
        samples=samples,
        seed=seed,
    )


def select_information_bits(
    block_length: int,
    message_length: int,
    crossover_probability: float,
    samples: int = DEFAULT_CONSTRUCTION_SAMPLES,
    seed: int | None = DEFAULT_CONSTRUCTION_SEED,
) -> tuple[int, ...]:
    """Pick the K most reliable bit-channels."""
    validate_block_length(block_length)
    validate_message_length(message_length, block_length)
    reliabilities = sampled_bhattacharyya_parameters_bsc(
        block_length=block_length,
        crossover_probability=crossover_probability,
        samples=samples,
        seed=seed,
    )
    ranked = sorted(range(block_length), key=lambda index: (reliabilities[index], index))
    return tuple(sorted(ranked[:message_length]))

