"""Monte Carlo Bhattacharyya estimation for BSC polar-code construction.

This module implements the sampling idea from our discussion:

* sample channel outputs for the all-zero transmission,
* run the genie-aided SC LLR recursion for every synthetic bit-channel,
* estimate ``Z_i = E[exp(-L_i / 2)]`` by Monte Carlo,
* rank channels by the estimated Bhattacharyya values.

The implementation keeps only ``O(N)`` persistent state for the running
statistics and performs one SC-style traversal per sampled block, which costs
``O(N log N)`` time.

Example:
    from polar_code.Bhatta import (
        estimate_bhattacharyya_parameters_bsc,
        select_information_bits_monte_carlo,
    )

    reliabilities = estimate_bhattacharyya_parameters_bsc(
        block_length=8,
        crossover_probability=0.11,
        samples=2000,
        seed=7,
    )
    info_set = select_information_bits_monte_carlo(
        block_length=8,
        message_length=4,
        crossover_probability=0.11,
        samples=2000,
        seed=7,
    )
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .channel import bsc_llr
from .utils import (
    validate_block_length,
    validate_crossover_probability,
    validate_message_length,
)


def estimate_bhattacharyya_parameters_bsc(
    block_length: int,
    crossover_probability: float,
    samples: int,
    seed: int | None = None,
) -> tuple[float, ...]:
    """Estimate synthetic-channel Bhattacharyya parameters for a BSC."""
    validate_block_length(block_length)
    validate_crossover_probability(crossover_probability)
    if samples <= 0:
        raise ValueError("samples must be a positive integer.")

    rng = random.Random(seed)
    # Store log(sum(exp(-L_i / 2))) so large sample counts stay numerically stable.
    log_sums = [-math.inf] * block_length

    for _ in range(samples):
        received = _sample_bsc_output_for_zero_codeword(
            block_length=block_length,
            crossover_probability=crossover_probability,
            rng=rng,
        )
        llr = bsc_llr(received, crossover_probability)
        _accumulate_leaf_log_sums(llr, offset=0, log_sums=log_sums)

    log_samples = math.log(samples)
    estimates: list[float] = []
    for log_sum in log_sums:
        if math.isinf(log_sum) and log_sum < 0.0:
            estimates.append(0.0)
            continue
        estimates.append(math.exp(log_sum - log_samples))
    return tuple(estimates)


def select_information_bits_monte_carlo(
    block_length: int,
    message_length: int,
    crossover_probability: float,
    samples: int,
    seed: int | None = None,
) -> tuple[int, ...]:
    """Pick the K most reliable bit-channels from sampled Bhattacharyya values."""
    validate_block_length(block_length)
    validate_message_length(message_length, block_length)
    reliabilities = estimate_bhattacharyya_parameters_bsc(
        block_length=block_length,
        crossover_probability=crossover_probability,
        samples=samples,
        seed=seed,
    )
    ranked = sorted(range(block_length), key=lambda index: (reliabilities[index], index))
    return tuple(sorted(ranked[:message_length]))


def _sample_bsc_output_for_zero_codeword(
    block_length: int,
    crossover_probability: float,
    rng: random.Random,
) -> list[int]:
    # By symmetry, sampling around the all-zero transmission is enough.
    if crossover_probability == 0.0:
        return [0] * block_length
    return [int(rng.random() < crossover_probability) for _ in range(block_length)]


def _accumulate_leaf_log_sums(
    llr: Sequence[float],
    offset: int,
    log_sums: list[float],
) -> list[int]:
    """Accumulate ``log(sum(exp(-L_i / 2)))`` along a genie-aided SC traversal."""
    if len(llr) == 1:
        log_sums[offset] = _logaddexp(log_sums[offset], -0.5 * llr[0])
        return [0]

    half = len(llr) // 2
    # Recurse on the left child exactly as SC decoding would build its LLR.
    left_llr = [
        _f_function(left, right)
        for left, right in zip(llr[:half], llr[half:])
    ]
    left_partial_sums = _accumulate_leaf_log_sums(left_llr, offset=offset, log_sums=log_sums)

    # Use genie-known previous bits, which are all zero in this construction.
    right_llr = [
        _g_function(left, right, bit)
        for left, right, bit in zip(llr[:half], llr[half:], left_partial_sums)
    ]
    right_partial_sums = _accumulate_leaf_log_sums(
        right_llr,
        offset=offset + half,
        log_sums=log_sums,
    )

    return [
        left_bit ^ right_bit
        for left_bit, right_bit in zip(left_partial_sums, right_partial_sums)
    ] + right_partial_sums


def _f_function(left: float, right: float) -> float:
    numerator = _logaddexp(0.0, left + right)
    denominator = _logaddexp(left, right)
    return numerator - denominator


def _g_function(left: float, right: float, bit_estimate: int) -> float:
    return right + (1 - 2 * bit_estimate) * left


def _logaddexp(x_value: float, y_value: float) -> float:
    maximum = max(x_value, y_value)
    minimum = min(x_value, y_value)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log1p(math.exp(minimum - maximum))
