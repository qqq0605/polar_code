"""Search helpers for finding the largest message length under an FER target."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .channel import bsc_transmit
from .codec import PolarCode
from .construction import DEFAULT_CONSTRUCTION_SAMPLES, DEFAULT_CONSTRUCTION_SEED
from .utils import (
    validate_block_length,
    validate_crossover_probability,
    validate_message_length,
)


@dataclass(frozen=True)
class MessageLengthEvaluation:
    message_length: int
    checked_trials: int
    bit_errors: int
    frame_errors: int
    bit_error_rate: float
    frame_error_rate: float
    passed: bool


@dataclass(frozen=True)
class MessageLengthSearchResult:
    block_length: int
    crossover_probability: float
    trials: int
    target_frame_error_rate: float
    max_frame_errors: int
    seed: int | None
    construction_samples: int
    construction_seed: int | None
    best_pass: MessageLengthEvaluation | None
    next_fail: MessageLengthEvaluation | None


def evaluate_message_length(
    block_length: int,
    message_length: int,
    crossover_probability: float,
    trials: int,
    *,
    max_frame_errors: int | None = None,
    seed: int | None = None,
    construction_samples: int = DEFAULT_CONSTRUCTION_SAMPLES,
    construction_seed: int | None = DEFAULT_CONSTRUCTION_SEED,
) -> MessageLengthEvaluation:
    """Evaluate one message length by Monte Carlo simulation.

    When ``max_frame_errors`` is set, the simulation stops early as soon as the
    target is impossible to satisfy.
    """

    validate_block_length(block_length)
    validate_message_length(message_length, block_length)
    validate_crossover_probability(crossover_probability)
    if trials <= 0:
        raise ValueError("trials must be a positive integer.")
    if max_frame_errors is not None and max_frame_errors < 0:
        raise ValueError("max_frame_errors must be non-negative when provided.")

    codec = PolarCode(
        block_length=block_length,
        message_length=message_length,
        crossover_probability=crossover_probability,
        construction_samples=construction_samples,
        construction_seed=construction_seed,
    )
    message_rng = random.Random(seed)
    channel_rng = random.Random(None if seed is None else seed + 1)

    bit_errors = 0
    frame_errors = 0
    checked_trials = 0

    for checked_trials in range(1, trials + 1):
        message = [message_rng.randrange(2) for _ in range(message_length)]
        codeword = codec.encode(message)
        received = bsc_transmit(codeword, crossover_probability, rng=channel_rng)
        decoded = codec.decode(received)

        errors = sum(
            int(expected != observed)
            for expected, observed in zip(message, decoded.estimated_message)
        )
        bit_errors += errors
        if errors > 0:
            frame_errors += 1
            # Once we exceed the allowed frame errors, the FER target cannot recover.
            if max_frame_errors is not None and frame_errors > max_frame_errors:
                break

    bit_error_rate = bit_errors / (checked_trials * message_length)
    frame_error_rate = frame_errors / checked_trials
    passed = (
        max_frame_errors is None
        or (checked_trials == trials and frame_errors <= max_frame_errors)
    )
    return MessageLengthEvaluation(
        message_length=message_length,
        checked_trials=checked_trials,
        bit_errors=bit_errors,
        frame_errors=frame_errors,
        bit_error_rate=bit_error_rate,
        frame_error_rate=frame_error_rate,
        passed=passed,
    )


def find_maximum_message_length(
    block_length: int,
    crossover_probability: float,
    trials: int,
    target_frame_error_rate: float,
    *,
    seed: int | None = None,
    construction_samples: int = DEFAULT_CONSTRUCTION_SAMPLES,
    construction_seed: int | None = DEFAULT_CONSTRUCTION_SEED,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
) -> MessageLengthSearchResult:
    """Find the largest message length whose FER target is satisfied.

    The search assumes that larger message lengths are not easier than smaller
    ones, and uses a pass/fail bracket followed by binary search.
    """

    validate_block_length(block_length)
    validate_crossover_probability(crossover_probability)
    if trials <= 0:
        raise ValueError("trials must be a positive integer.")
    if not 0.0 <= target_frame_error_rate <= 1.0:
        raise ValueError("target_frame_error_rate must satisfy 0.0 <= value <= 1.0.")

    low = 1 if lower_bound is None else lower_bound
    high = block_length if upper_bound is None else upper_bound
    if not 1 <= low <= high <= block_length:
        raise ValueError("Bounds must satisfy 1 <= lower_bound <= upper_bound <= block_length.")

    max_frame_errors = math.floor(trials * target_frame_error_rate + 1e-12)
    evaluations: dict[int, MessageLengthEvaluation] = {}

    def evaluate(message_length: int) -> MessageLengthEvaluation:
        if message_length not in evaluations:
            evaluations[message_length] = evaluate_message_length(
                block_length=block_length,
                message_length=message_length,
                crossover_probability=crossover_probability,
                trials=trials,
                max_frame_errors=max_frame_errors,
                seed=seed,
                construction_samples=construction_samples,
                construction_seed=construction_seed,
            )
        return evaluations[message_length]

    low_evaluation = evaluate(low)
    if not low_evaluation.passed and low != 1:
        high = low
        low = 1
        low_evaluation = evaluate(low)
    if not low_evaluation.passed:
        return MessageLengthSearchResult(
            block_length=block_length,
            crossover_probability=crossover_probability,
            trials=trials,
            target_frame_error_rate=target_frame_error_rate,
            max_frame_errors=max_frame_errors,
            seed=seed,
            construction_samples=construction_samples,
            construction_seed=construction_seed,
            best_pass=None,
            next_fail=low_evaluation,
        )

    high_evaluation = evaluate(high)
    if high_evaluation.passed and high != block_length:
        low = high
        low_evaluation = high_evaluation
        high = block_length
        high_evaluation = evaluate(high)
    if high_evaluation.passed:
        return MessageLengthSearchResult(
            block_length=block_length,
            crossover_probability=crossover_probability,
            trials=trials,
            target_frame_error_rate=target_frame_error_rate,
            max_frame_errors=max_frame_errors,
            seed=seed,
            construction_samples=construction_samples,
            construction_seed=construction_seed,
            best_pass=high_evaluation,
            next_fail=None,
        )

    while low + 1 < high:
        midpoint = (low + high) // 2
        midpoint_evaluation = evaluate(midpoint)
        if midpoint_evaluation.passed:
            low = midpoint
            low_evaluation = midpoint_evaluation
        else:
            high = midpoint
            high_evaluation = midpoint_evaluation

    return MessageLengthSearchResult(
        block_length=block_length,
        crossover_probability=crossover_probability,
        trials=trials,
        target_frame_error_rate=target_frame_error_rate,
        max_frame_errors=max_frame_errors,
        seed=seed,
        construction_samples=construction_samples,
        construction_seed=construction_seed,
        best_pass=low_evaluation,
        next_fail=high_evaluation,
    )
