"""Core polar encoder/decoder implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from .channel import bsc_llr, bsc_transmit
from .construction import bhattacharyya_bounds_bsc, select_information_bits
from .utils import (
    ensure_binary_vector,
    validate_block_length,
    validate_crossover_probability,
    validate_message_length,
)


@dataclass(frozen=True)
class DecodeResult:
    estimated_message: tuple[int, ...]
    estimated_source: tuple[int, ...]
    info_set: tuple[int, ...]


@dataclass(frozen=True)
class SimulationResult:
    trials: int
    bit_errors: int
    frame_errors: int
    bit_error_rate: float
    frame_error_rate: float


class PolarCode:
    """Polar code over a BSC with Bhattacharyya construction and SC decoding."""

    def __init__(
        self,
        block_length: int,
        message_length: int,
        crossover_probability: float,
    ) -> None:
        validate_block_length(block_length)
        validate_message_length(message_length, block_length)
        validate_crossover_probability(crossover_probability)

        self.block_length = block_length
        self.message_length = message_length
        self.crossover_probability = crossover_probability
        self.reliabilities = tuple(
            bhattacharyya_bounds_bsc(block_length, crossover_probability)
        )
        self.info_set = select_information_bits(
            block_length=block_length,
            message_length=message_length,
            crossover_probability=crossover_probability,
        )
        info_set_lookup = set(self.info_set)
        self.frozen_set = tuple(
            index for index in range(block_length) if index not in info_set_lookup
        )
        self._is_frozen = tuple(index not in info_set_lookup for index in range(block_length))

    def build_source_vector(self, message_bits: Sequence[int]) -> list[int]:
        message = ensure_binary_vector(
            message_bits,
            expected_length=self.message_length,
            label="message bits",
        )
        source = [0] * self.block_length
        for info_index, bit in zip(self.info_set, message):
            source[info_index] = bit
        return source

    def encode(self, message_bits: Sequence[int]) -> list[int]:
        source = self.build_source_vector(message_bits)
        return self._polar_transform(source)

    def decode(self, received_bits: Sequence[int]) -> DecodeResult:
        received = ensure_binary_vector(
            received_bits,
            expected_length=self.block_length,
            label="received bits",
        )
        llr = bsc_llr(received, self.crossover_probability)
        estimated_source, _ = self._sc_decode(llr, offset=0)
        estimated_source = tuple(estimated_source)
        estimated_message = tuple(estimated_source[index] for index in self.info_set)
        return DecodeResult(
            estimated_message=estimated_message,
            estimated_source=estimated_source,
            info_set=self.info_set,
        )

    def simulate(self, trials: int, seed: int | None = None) -> SimulationResult:
        if trials <= 0:
            raise ValueError("trials must be a positive integer.")

        rng = random.Random(seed)
        bit_errors = 0
        frame_errors = 0

        for _ in range(trials):
            message = [rng.randrange(2) for _ in range(self.message_length)]
            codeword = self.encode(message)
            received = bsc_transmit(codeword, self.crossover_probability, rng=rng)
            decoded = self.decode(received)

            errors = sum(
                int(expected != observed)
                for expected, observed in zip(message, decoded.estimated_message)
            )
            bit_errors += errors
            frame_errors += int(errors > 0)

        return SimulationResult(
            trials=trials,
            bit_errors=bit_errors,
            frame_errors=frame_errors,
            bit_error_rate=bit_errors / (trials * self.message_length),
            frame_error_rate=frame_errors / trials,
        )

    @staticmethod
    def _polar_transform(source_bits: Sequence[int]) -> list[int]:
        vector = ensure_binary_vector(source_bits, label="source bits")
        if len(vector) == 1:
            return vector

        half = len(vector) // 2
        left = vector[:half]
        right = vector[half:]
        upper = [left_bit ^ right_bit for left_bit, right_bit in zip(left, right)]
        lower = right
        return PolarCode._polar_transform(upper) + PolarCode._polar_transform(lower)

    def _sc_decode(self, llr: Sequence[float], offset: int) -> tuple[list[int], list[int]]:
        if len(llr) == 1:
            if self._is_frozen[offset]:
                return [0], [0]
            decision = 0 if llr[0] >= 0.0 else 1
            return [decision], [decision]

        half = len(llr) // 2
        left_llr = [
            self._f_function(left, right)
            for left, right in zip(llr[:half], llr[half:])
        ]
        left_source, left_partial_sums = self._sc_decode(left_llr, offset=offset)

        right_llr = [
            self._g_function(left, right, bit)
            for left, right, bit in zip(llr[:half], llr[half:], left_partial_sums)
        ]
        right_source, right_partial_sums = self._sc_decode(right_llr, offset=offset + half)

        source_bits = left_source + right_source
        partial_sums = [
            left_bit ^ right_bit
            for left_bit, right_bit in zip(left_partial_sums, right_partial_sums)
        ] + right_partial_sums
        return source_bits, partial_sums

    @staticmethod
    def _f_function(left: float, right: float) -> float:
        numerator = PolarCode._logaddexp(0.0, left + right)
        denominator = PolarCode._logaddexp(left, right)
        return numerator - denominator

    @staticmethod
    def _g_function(left: float, right: float, bit_estimate: int) -> float:
        return right + (1 - 2 * bit_estimate) * left

    @staticmethod
    def _logaddexp(x_value: float, y_value: float) -> float:
        maximum = max(x_value, y_value)
        minimum = min(x_value, y_value)
        if math.isinf(maximum):
            return maximum
        return maximum + math.log1p(math.exp(minimum - maximum))
