"""Validation and bit-string helpers."""

from __future__ import annotations

from typing import Iterable, Sequence


def validate_block_length(block_length: int) -> None:
    if block_length <= 0 or block_length & (block_length - 1):
        raise ValueError("block_length must be a positive power of two.")


def validate_message_length(message_length: int, block_length: int) -> None:
    if not 1 <= message_length <= block_length:
        raise ValueError("message_length must satisfy 1 <= message_length <= block_length.")


def validate_crossover_probability(crossover_probability: float) -> None:
    if not 0.0 <= crossover_probability < 0.5:
        raise ValueError("crossover_probability must satisfy 0.0 <= p < 0.5 for a BSC.")


def ensure_binary_vector(
    bits: Sequence[int] | Iterable[int],
    expected_length: int | None = None,
    label: str = "bits",
) -> list[int]:
    vector = [int(bit) for bit in bits]
    if expected_length is not None and len(vector) != expected_length:
        raise ValueError(f"{label} must contain exactly {expected_length} bits.")

    invalid = [bit for bit in vector if bit not in (0, 1)]
    if invalid:
        raise ValueError(f"{label} must contain only binary values 0 or 1.")
    return vector


def bits_from_text(text: str) -> list[int]:
    stripped = [char for char in text if not char.isspace() and char != "_"]
    if not stripped:
        raise ValueError("bit string cannot be empty.")
    if any(char not in {"0", "1"} for char in stripped):
        raise ValueError("bit string must contain only 0 and 1.")
    return [int(char) for char in stripped]


def bits_to_text(bits: Sequence[int]) -> str:
    return "".join(str(bit) for bit in bits)

