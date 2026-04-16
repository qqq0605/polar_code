"""Command-line interface for the polar code project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .channel import bsc_transmit
from .codec import PolarCode
from .construction import DEFAULT_CONSTRUCTION_SAMPLES, DEFAULT_CONSTRUCTION_SEED
from .plots import generate_default_plots
from .search import find_maximum_message_length
from .utils import bits_from_text, bits_to_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polar-code",
        description="Polar code encoder/decoder for BSC channels.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    channel_common = argparse.ArgumentParser(add_help=False)
    channel_common.add_argument("--block-length", type=int, required=True, help="Code length N = 2^k.")
    channel_common.add_argument(
        "--crossover-probability",
        type=float,
        required=True,
        help="BSC crossover probability p.",
    )
    common = argparse.ArgumentParser(add_help=False, parents=[channel_common])
    common.add_argument("--message-length", type=int, required=True, help="Message length K.")

    encode_parser = subparsers.add_parser("encode", parents=[common], help="Encode a message.")
    encode_parser.add_argument("--message", required=True, help="Binary message, e.g. 1011.")

    decode_parser = subparsers.add_parser("decode", parents=[common], help="Decode a received word.")
    decode_parser.add_argument("--received", required=True, help="Received binary vector.")

    simulate_parser = subparsers.add_parser(
        "simulate",
        parents=[common],
        help="Run Monte-Carlo simulation over a BSC.",
    )
    simulate_parser.add_argument("--trials", type=int, required=True, help="Number of trials.")
    simulate_parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")

    transmit_parser = subparsers.add_parser(
        "transmit",
        parents=[common],
        help="Encode, pass through BSC, and decode one message.",
    )
    transmit_parser.add_argument("--message", required=True, help="Binary message, e.g. 1011.")
    transmit_parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")

    search_parser = subparsers.add_parser(
        "find-message-length",
        parents=[channel_common],
        help="Find the largest K that satisfies an FER target.",
    )
    search_parser.add_argument("--trials", type=int, required=True, help="Number of trials.")
    search_parser.add_argument(
        "--target-frame-error-rate",
        type=float,
        required=True,
        help="FER target used to accept or reject a message length.",
    )
    search_parser.add_argument("--seed", type=int, default=None, help="Optional simulation RNG seed.")
    search_parser.add_argument(
        "--construction-samples",
        type=int,
        default=DEFAULT_CONSTRUCTION_SAMPLES,
        help="Monte Carlo samples used by the sampled-Bhattacharyya construction.",
    )
    search_parser.add_argument(
        "--construction-seed",
        type=int,
        default=DEFAULT_CONSTRUCTION_SEED,
        help="Seed used by the sampled-Bhattacharyya construction.",
    )
    search_parser.add_argument(
        "--lower-bound",
        type=int,
        default=None,
        help="Optional lower hint for the binary search interval.",
    )
    search_parser.add_argument(
        "--upper-bound",
        type=int,
        default=None,
        help="Optional upper hint for the binary search interval.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate BER/FER sweep plots as SVG and CSV.",
    )
    plot_parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store SVG and CSV outputs.",
    )
    plot_parser.add_argument(
        "--trials",
        type=int,
        default=2000,
        help="Simulation trials for each point.",
    )
    plot_parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed used for reproducible sweeps.",
    )
    plot_parser.add_argument(
        "--crossover-probability",
        type=float,
        default=0.05,
        help="BSC crossover probability p used in both plots.",
    )
    plot_parser.add_argument(
        "--block-lengths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Block lengths used in the vs-N sweep.",
    )
    plot_parser.add_argument(
        "--code-rate",
        type=float,
        default=0.5,
        help="Target code rate used in the vs-N sweep.",
    )
    plot_parser.add_argument(
        "--rate-block-length",
        type=int,
        default=64,
        help="Fixed block length used in the vs-code-rate sweep.",
    )
    plot_parser.add_argument(
        "--message-lengths",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32, 40],
        help="Message lengths used in the vs-code-rate sweep.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "plot":
            outputs = generate_default_plots(
                output_dir=Path(args.output_dir),
                trials=args.trials,
                seed=args.seed,
                crossover_probability=args.crossover_probability,
                block_lengths=args.block_lengths,
                code_rate=args.code_rate,
                rate_block_length=args.rate_block_length,
                message_lengths=args.message_lengths,
            )
            for name, path in outputs.items():
                print(f"{name}={path}")
            return 0

        if args.command == "find-message-length":
            result = find_maximum_message_length(
                block_length=args.block_length,
                crossover_probability=args.crossover_probability,
                trials=args.trials,
                target_frame_error_rate=args.target_frame_error_rate,
                seed=args.seed,
                construction_samples=args.construction_samples,
                construction_seed=args.construction_seed,
                lower_bound=args.lower_bound,
                upper_bound=args.upper_bound,
            )
            print(f"trials={result.trials}")
            print(f"target_frame_error_rate={result.target_frame_error_rate:.8f}")
            print(f"max_frame_errors={result.max_frame_errors}")
            print(f"construction_samples={result.construction_samples}")
            print(f"construction_seed={result.construction_seed}")
            if result.best_pass is None:
                print("best_message_length=0")
                print("best_code_rate=0.00000000")
            else:
                print(f"best_message_length={result.best_pass.message_length}")
                print(f"best_code_rate={result.best_pass.message_length / result.block_length:.8f}")
                print(f"best_checked_trials={result.best_pass.checked_trials}")
                print(f"best_bit_errors={result.best_pass.bit_errors}")
                print(f"best_frame_errors={result.best_pass.frame_errors}")
                print(f"best_frame_error_rate={result.best_pass.frame_error_rate:.8f}")
            if result.next_fail is None:
                print("next_fail_message_length=None")
            else:
                print(f"next_fail_message_length={result.next_fail.message_length}")
                print(f"next_fail_checked_trials={result.next_fail.checked_trials}")
                print(f"next_fail_bit_errors={result.next_fail.bit_errors}")
                print(f"next_fail_frame_errors={result.next_fail.frame_errors}")
                print(f"next_fail_frame_error_rate={result.next_fail.frame_error_rate:.8f}")
            return 0

        codec = PolarCode(
            block_length=args.block_length,
            message_length=args.message_length,
            crossover_probability=args.crossover_probability,
        )

        if args.command == "encode":
            message = bits_from_text(args.message)
            source = codec.build_source_vector(message)
            codeword = codec.encode(message)
            print(f"info_set={codec.info_set}")
            print(f"source={bits_to_text(source)}")
            print(f"codeword={bits_to_text(codeword)}")
            return 0

        if args.command == "decode":
            received = bits_from_text(args.received)
            decoded = codec.decode(received)
            print(f"info_set={decoded.info_set}")
            print(f"estimated_source={bits_to_text(decoded.estimated_source)}")
            print(f"estimated_message={bits_to_text(decoded.estimated_message)}")
            return 0

        if args.command == "simulate":
            result = codec.simulate(trials=args.trials, seed=args.seed)
            print(f"info_set={codec.info_set}")
            print(f"trials={result.trials}")
            print(f"bit_errors={result.bit_errors}")
            print(f"frame_errors={result.frame_errors}")
            print(f"bit_error_rate={result.bit_error_rate:.8f}")
            print(f"frame_error_rate={result.frame_error_rate:.8f}")
            return 0

        if args.command == "transmit":
            import random

            message = bits_from_text(args.message)
            rng = random.Random(args.seed)
            codeword = codec.encode(message)
            received = bsc_transmit(codeword, codec.crossover_probability, rng=rng)
            decoded = codec.decode(received)
            print(f"info_set={codec.info_set}")
            print(f"message={bits_to_text(message)}")
            print(f"codeword={bits_to_text(codeword)}")
            print(f"received={bits_to_text(received)}")
            print(f"estimated_message={bits_to_text(decoded.estimated_message)}")
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
