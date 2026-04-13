"""Command-line interface for the polar code project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .channel import bsc_transmit
from .codec import PolarCode
from .plots import generate_default_plots
from .utils import bits_from_text, bits_to_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polar-code",
        description="Polar code encoder/decoder for BSC channels.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--block-length", type=int, required=True, help="Code length N = 2^k.")
    common.add_argument("--message-length", type=int, required=True, help="Message length K.")
    common.add_argument(
        "--crossover-probability",
        type=float,
        required=True,
        help="BSC crossover probability p.",
    )

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
