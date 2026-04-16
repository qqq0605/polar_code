"""Performance sweeps and SVG plot generation for polar codes."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from html import escape
import math
from pathlib import Path
from typing import Callable, Sequence

from .codec import PolarCode
from .utils import validate_block_length


@dataclass(frozen=True)
class CurvePoint:
    label: str
    x_value: float
    block_length: int
    message_length: int
    code_rate: float
    trials: int
    seed: int | None
    bit_error_rate: float
    frame_error_rate: float


def sweep_block_lengths(
    block_lengths: Sequence[int],
    code_rate: float,
    crossover_probability: float,
    trials: int,
    seed: int | None = None,
) -> list[CurvePoint]:
    if not block_lengths:
        raise ValueError("block_lengths must not be empty.")
    if not 0.0 < code_rate <= 1.0:
        raise ValueError("code_rate must satisfy 0.0 < code_rate <= 1.0.")

    points: list[CurvePoint] = []
    for index, block_length in enumerate(block_lengths):
        validate_block_length(block_length)
        message_length = max(1, min(block_length, round(block_length * code_rate)))
        point_seed = None if seed is None else seed + index
        codec = PolarCode(
            block_length=block_length,
            message_length=message_length,
            crossover_probability=crossover_probability,
        )
        result = codec.simulate(trials=trials, seed=point_seed)
        points.append(
            CurvePoint(
                label=str(block_length),
                x_value=float(block_length),
                block_length=block_length,
                message_length=message_length,
                code_rate=message_length / block_length,
                trials=trials,
                seed=point_seed,
                bit_error_rate=result.bit_error_rate,
                frame_error_rate=result.frame_error_rate,
            )
        )
    return points


def sweep_code_rates(
    block_length: int,
    message_lengths: Sequence[int],
    crossover_probability: float,
    trials: int,
    seed: int | None = None,
) -> list[CurvePoint]:
    validate_block_length(block_length)
    if not message_lengths:
        raise ValueError("message_lengths must not be empty.")

    points: list[CurvePoint] = []
    for index, message_length in enumerate(message_lengths):
        point_seed = None if seed is None else seed + index
        codec = PolarCode(
            block_length=block_length,
            message_length=message_length,
            crossover_probability=crossover_probability,
        )
        result = codec.simulate(trials=trials, seed=point_seed)
        code_rate = message_length / block_length
        points.append(
            CurvePoint(
                label=f"{code_rate:.3f}",
                x_value=code_rate,
                block_length=block_length,
                message_length=message_length,
                code_rate=code_rate,
                trials=trials,
                seed=point_seed,
                bit_error_rate=result.bit_error_rate,
                frame_error_rate=result.frame_error_rate,
            )
        )
    return points


def save_curve_csv(path: str | Path, points: Sequence[CurvePoint]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "x_value",
                "block_length",
                "message_length",
                "code_rate",
                "trials",
                "seed",
                "bit_error_rate",
                "frame_error_rate",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.label,
                    point.x_value,
                    point.block_length,
                    point.message_length,
                    f"{point.code_rate:.12f}",
                    point.trials,
                    point.seed,
                    f"{point.bit_error_rate:.12f}",
                    f"{point.frame_error_rate:.12f}",
                ]
            )


def save_dual_metric_svg(
    path: str | Path,
    points: Sequence[CurvePoint],
    *,
    title: str,
    subtitle: str,
    x_axis_label: str,
    x_tick_formatter: Callable[[CurvePoint], str] | None = None,
    x_transform: Callable[[float], float] | None = None,
) -> None:
    if not points:
        raise ValueError("points must not be empty.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tick_formatter = x_tick_formatter if x_tick_formatter is not None else lambda point: point.label
    transform = x_transform if x_transform is not None else lambda value: value

    width = 960
    height = 600
    left_margin = 90
    right_margin = 40
    top_margin = 95
    bottom_margin = 80
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin
    chart_left = left_margin
    chart_top = top_margin
    chart_bottom = top_margin + chart_height

    x_values = [transform(point.x_value) for point in points]
    min_x = min(x_values)
    max_x = max(x_values)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0

    max_metric = max(
        max(point.bit_error_rate, point.frame_error_rate)
        for point in points
    )
    y_max = _nice_upper(max_metric * 1.15 if max_metric > 0.0 else 0.02)
    y_ticks = _build_y_ticks(y_max)

    def scale_x(value: float) -> float:
        transformed = transform(value)
        return chart_left + (transformed - min_x) / (max_x - min_x) * chart_width

    def scale_y(value: float) -> float:
        return chart_bottom - (value / y_max) * chart_height

    def build_path(metric_getter: Callable[[CurvePoint], float]) -> str:
        commands: list[str] = []
        for index, point in enumerate(points):
            prefix = "M" if index == 0 else "L"
            commands.append(f"{prefix} {scale_x(point.x_value):.2f} {scale_y(metric_getter(point)):.2f}")
        return " ".join(commands)

    ber_path = build_path(lambda point: point.bit_error_rate)
    fer_path = build_path(lambda point: point.frame_error_rate)

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf7" />',
        f'<text x="{chart_left}" y="42" font-size="28" font-family="Helvetica, Arial, sans-serif" fill="#111827">{escape(title)}</text>',
        f'<text x="{chart_left}" y="68" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{escape(subtitle)}</text>',
        f'<line x1="{chart_left}" y1="{chart_bottom}" x2="{chart_left + chart_width}" y2="{chart_bottom}" stroke="#111827" stroke-width="2"/>',
        f'<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_bottom}" stroke="#111827" stroke-width="2"/>',
    ]

    for tick_value in y_ticks:
        y_position = scale_y(tick_value)
        svg_lines.append(
            f'<line x1="{chart_left}" y1="{y_position:.2f}" x2="{chart_left + chart_width}" y2="{y_position:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{chart_left - 12}" y="{y_position + 5:.2f}" text-anchor="end" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#374151">{tick_value:.3f}</text>'
        )

    for point in points:
        x_position = scale_x(point.x_value)
        svg_lines.append(
            f'<line x1="{x_position:.2f}" y1="{chart_bottom}" x2="{x_position:.2f}" y2="{chart_bottom + 6}" stroke="#111827" stroke-width="1.5"/>'
        )
        svg_lines.append(
            f'<text x="{x_position:.2f}" y="{chart_bottom + 28}" text-anchor="middle" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#374151">{escape(tick_formatter(point))}</text>'
        )

    svg_lines.extend(
        [
            f'<path d="{ber_path}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
            f'<path d="{fer_path}" fill="none" stroke="#dc2626" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
        ]
    )

    for point in points:
        ber_x = scale_x(point.x_value)
        ber_y = scale_y(point.bit_error_rate)
        fer_y = scale_y(point.frame_error_rate)
        svg_lines.append(
            f'<circle cx="{ber_x:.2f}" cy="{ber_y:.2f}" r="5" fill="#0f766e"/>'
        )
        svg_lines.append(
            f'<circle cx="{ber_x:.2f}" cy="{fer_y:.2f}" r="5" fill="#dc2626"/>'
        )

    legend_x = chart_left + chart_width - 160
    legend_y = 48
    svg_lines.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 32}" y2="{legend_y}" stroke="#0f766e" stroke-width="4"/>',
            f'<text x="{legend_x + 42}" y="{legend_y + 5}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">BER</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 26}" x2="{legend_x + 32}" y2="{legend_y + 26}" stroke="#dc2626" stroke-width="4"/>',
            f'<text x="{legend_x + 42}" y="{legend_y + 31}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">FER</text>',
            f'<text x="{chart_left + chart_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#111827">{escape(x_axis_label)}</text>',
            f'<text transform="translate(24 {chart_top + chart_height / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="16" font-family="Helvetica, Arial, sans-serif" fill="#111827">Error rate</text>',
        ]
    )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def generate_default_plots(
    output_dir: str | Path,
    *,
    trials: int = 2000,
    seed: int = 7,
    crossover_probability: float = 0.05,
    block_lengths: Sequence[int] = (8, 16, 32, 64, 128),
    code_rate: float = 0.5,
    rate_block_length: int = 64,
    message_lengths: Sequence[int] = (8, 16, 24, 32, 40),
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    vs_n_points = sweep_block_lengths(
        block_lengths=block_lengths,
        code_rate=code_rate,
        crossover_probability=crossover_probability,
        trials=trials,
        seed=seed,
    )
    vs_rate_points = sweep_code_rates(
        block_length=rate_block_length,
        message_lengths=message_lengths,
        crossover_probability=crossover_probability,
        trials=trials,
        seed=seed + 1000,
    )

    vs_n_csv = output_root / "ber_fer_vs_n.csv"
    vs_n_svg = output_root / "ber_fer_vs_n.svg"
    vs_rate_csv = output_root / "ber_fer_vs_code_rate.csv"
    vs_rate_svg = output_root / "ber_fer_vs_code_rate.svg"

    save_curve_csv(vs_n_csv, vs_n_points)
    save_curve_csv(vs_rate_csv, vs_rate_points)

    save_dual_metric_svg(
        vs_n_svg,
        vs_n_points,
        title="BER/FER vs Block Length N",
        subtitle=(
            f"BSC p={crossover_probability}, trials={trials}, "
            f"target code rate={code_rate:.3f}"
        ),
        x_axis_label="Block length N (log2 spacing)",
        x_tick_formatter=lambda point: point.label,
        x_transform=lambda value: math.log2(value),
    )
    save_dual_metric_svg(
        vs_rate_svg,
        vs_rate_points,
        title="BER/FER vs Code Rate",
        subtitle=(
            f"BSC p={crossover_probability}, trials={trials}, "
            f"N={rate_block_length}"
        ),
        x_axis_label="Code rate K/N",
        x_tick_formatter=lambda point: f"{point.code_rate:.3f}",
    )

    return {
        "vs_n_csv": vs_n_csv,
        "vs_n_svg": vs_n_svg,
        "vs_rate_csv": vs_rate_csv,
        "vs_rate_svg": vs_rate_svg,
    }


def _nice_upper(value: float) -> float:
    if value <= 0.0:
        return 1.0

    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)

    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def _build_y_ticks(y_max: float) -> list[float]:
    return [y_max * index / 5 for index in range(6)]
