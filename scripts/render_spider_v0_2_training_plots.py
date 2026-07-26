#!/usr/bin/env python3
"""Render dependency-free Spider v0.2 research figures.

The figures are derived only from the certified aggregate and per-run history
files. SVG keeps the reporting path reproducible without introducing a plotting
dependency into the training environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any


SVG_WIDTH = 1120
SVG_HEIGHT = 680
PLOT_LEFT = 84
PLOT_RIGHT = 1080
PLOT_TOP = 72
PLOT_BOTTOM = 590
MODEL_COLOURS = {
    "recurrent": "#1769aa",
    "pooled": "#e56b2f",
}
SEED_DASHES = {
    1701: "",
    1802: "10 5",
    1903: "2 4",
}


@dataclass(frozen=True)
class HistorySeries:
    """One downsampled optimizer history used by the learning-curve figure."""

    experiment_id: str
    model: str
    seed: int
    steps: tuple[int, ...]
    losses: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.losses):
            raise ValueError("steps and losses must have the same length")
        if not self.steps:
            raise ValueError("history series cannot be empty")
        if any(step < 0 for step in self.steps):
            raise ValueError("history steps must be nonnegative")
        if tuple(sorted(self.steps)) != self.steps:
            raise ValueError("history steps must be sorted")
        if any(not math.isfinite(loss) for loss in self.losses):
            raise ValueError("history losses must be finite")


def _svg_document(body: str, *, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" '
        f'role="img" aria-label="{escape(title)}">\n'
        "<style>"
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#18212b}"
        ".title{font-size:24px;font-weight:700}"
        ".subtitle{font-size:14px;fill:#52606d}"
        ".axis{stroke:#51606f;stroke-width:1}"
        ".grid{stroke:#d9e1e8;stroke-width:1}"
        ".tick{font-size:12px;fill:#52606d}"
        ".legend{font-size:12px}"
        "</style>\n"
        f"<title>{escape(title)}</title>\n"
        '<rect width="1120" height="680" fill="#ffffff"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def _scale(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    if source_max <= source_min:
        return (target_min + target_max) / 2.0
    fraction = (value - source_min) / (source_max - source_min)
    return target_min + fraction * (target_max - target_min)


def _axis_grid(
    *,
    x_ticks: Sequence[tuple[float, str]],
    y_ticks: Sequence[tuple[float, str]],
    x_label: str,
    y_label: str,
) -> str:
    parts = [
        f'<line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" '
        f'x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}"/>',
        f'<line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_TOP}" '
        f'x2="{PLOT_LEFT}" y2="{PLOT_BOTTOM}"/>',
    ]
    for x, label in x_ticks:
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{PLOT_TOP}" '
                f'x2="{x:.2f}" y2="{PLOT_BOTTOM}"/>',
                f'<text class="tick" x="{x:.2f}" y="{PLOT_BOTTOM + 22}" '
                f'text-anchor="middle">{escape(label)}</text>',
            ]
        )
    for y, label in y_ticks:
        parts.extend(
            [
                f'<line class="grid" x1="{PLOT_LEFT}" y1="{y:.2f}" '
                f'x2="{PLOT_RIGHT}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{PLOT_LEFT - 10}" y="{y + 4:.2f}" '
                f'text-anchor="end">{escape(label)}</text>',
            ]
        )
    parts.extend(
        [
            f'<text class="tick" x="{(PLOT_LEFT + PLOT_RIGHT) / 2:.2f}" '
            f'y="{PLOT_BOTTOM + 50}" text-anchor="middle">'
            f"{escape(x_label)}</text>",
            f'<text class="tick" transform="translate(24 '
            f'{(PLOT_TOP + PLOT_BOTTOM) / 2:.2f}) rotate(-90)" '
            f'text-anchor="middle">{escape(y_label)}</text>',
        ]
    )
    return "\n".join(parts)


def render_learning_curves_svg(
    series: Sequence[HistorySeries],
) -> str:
    """Render loss trajectories for the exact paired experiment matrix."""

    if not series:
        raise ValueError("at least one history series is required")
    max_step = max(item.steps[-1] for item in series)
    max_loss = max(loss for item in series for loss in item.losses)
    upper_loss = max(1.0, math.ceil(max_loss * 2.0) / 2.0)
    x_ticks = tuple(
        (
            _scale(step, 0, max_step, PLOT_LEFT, PLOT_RIGHT),
            str(step),
        )
        for step in range(0, max_step + 1, max(1, max_step // 6))
    )
    if x_ticks[-1][1] != str(max_step):
        x_ticks = (*x_ticks, (float(PLOT_RIGHT), str(max_step)))
    y_ticks = tuple(
        (
            _scale(value, 0, upper_loss, PLOT_BOTTOM, PLOT_TOP),
            f"{value:.1f}",
        )
        for value in (
            upper_loss * index / 5.0 for index in range(6)
        )
    )
    parts = [
        '<text class="title" x="56" y="38">Training loss</text>',
        '<text class="subtitle" x="56" y="59">'
        "Frozen 6,000-step recurrence-necessity matrix</text>",
        _axis_grid(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_label="Optimizer step",
            y_label="Logged training loss",
        ),
    ]
    for index, item in enumerate(
        sorted(series, key=lambda value: (value.model, value.seed))
    ):
        colour = MODEL_COLOURS.get(item.model, "#59636e")
        dash = SEED_DASHES.get(item.seed, "")
        points = " ".join(
            f"{_scale(step, 0, max_step, PLOT_LEFT, PLOT_RIGHT):.2f},"
            f"{_scale(loss, 0, upper_loss, PLOT_BOTTOM, PLOT_TOP):.2f}"
            for step, loss in zip(item.steps, item.losses, strict=True)
        )
        dash_attribute = (
            f' stroke-dasharray="{dash}"' if dash else ""
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="2.2"{dash_attribute}/>'
        )
        legend_x = 660 + (index % 2) * 215
        legend_y = 92 + (index // 2) * 22
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y - 4}" '
                f'x2="{legend_x + 32}" y2="{legend_y - 4}" '
                f'stroke="{colour}" stroke-width="2.2"'
                f"{dash_attribute}/>",
                f'<text class="legend" x="{legend_x + 39}" '
                f'y="{legend_y}">{escape(item.experiment_id)}</text>',
            ]
        )
    return _svg_document(
        "\n".join(parts),
        title="Spider v0.2 training-loss curves",
    )


def _require_mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _require_float(value: object, *, field: str) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def render_structural_comparison_svg(
    summary: Mapping[str, Any],
) -> str:
    """Render paired structural success and direct recurrent-state ablations."""

    paired = _require_mapping(summary.get("paired"), field="paired")
    rows = _require_mapping(paired.get("by_seed"), field="paired.by_seed")
    if not rows:
        raise ValueError("paired.by_seed cannot be empty")
    state_use = _require_mapping(
        summary.get("state_use"),
        field="state_use",
    )
    interventions = _require_mapping(
        state_use.get("by_intervention"),
        field="state_use.by_intervention",
    )
    seeds = sorted(rows, key=int)
    chart_left = 78
    chart_right = 570
    chart_top = 112
    chart_bottom = 540
    chart_width = chart_right - chart_left
    max_success = 1.0
    parts = [
        '<text class="title" x="56" y="38">'
        "Fixed-horizon structural success</text>",
        '<text class="subtitle" x="56" y="59">'
        "Paired architecture comparison and direct state-use degradation"
        "</text>",
        '<text x="78" y="92" font-size="16" font-weight="650">'
        "Architecture comparison</text>",
        '<text x="650" y="92" font-size="16" font-weight="650">'
        "Direct state-use degradation</text>",
    ]
    for index in range(6):
        value = index / 5.0
        y = _scale(
            value,
            0,
            max_success,
            chart_bottom,
            chart_top,
        )
        parts.extend(
            [
                f'<line class="grid" x1="{chart_left}" y1="{y:.2f}" '
                f'x2="{chart_right}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{chart_left - 9}" y="{y + 4:.2f}" '
                f'text-anchor="end">{value:.1f}</text>',
            ]
        )
    group_width = chart_width / len(seeds)
    bar_width = min(42.0, group_width / 3.2)
    for seed_index, seed in enumerate(seeds):
        row = _require_mapping(rows[seed], field=f"paired.by_seed.{seed}")
        centre = chart_left + group_width * (seed_index + 0.5)
        for model_index, model in enumerate(("recurrent", "pooled")):
            value = _require_float(
                row.get(f"{model}_structural"),
                field=f"{seed}.{model}_structural",
            )
            height = _scale(value, 0, 1, 0, chart_bottom - chart_top)
            x = centre + (model_index - 0.5) * bar_width
            y = chart_bottom - height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{height:.2f}" fill="{MODEL_COLOURS[model]}"/>'
            )
        parts.append(
            f'<text class="tick" x="{centre:.2f}" y="{chart_bottom + 22}" '
            f'text-anchor="middle">{escape(seed)}</text>'
        )
    parts.extend(
        [
            '<rect x="164" y="575" width="14" height="14" '
            f'fill="{MODEL_COLOURS["recurrent"]}"/>',
            '<text class="legend" x="184" y="587">recurrent</text>',
            '<rect x="286" y="575" width="14" height="14" '
            f'fill="{MODEL_COLOURS["pooled"]}"/>',
            '<text class="legend" x="306" y="587">pooled</text>',
        ]
    )
    labels = [
        ("reset", "reset"),
        ("detach", "detach"),
        ("shuffle", "shuffle"),
        ("pooled_current_node", "pooled current node"),
    ]
    ablation_left = 650
    ablation_right = 1058
    for index, (name, label) in enumerate(labels):
        item = _require_mapping(
            interventions.get(name),
            field=f"state_use.by_intervention.{name}",
        )
        degradation = _require_float(
            item.get("mean_degradation"),
            field=f"{name}.mean_degradation",
        )
        y = 142 + index * 94
        width = _scale(
            max(0.0, degradation),
            0,
            1,
            0,
            ablation_right - ablation_left,
        )
        parts.extend(
            [
                f'<text class="tick" x="{ablation_left}" y="{y}">'
                f"{escape(label)}</text>",
                f'<rect x="{ablation_left}" y="{y + 13}" '
                f'width="{ablation_right - ablation_left}" height="28" '
                'fill="#edf1f5"/>',
                f'<rect x="{ablation_left}" y="{y + 13}" '
                f'width="{width:.2f}" height="28" fill="#6f42c1"/>',
                f'<text class="tick" x="{ablation_left + width + 7:.2f}" '
                f'y="{y + 33}">{degradation:+.3f}</text>',
            ]
        )
    return _svg_document(
        "\n".join(parts),
        title="Spider v0.2 fixed-horizon structural comparison",
    )


def load_history_series(run_root: Path) -> tuple[HistorySeries, ...]:
    """Load the exact optimizer histories found below an isolated-run root."""

    result: list[HistorySeries] = []
    for history_path in sorted(run_root.glob("*/*/run/history.jsonl")):
        run_directory = history_path.parents[1]
        record = json.loads(
            (run_directory / "experiment_record.json").read_text()
        )
        experiment_id = str(record["experiment_id"])
        by_step: dict[int, float] = {}
        for line in history_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            by_step[int(item["step"])] = float(item["loss"])
        steps = tuple(sorted(by_step))
        result.append(
            HistorySeries(
                experiment_id=experiment_id,
                model=str(record["model"]),
                seed=int(record["seed"]),
                steps=steps,
                losses=tuple(by_step[step] for step in steps),
            )
        )
    if not result:
        raise FileNotFoundError(f"no run histories found below {run_root}")
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "artifacts/spider_v0_2/training/TRAINING_SUMMARY.json"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/spider_v0_2/training/isolated"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/spider_v0_2/plots"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    histories = load_history_series(args.run_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "learning_curves.svg").write_text(
        render_learning_curves_svg(histories)
    )
    (args.output_root / "structural_comparison.svg").write_text(
        render_structural_comparison_svg(summary)
    )
    print(
        json.dumps(
            {
                "history_count": len(histories),
                "outputs": [
                    str(args.output_root / "learning_curves.svg"),
                    str(args.output_root / "structural_comparison.svg"),
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
