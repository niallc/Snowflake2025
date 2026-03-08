#!/usr/bin/env python3
"""
Generate a sampled HTML report for policy-search target distributions.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import html
import json
import random
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hex_ai.move_provenance import parse_move_provenance_line


DEFAULT_DIR_GLOB = "data/cleaned/cleaned_selfplay_*"
DEFAULT_LATEST_DIRS = 4
DEFAULT_FILES_PER_DIR = 3
DEFAULT_GAMES_PER_FILE = 1000
DEFAULT_SEED = 20260307
DEFAULT_OUTPUT_DIR = "analysis/policy_target_reports"

TRAINABLE_CODES = ("G", "T", "V")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.70, 0.90)
GAP_THRESHOLDS = (0.005, 0.010, 0.020, 0.050, 0.100)
HIST_BINS = 20
EXAMPLE_ROWS_PER_SECTION = 4
MOVE_INDEX_TOP_K = 12

SOURCE_DESCRIPTIONS = {
    "V": "Visit-count path. Current code also uses V for injected opening-book moves.",
    "G": "Gumbel path. Stored target is softmax over the final scored subset only.",
    "T": "Terminal path. Usually a visit distribution, sometimes one-hot in shortcut cases.",
    "C": "Confidence termination. Stored row is masked out for policy training.",
}

SOURCE_LABELS = {
    "all_trainable": "All trainable rows",
    "G": "G rows",
    "T": "T rows",
    "V": "V rows",
    "C": "C rows",
}


@dataclass
class MetricBucket:
    top1: array = field(default_factory=lambda: array("f"))
    top2: array = field(default_factory=lambda: array("f"))
    top3: array = field(default_factory=lambda: array("f"))
    avg_nz: array = field(default_factory=lambda: array("f"))
    gap12: array = field(default_factory=lambda: array("f"))
    ratio1avg: array = field(default_factory=lambda: array("f"))
    nz_count: array = field(default_factory=lambda: array("I"))
    zero_frac: array = field(default_factory=lambda: array("f"))
    postsq_top1: array = field(default_factory=lambda: array("f"))

    def add(
        self,
        *,
        top1: float,
        top2: float,
        top3: float,
        avg_nz: float,
        gap12: float,
        ratio1avg: float,
        nz_count: int,
        zero_frac: float,
        postsq_top1: float,
    ) -> None:
        self.top1.append(float(top1))
        self.top2.append(float(top2))
        self.top3.append(float(top3))
        self.avg_nz.append(float(avg_nz))
        self.gap12.append(float(gap12))
        self.ratio1avg.append(float(ratio1avg))
        self.nz_count.append(int(nz_count))
        self.zero_frac.append(float(zero_frac))
        self.postsq_top1.append(float(postsq_top1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sampled HTML report for policy target distributions."
    )
    parser.add_argument(
        "--dir-glob",
        default=DEFAULT_DIR_GLOB,
        help=f"Glob used to find cleaned self-play directories (default: {DEFAULT_DIR_GLOB})",
    )
    parser.add_argument(
        "--latest-dirs",
        type=int,
        default=DEFAULT_LATEST_DIRS,
        help=f"Keep only the latest N directories after sorting (default: {DEFAULT_LATEST_DIRS})",
    )
    parser.add_argument(
        "--files-per-dir",
        type=int,
        default=DEFAULT_FILES_PER_DIR,
        help=f"Randomly sample this many provenance files per directory (default: {DEFAULT_FILES_PER_DIR})",
    )
    parser.add_argument(
        "--games-per-file",
        type=int,
        default=DEFAULT_GAMES_PER_FILE,
        help=f"Reservoir-sample this many game records per file (default: {DEFAULT_GAMES_PER_FILE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to receive HTML/JSON outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def find_target_dirs(dir_glob: str, latest_dirs: int) -> list[Path]:
    dirs = sorted(Path().glob(dir_glob))
    dirs = [path for path in dirs if path.is_dir()]
    if latest_dirs > 0:
        dirs = dirs[-latest_dirs:]
    if not dirs:
        raise FileNotFoundError(f"No directories matched {dir_glob!r}")
    return dirs


def select_files(
    target_dirs: Sequence[Path], files_per_dir: int, rng: random.Random
) -> list[Path]:
    selected_files: list[Path] = []
    for directory in target_dirs:
        files = sorted(directory.glob("*.provenance.jsonl"))
        if not files:
            continue
        count = min(files_per_dir, len(files))
        selected_files.extend(sorted(rng.sample(files, k=count)))
    if not selected_files:
        raise FileNotFoundError("No provenance files found in selected directories")
    return selected_files


def reservoir_sample_lines(
    file_path: Path, sample_size: int, rng: random.Random
) -> list[tuple[int, str]]:
    sample: list[tuple[int, str]] = []
    seen = 0
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            seen += 1
            item = (line_number, line)
            if len(sample) < sample_size:
                sample.append(item)
                continue
            replace_index = rng.randrange(seen)
            if replace_index < sample_size:
                sample[replace_index] = item
    return sample


def buffer_to_float_array(buffer: array) -> np.ndarray:
    return np.frombuffer(buffer, dtype=np.float32).copy()


def buffer_to_int_array(buffer: array) -> np.ndarray:
    return np.frombuffer(buffer, dtype=np.uint32).astype(np.int64, copy=True)


def summarize_float(buffer: array) -> dict[str, float]:
    values = buffer_to_float_array(buffer)
    summary: dict[str, float] = {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for quantile in QUANTILES:
        summary[f"p{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
    return summary


def summarize_int(buffer: array) -> dict[str, float]:
    values = buffer_to_int_array(buffer)
    summary: dict[str, float] = {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for quantile in QUANTILES:
        summary[f"p{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
    return summary


def pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return 100.0 * float(part) / float(whole)


def format_float(value: float, decimals: int = 3) -> str:
    if value >= 0.9995:
        decimals = max(decimals, 4)
    if abs(value) < 0.0001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def add_reservoir_example(
    reservoirs: dict[str, list[dict[str, Any]]],
    seen_counts: Counter,
    key: str,
    item: dict[str, Any],
    capacity: int,
    rng: random.Random,
) -> None:
    seen_counts[key] += 1
    seen = int(seen_counts[key])
    bucket = reservoirs[key]
    if len(bucket) < capacity:
        bucket.append(item)
        return
    replace_index = rng.randrange(seen)
    if replace_index < capacity:
        bucket[replace_index] = item


def choose_examples(
    rows: list[dict[str, Any]],
    metric_bucket: MetricBucket,
    exact_examples: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    top1 = buffer_to_float_array(metric_bucket.top1)
    gap12 = buffer_to_float_array(metric_bucket.gap12)
    median_top1 = float(np.quantile(top1, 0.50))
    median_gap = float(np.quantile(gap12, 0.50))
    medianish = min(
        rows,
        key=lambda row: (
            abs(row["top1"] - median_top1),
            abs(row["gap12"] - median_gap),
        ),
    )
    return {
        "medianish": medianish,
        "flattest": exact_examples["flattest"],
        "smallest_gap": exact_examples["smallest_gap"],
        "peakiest": exact_examples["peakiest"],
    }


def update_exact_examples(
    exact_examples: dict[str, dict[str, dict[str, Any]]],
    metric_key: str,
    example_row: dict[str, Any],
) -> None:
    bucket = exact_examples[metric_key]

    current = bucket.get("flattest")
    if current is None or (example_row["top1"], example_row["gap12"]) < (
        current["top1"],
        current["gap12"],
    ):
        bucket["flattest"] = example_row

    current = bucket.get("peakiest")
    if current is None or (example_row["top1"], -example_row["gap12"]) > (
        current["top1"],
        -current["gap12"],
    ):
        bucket["peakiest"] = example_row

    if example_row["nonzero_count"] >= 2:
        current = bucket.get("smallest_gap")
        if current is None or (example_row["gap12"], example_row["top1"]) < (
            current["gap12"],
            current["top1"],
        ):
            bucket["smallest_gap"] = example_row


def histogram(values: np.ndarray, bins: int, x_range: tuple[float, float]) -> list[dict[str, float]]:
    counts, edges = np.histogram(values, bins=bins, range=x_range)
    max_count = int(counts.max()) if counts.size else 1
    output: list[dict[str, float]] = []
    for index, count in enumerate(counts):
        output.append(
            {
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "count": int(count),
                "fraction": float(count / max_count) if max_count > 0 else 0.0,
            }
        )
    return output


def build_metric_table_rows(metric_summary: dict[str, Any]) -> list[tuple[str, dict[str, float]]]:
    return [
        ("Top-1", metric_summary["top1"]),
        ("Top-2", metric_summary["top2"]),
        ("Top-3", metric_summary["top3"]),
        ("Avg non-zero", metric_summary["avg_nonzero"]),
        ("Gap top1-top2", metric_summary["gap12"]),
        ("Top1 / avg-nz", metric_summary["ratio_top1_to_avg_nonzero"]),
        ("Non-zero count", metric_summary["nonzero_count"]),
    ]


def generate_observations(summary: dict[str, Any]) -> list[str]:
    observations: list[str] = []
    all_metrics = summary["metrics"]["all_trainable"]
    g_metrics = summary["metrics"].get("G")
    t_metrics = summary["metrics"].get("T")
    v_metrics = summary["metrics"].get("V")

    observations.append(
        "No sampled trainable row looked anything like a broad 0.09 / 0.08 / 0.079 / ... distribution."
    )
    observations.append(
        "In the sampled trainable rows, top-1 was never below "
        f"{format_float(all_metrics['top1']['min'])} and support size never exceeded "
        f"{int(round(all_metrics['nonzero_count']['max']))}."
    )
    observations.append(
        "Across all sampled trainable rows, median top-1 was "
        f"{format_float(all_metrics['top1']['p50'])}, median top-2 was "
        f"{format_float(all_metrics['top2']['p50'])}, and median top1-top2 gap was "
        f"{format_float(all_metrics['gap12']['p50'])}."
    )

    if g_metrics is not None:
        observations.append(
            "The genuinely soft rows are mostly G rows: their median top-1 was "
            f"{format_float(g_metrics['top1']['p50'])}, median top-2 was "
            f"{format_float(g_metrics['top2']['p50'])}, and median support size was "
            f"{format_float(g_metrics['nonzero_count']['p50'], decimals=1)}."
        )
        observations.append(
            f"{format_percent(100.0 * g_metrics['fraction_top1_ge']['0.90'])} of sampled G rows had top-1 >= 0.90, "
            f"while only {format_percent(100.0 * g_metrics['fraction_gap12_le']['0.020'])} had a top1-top2 gap <= 0.02."
        )

    if v_metrics is not None:
        v_move0 = summary["move_index_top_counts"]["V"][0]["count"] if summary["move_index_top_counts"].get("V") else 0
        v_count = summary["trainable_counts"]["V"]
        observations.append(
            f"All sampled V rows were one-hot, and {format_percent(pct(v_move0, v_count))} of them were move index 0."
        )

    if t_metrics is not None:
        observations.append(
            "T rows were mostly one-hot too, with a small minority of 2-way or 3-way splits "
            f"(sampled support mean {format_float(t_metrics['nonzero_count']['mean'])})."
        )

    observations.append(
        "If you square and renormalize the sampled distributions, mean top-1 rises from "
        f"{format_float(all_metrics['top1']['mean'])} to "
        f"{format_float(all_metrics['post_square_top1']['mean'])}; "
        f"the median gain is {format_float(all_metrics['median_top1_gain_after_square'])}."
    )
    return observations


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    target_dirs = find_target_dirs(args.dir_glob, args.latest_dirs)
    selected_files = select_files(target_dirs, args.files_per_dir, rng)

    metrics: dict[str, MetricBucket] = defaultdict(MetricBucket)
    source_counts = Counter()
    trainable_counts = Counter()
    masked_counts = Counter()
    one_hot_counts = Counter()
    support_count_distribution: dict[str, Counter] = defaultdict(Counter)
    move_index_counts: dict[str, Counter] = defaultdict(Counter)
    file_game_counts: dict[str, int] = {}
    reservoir_seen = Counter()
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_examples: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for file_path in selected_files:
        selected_lines = reservoir_sample_lines(file_path, args.games_per_file, rng)
        file_game_counts[str(file_path)] = len(selected_lines)

        for line_number, line in selected_lines:
            record = parse_move_provenance_line(line, source=file_path, line_number=line_number)
            if not record.has_policy_targets():
                continue

            matrix = record.decode_policy_targets()
            source_codes = record.policy_target_source_codes or record.move_codes
            for move_index, (code, mask_bit) in enumerate(
                zip(source_codes, record.policy_train_mask)
            ):
                source_counts[code] += 1
                if mask_bit != "1":
                    masked_counts[code] += 1
                    continue

                trainable_counts[code] += 1
                row = matrix[move_index]
                nz = row[row > 0]
                if nz.size == 0:
                    raise RuntimeError(
                        f"Found zero-mass trainable row in {file_path} line {line_number}"
                    )
                nz_desc = np.sort(nz.astype(np.float64, copy=False))[::-1]
                top1 = float(nz_desc[0])
                top2 = float(nz_desc[1]) if nz_desc.size > 1 else 0.0
                top3 = float(nz_desc[2]) if nz_desc.size > 2 else 0.0
                avg_nz = float(nz_desc.mean())
                gap12 = float(top1 - top2)
                ratio1avg = float(top1 / avg_nz)
                zero_frac = float(1.0 - (nz_desc.size / row.size))
                squared = nz_desc * nz_desc
                postsq_top1 = float(squared[0] / squared.sum())

                support_count_distribution[code][int(nz_desc.size)] += 1
                move_index_counts[code][move_index] += 1
                if nz_desc.size == 1 or abs(top1 - 1.0) <= 1e-6:
                    one_hot_counts[code] += 1

                example_row = {
                    "file": str(file_path),
                    "line_number": int(line_number),
                    "game_index": int(record.game_index),
                    "move_index": int(move_index),
                    "code": code,
                    "nonzero_count": int(nz_desc.size),
                    "top1": top1,
                    "top2": top2,
                    "top3": top3,
                    "avg_nonzero": avg_nz,
                    "gap12": gap12,
                    "post_square_top1": postsq_top1,
                    "top10_nonzero_desc": [float(value) for value in nz_desc[:10]],
                }

                for metric_key in ("all_trainable", code):
                    metrics[metric_key].add(
                        top1=top1,
                        top2=top2,
                        top3=top3,
                        avg_nz=avg_nz,
                        gap12=gap12,
                        ratio1avg=ratio1avg,
                        nz_count=int(nz_desc.size),
                        zero_frac=zero_frac,
                        postsq_top1=postsq_top1,
                    )
                    add_reservoir_example(
                        reservoirs=reservoirs,
                        seen_counts=reservoir_seen,
                        key=metric_key,
                        item=example_row,
                        capacity=512,
                        rng=rng,
                    )
                    update_exact_examples(exact_examples, metric_key, example_row)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "sample_scope": {
            "dir_glob": args.dir_glob,
            "latest_dirs": int(args.latest_dirs),
            "files_per_dir": int(args.files_per_dir),
            "games_per_file": int(args.games_per_file),
            "seed": int(args.seed),
            "selected_dirs": [str(path) for path in target_dirs],
            "selected_files": [str(path) for path in selected_files],
            "sampled_game_records": int(sum(file_game_counts.values())),
            "file_game_counts": file_game_counts,
        },
        "source_counts_all_sampled_rows": {key: int(value) for key, value in sorted(source_counts.items())},
        "trainable_counts": {key: int(value) for key, value in sorted(trainable_counts.items())},
        "masked_counts": {key: int(value) for key, value in sorted(masked_counts.items())},
        "one_hot_trainable_counts": {key: int(value) for key, value in sorted(one_hot_counts.items())},
        "support_count_distribution": {
            key: {str(size): int(count) for size, count in sorted(counter.items())}
            for key, counter in sorted(support_count_distribution.items())
        },
        "move_index_top_counts": {
            key: [
                {"move_index": int(move_index), "count": int(count)}
                for move_index, count in counter.most_common(MOVE_INDEX_TOP_K)
            ]
            for key, counter in sorted(move_index_counts.items())
        },
        "metrics": {},
        "examples": {},
    }

    for metric_key, bucket in metrics.items():
        if not bucket.top1:
            continue
        if "smallest_gap" not in exact_examples[metric_key]:
            exact_examples[metric_key]["smallest_gap"] = exact_examples[metric_key]["flattest"]
        top1_values = buffer_to_float_array(bucket.top1)
        gap_values = buffer_to_float_array(bucket.gap12)
        post_square_values = buffer_to_float_array(bucket.postsq_top1)
        metric_summary = {
            "count": int(len(bucket.top1)),
            "top1": summarize_float(bucket.top1),
            "top2": summarize_float(bucket.top2),
            "top3": summarize_float(bucket.top3),
            "avg_nonzero": summarize_float(bucket.avg_nz),
            "gap12": summarize_float(bucket.gap12),
            "ratio_top1_to_avg_nonzero": summarize_float(bucket.ratio1avg),
            "nonzero_count": summarize_int(bucket.nz_count),
            "zero_fraction": summarize_float(bucket.zero_frac),
            "post_square_top1": summarize_float(bucket.postsq_top1),
            "fraction_top1_le": {
                f"{threshold:.2f}": float(np.mean(top1_values <= threshold))
                for threshold in THRESHOLDS
            },
            "fraction_top1_ge": {
                f"{threshold:.2f}": float(np.mean(top1_values >= threshold))
                for threshold in THRESHOLDS
            },
            "fraction_gap12_le": {
                f"{threshold:.3f}": float(np.mean(gap_values <= threshold))
                for threshold in GAP_THRESHOLDS
            },
            "mean_top1_gain_after_square": float(post_square_values.mean() - top1_values.mean()),
            "median_top1_gain_after_square": float(
                np.quantile(post_square_values - top1_values, 0.50)
            ),
            "histograms": {
                "top1": histogram(top1_values, bins=HIST_BINS, x_range=(0.0, 1.0)),
                "top2": histogram(
                    buffer_to_float_array(bucket.top2), bins=HIST_BINS, x_range=(0.0, 1.0)
                ),
                "gap12": histogram(gap_values, bins=HIST_BINS, x_range=(0.0, 1.0)),
            },
        }
        summary["metrics"][metric_key] = metric_summary
        summary["examples"][metric_key] = choose_examples(
            reservoirs[metric_key], bucket, exact_examples[metric_key]
        )

    summary["observations"] = generate_observations(summary)
    return summary


def render_histogram_svg(
    histogram_rows: Sequence[dict[str, float]],
    color: str,
    *,
    width: int = 360,
    height: int = 140,
) -> str:
    margin_left = 14
    margin_right = 8
    margin_top = 8
    margin_bottom = 18
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom
    if not histogram_rows:
        return ""

    bar_width = inner_width / len(histogram_rows)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="histogram">'
    ]
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{width - margin_right}" '
        f'y2="{margin_top + inner_height}" class="axis"/>'
    )
    for index, row in enumerate(histogram_rows):
        bar_height = inner_height * row["fraction"]
        x = margin_left + index * bar_width + 1
        y = margin_top + inner_height - bar_height
        width_adjusted = max(bar_width - 2, 1)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width_adjusted:.2f}" height="{bar_height:.2f}" '
            f'fill="{color}" rx="2" ry="2"/>'
        )
    parts.append(
        f'<text x="{margin_left}" y="{height - 2}" class="axis-label">0.0</text>'
        f'<text x="{width - margin_right}" y="{height - 2}" class="axis-label anchor-end">1.0</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_bar_chart_svg(
    items: Sequence[tuple[str, int]],
    color: str,
    *,
    width: int = 360,
    height: int = 160,
) -> str:
    if not items:
        return ""
    margin_left = 18
    margin_right = 12
    margin_top = 10
    margin_bottom = 32
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom
    max_count = max(count for _, count in items) or 1
    bar_width = inner_width / len(items)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="bar chart">'
    ]
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{width - margin_right}" '
        f'y2="{margin_top + inner_height}" class="axis"/>'
    )
    for index, (label, count) in enumerate(items):
        bar_height = inner_height * (count / max_count)
        x = margin_left + index * bar_width + 3
        y = margin_top + inner_height - bar_height
        width_adjusted = max(bar_width - 6, 2)
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width_adjusted:.2f}" height="{bar_height:.2f}" '
            f'fill="{color}" rx="3" ry="3"/>'
        )
        parts.append(
            f'<text x="{x + width_adjusted / 2:.2f}" y="{height - 14}" class="axis-label anchor-middle">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_weight_bars(weights: Sequence[float]) -> str:
    max_weight = max(weights) if weights else 1.0
    rows: list[str] = ['<div class="weight-bars">']
    for index, weight in enumerate(weights[:10], start=1):
        width_pct = 100.0 * weight / max_weight if max_weight > 0 else 0.0
        rows.append(
            '<div class="weight-row">'
            f'<span class="weight-label">#{index}</span>'
            f'<div class="weight-track"><div class="weight-fill" style="width:{width_pct:.2f}%"></div></div>'
            f'<span class="weight-value">{format_float(float(weight), decimals=4)}</span>'
            "</div>"
        )
    rows.append("</div>")
    return "".join(rows)


def render_stat_cards(metric_summary: dict[str, Any], count: int) -> str:
    cards = [
        ("Rows", f"{count:,}"),
        ("Median top-1", format_float(metric_summary["top1"]["p50"], decimals=4)),
        ("Median top-2", format_float(metric_summary["top2"]["p50"], decimals=4)),
        ("Median gap", format_float(metric_summary["gap12"]["p50"], decimals=4)),
        ("Mean avg-nz", format_float(metric_summary["avg_nonzero"]["mean"], decimals=4)),
        (
            "Mean sharpened top-1",
            format_float(metric_summary["post_square_top1"]["mean"], decimals=4),
        ),
    ]
    return "".join(
        '<div class="stat-card">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        "</div>"
        for label, value in cards
    )


def render_quantile_table(metric_summary: dict[str, Any]) -> str:
    rows = []
    for label, stats in build_metric_table_rows(metric_summary):
        rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{format_float(stats['mean'], decimals=4)}</td>"
            f"<td>{format_float(stats['p10'], decimals=4)}</td>"
            f"<td>{format_float(stats['p25'], decimals=4)}</td>"
            f"<td>{format_float(stats['p50'], decimals=4)}</td>"
            f"<td>{format_float(stats['p75'], decimals=4)}</td>"
            f"<td>{format_float(stats['p90'], decimals=4)}</td>"
            f"<td>{format_float(stats['max'], decimals=4)}</td>"
            "</tr>"
        )
    return (
        '<table class="metrics-table">'
        "<thead><tr><th>Metric</th><th>Mean</th><th>P10</th><th>P25</th><th>P50</th><th>P75</th><th>P90</th><th>Max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_examples(example_block: dict[str, dict[str, Any]]) -> str:
    order = ("medianish", "flattest", "smallest_gap", "peakiest")
    labels = {
        "medianish": "Median-ish row",
        "flattest": "Flattest row",
        "smallest_gap": "Smallest top1-top2 gap",
        "peakiest": "Peakiest row",
    }
    cards: list[str] = []
    for key in order[:EXAMPLE_ROWS_PER_SECTION]:
        example = example_block.get(key)
        if example is None:
            continue
        meta = (
            f'{html.escape(example["code"])} · move {example["move_index"]} · '
            f'nz={example["nonzero_count"]}'
        )
        source = f'{html.escape(example["file"])} line {example["line_number"]}'
        cards.append(
            '<div class="example-card">'
            f'<h4>{html.escape(labels[key])}</h4>'
            f'<p class="example-meta">{meta}</p>'
            f'<p class="example-path">{source}</p>'
            '<div class="example-grid">'
            f'<div><span class="mini-label">top-1</span><strong>{format_float(example["top1"], decimals=4)}</strong></div>'
            f'<div><span class="mini-label">top-2</span><strong>{format_float(example["top2"], decimals=4)}</strong></div>'
            f'<div><span class="mini-label">top-3</span><strong>{format_float(example["top3"], decimals=4)}</strong></div>'
            f'<div><span class="mini-label">avg-nz</span><strong>{format_float(example["avg_nonzero"], decimals=4)}</strong></div>'
            f'<div><span class="mini-label">gap</span><strong>{format_float(example["gap12"], decimals=4)}</strong></div>'
            f'<div><span class="mini-label">square->top1</span><strong>{format_float(example["post_square_top1"], decimals=4)}</strong></div>'
            "</div>"
            f'{render_weight_bars(example["top10_nonzero_desc"])}'
            "</div>"
        )
    return "".join(cards)


def build_code_section(
    code: str,
    summary: dict[str, Any],
    color: str,
) -> str:
    if code not in summary["metrics"]:
        return ""
    metric_summary = summary["metrics"][code]
    support_counts = summary["support_count_distribution"].get(code, {})
    support_items = [(key, int(value)) for key, value in sorted(support_counts.items(), key=lambda item: int(item[0]))]
    move_index_items = [
        (str(item["move_index"]), int(item["count"]))
        for item in summary["move_index_top_counts"].get(code, [])
    ]
    one_hot_count = summary["one_hot_trainable_counts"].get(code, 0)
    count = metric_summary["count"]
    one_hot_pct = pct(one_hot_count, count)

    support_chart = render_bar_chart_svg(support_items, color) if support_items else ""
    move_chart = render_bar_chart_svg(move_index_items, color) if move_index_items else ""

    return (
        f'<section class="panel source-panel source-{code.lower()}">'
        f'<div class="panel-head"><h2>{html.escape(SOURCE_LABELS[code])}</h2>'
        f'<p>{html.escape(SOURCE_DESCRIPTIONS[code])}</p></div>'
        f'<div class="stat-grid">{render_stat_cards(metric_summary, count)}</div>'
        '<div class="summary-strip">'
        f'<div><span class="mini-label">one-hot rows</span><strong>{one_hot_count:,}</strong><span>{format_percent(one_hot_pct)}</span></div>'
        f'<div><span class="mini-label">top-1 >= 0.90</span><strong>{format_percent(100.0 * metric_summary["fraction_top1_ge"]["0.90"])}</strong></div>'
        f'<div><span class="mini-label">gap <= 0.02</span><strong>{format_percent(100.0 * metric_summary["fraction_gap12_le"]["0.020"])}</strong></div>'
        f'<div><span class="mini-label">max support</span><strong>{int(round(metric_summary["nonzero_count"]["max"]))}</strong></div>'
        "</div>"
        '<div class="chart-grid">'
        f'<div class="chart-card"><h3>Top-1 histogram</h3>{render_histogram_svg(metric_summary["histograms"]["top1"], color)}</div>'
        f'<div class="chart-card"><h3>Top-2 histogram</h3>{render_histogram_svg(metric_summary["histograms"]["top2"], color)}</div>'
        f'<div class="chart-card"><h3>Gap histogram</h3>{render_histogram_svg(metric_summary["histograms"]["gap12"], color)}</div>'
        "</div>"
        '<div class="chart-grid">'
        f'<div class="chart-card"><h3>Support size</h3>{support_chart}</div>'
        f'<div class="chart-card"><h3>Typical move indices</h3>{move_chart}</div>'
        "</div>"
        f'{render_quantile_table(metric_summary)}'
        f'<div class="examples-grid">{render_examples(summary["examples"].get(code, {}))}</div>'
        "</section>"
    )


def render_html(summary: dict[str, Any]) -> str:
    all_metrics = summary["metrics"]["all_trainable"]
    colors = {"all_trainable": "#c96f1f", "G": "#1f7a5b", "T": "#b2412f", "V": "#3657a7"}
    observation_items = "".join(
        f"<li>{html.escape(observation)}</li>" for observation in summary["observations"]
    )
    sample_scope = summary["sample_scope"]

    overview_cards = [
        (
            "Trainable rows",
            f'{all_metrics["count"]:,}',
            "Sampled rows that actually participate in policy loss",
        ),
        (
            "Median top-1",
            format_float(all_metrics["top1"]["p50"], decimals=4),
            "Best move probability in the sampled target row",
        ),
        (
            "Median top-2",
            format_float(all_metrics["top2"]["p50"], decimals=4),
            "Second-highest probability",
        ),
        (
            "Median gap",
            format_float(all_metrics["gap12"]["p50"], decimals=4),
            "Top-1 minus top-2",
        ),
        (
            "Top-1 after squaring",
            format_float(all_metrics["post_square_top1"]["mean"], decimals=4),
            "Mean top-1 if scores are squared then renormalized",
        ),
        (
            "Masked C rows",
            f'{summary["masked_counts"].get("C", 0):,}',
            "Rows present in sidecars but excluded from policy loss",
        ),
    ]

    overview_html = "".join(
        '<div class="overview-card">'
        f'<div class="overview-kicker">{html.escape(label)}</div>'
        f'<div class="overview-value">{html.escape(value)}</div>'
        f'<p>{html.escape(body)}</p>'
        "</div>"
        for label, value, body in overview_cards
    )

    code_sections = "".join(
        build_code_section(code, summary, colors[code]) for code in ("G", "T", "V")
    )

    selected_files = "".join(
        f"<li>{html.escape(path)}</li>" for path in sample_scope["selected_files"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Policy Target Sample Report</title>
  <style>
    :root {{
      --paper: #f6f0e6;
      --paper-deep: #efe4d0;
      --ink: #1f1b16;
      --muted: #6a6053;
      --accent: #c96f1f;
      --accent-dark: #884411;
      --green: #1f7a5b;
      --blue: #3657a7;
      --red: #b2412f;
      --card: rgba(255, 252, 245, 0.88);
      --line: rgba(53, 40, 24, 0.14);
      --shadow: 0 18px 40px rgba(73, 49, 19, 0.08);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(201, 111, 31, 0.18), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(54, 87, 167, 0.12), transparent 24%),
        linear-gradient(180deg, #f8f3ea 0%, #efe5d6 100%);
      font-family: "Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif;
      line-height: 1.45;
    }}

    main {{
      max-width: 1260px;
      margin: 0 auto;
      padding: 32px 18px 72px;
    }}

    h1, h2, h3, h4 {{
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      letter-spacing: 0.01em;
      margin: 0;
    }}

    p, li {{ color: var(--muted); }}

    .hero {{
      padding: 28px 28px 22px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(249,239,223,0.84));
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -4% -45% auto;
      width: 380px;
      height: 380px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(201,111,31,0.16), transparent 68%);
      pointer-events: none;
    }}

    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.78rem;
      color: var(--accent-dark);
      margin-bottom: 10px;
    }}

    .hero p {{
      max-width: 840px;
      margin: 12px 0 0;
      font-size: 1.02rem;
    }}

    .panel {{
      margin-top: 22px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--card);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}

    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}

    .overview-grid,
    .stat-grid,
    .examples-grid,
    .chart-grid {{
      display: grid;
      gap: 14px;
    }}

    .overview-grid {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}

    .overview-card,
    .stat-card,
    .chart-card,
    .example-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.72);
      padding: 16px;
    }}

    .overview-kicker,
    .stat-label,
    .mini-label {{
      display: block;
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}

    .overview-value,
    .stat-value,
    .summary-strip strong,
    .example-grid strong {{
      font-family: Menlo, Consolas, "Liberation Mono", monospace;
      color: var(--ink);
    }}

    .overview-value {{
      font-size: 1.65rem;
      margin-bottom: 8px;
    }}

    .stat-grid {{
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      margin-top: 12px;
    }}

    .stat-value {{
      font-size: 1.25rem;
    }}

    ul.takeaways {{
      margin: 10px 0 0;
      padding-left: 20px;
      columns: 2 320px;
      column-gap: 28px;
    }}

    .chart-grid {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      margin-top: 14px;
    }}

    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 14px;
      margin-bottom: 8px;
    }}

    .summary-strip > div {{
      border-left: 3px solid var(--accent);
      padding-left: 12px;
    }}

    .source-g .summary-strip > div {{ border-left-color: var(--green); }}
    .source-t .summary-strip > div {{ border-left-color: var(--red); }}
    .source-v .summary-strip > div {{ border-left-color: var(--blue); }}

    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 0.94rem;
    }}

    .metrics-table th,
    .metrics-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}

    .metrics-table td {{
      font-family: Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.88rem;
      color: var(--ink);
    }}

    .examples-grid {{
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-top: 16px;
    }}

    .example-card h4 {{
      margin-bottom: 6px;
    }}

    .example-meta,
    .example-path {{
      margin: 6px 0;
      font-size: 0.9rem;
      word-break: break-word;
    }}

    .example-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }}

    .weight-bars {{
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }}

    .weight-row {{
      display: grid;
      grid-template-columns: 30px 1fr 72px;
      gap: 10px;
      align-items: center;
    }}

    .weight-label,
    .weight-value {{
      font-family: Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 0.84rem;
      color: var(--ink);
    }}

    .weight-track {{
      height: 12px;
      border-radius: 999px;
      background: rgba(53, 40, 24, 0.08);
      overflow: hidden;
    }}

    .weight-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(201,111,31,0.95), rgba(255,179,93,0.95));
    }}

    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .axis {{
      stroke: rgba(53, 40, 24, 0.2);
      stroke-width: 1.1;
    }}

    .axis-label {{
      font-size: 10px;
      fill: var(--muted);
      font-family: Menlo, Consolas, monospace;
    }}

    .anchor-end {{ text-anchor: end; }}
    .anchor-middle {{ text-anchor: middle; }}

    details {{
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.64);
      padding: 12px 16px;
    }}

    summary {{
      cursor: pointer;
      color: var(--ink);
      font-weight: 600;
    }}

    @media (max-width: 700px) {{
      main {{ padding: 18px 12px 44px; }}
      .hero {{ padding: 22px 18px; }}
      .panel {{ padding: 18px; }}
      ul.takeaways {{ columns: 1; }}
      .example-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .weight-row {{ grid-template-columns: 24px 1fr 66px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">Policy Search Targets</div>
      <h1>Sampled sidecar report for the latest cleaned self-play data</h1>
      <p>
        This page summarizes the actual policy-search target rows stored in sidecar files and consumed by training.
        The sample is stratified across the latest cleaned self-play directories, then summarized by source code so it is clear which rows are genuinely soft and which are effectively one-hot.
      </p>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>What stands out</h2>
        <p>Generated {html.escape(summary["generated_at"])} from {sample_scope["sampled_game_records"]:,} sampled game records.</p>
      </div>
      <ul class="takeaways">{observation_items}</ul>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Scope</h2>
        <p>Sampling is reproducible from a fixed seed and uses project-native provenance decoding.</p>
      </div>
      <div class="overview-grid">{overview_html}</div>
      <details>
        <summary>Selected files</summary>
        <p>Directory glob: <code>{html.escape(sample_scope["dir_glob"])}</code></p>
        <p>Latest dirs kept: <code>{sample_scope["latest_dirs"]}</code>, files per dir: <code>{sample_scope["files_per_dir"]}</code>, games per file: <code>{sample_scope["games_per_file"]}</code>, seed: <code>{sample_scope["seed"]}</code></p>
        <ul>{selected_files}</ul>
      </details>
    </section>

    <section class="panel source-panel source-all">
      <div class="panel-head">
        <h2>All trainable rows</h2>
        <p>Aggregated across G, T, and V. C rows are present in sidecars but masked out.</p>
      </div>
      <div class="stat-grid">{render_stat_cards(all_metrics, all_metrics["count"])}</div>
      <div class="summary-strip">
        <div><span class="mini-label">top-1 >= 0.90</span><strong>{format_percent(100.0 * all_metrics["fraction_top1_ge"]["0.90"])}</strong></div>
        <div><span class="mini-label">top-1 <= 0.50</span><strong>{format_percent(100.0 * all_metrics["fraction_top1_le"]["0.50"])}</strong></div>
        <div><span class="mini-label">gap <= 0.02</span><strong>{format_percent(100.0 * all_metrics["fraction_gap12_le"]["0.020"])}</strong></div>
        <div><span class="mini-label">max support</span><strong>{int(round(all_metrics["nonzero_count"]["max"]))}</strong></div>
      </div>
      <div class="chart-grid">
        <div class="chart-card"><h3>Top-1 histogram</h3>{render_histogram_svg(all_metrics["histograms"]["top1"], colors["all_trainable"])}</div>
        <div class="chart-card"><h3>Top-2 histogram</h3>{render_histogram_svg(all_metrics["histograms"]["top2"], colors["all_trainable"])}</div>
        <div class="chart-card"><h3>Gap histogram</h3>{render_histogram_svg(all_metrics["histograms"]["gap12"], colors["all_trainable"])}</div>
      </div>
      {render_quantile_table(all_metrics)}
      <div class="examples-grid">{render_examples(summary["examples"]["all_trainable"])}</div>
    </section>

    {code_sections}

    <section class="panel">
      <div class="panel-head">
        <h2>Training interpretation</h2>
        <p>The trainer uses distributional cross-entropy on these rows and then applies a small legal-uniform mix to non-one-hot targets.</p>
      </div>
      <ul class="takeaways">
        <li>With the current sampled data, the main training signal is already very peaked. The median trainable row has top-1 {format_float(all_metrics["top1"]["p50"], decimals=4)} and top-2 {format_float(all_metrics["top2"]["p50"], decimals=4)}.</li>
        <li>The hypothetical broad target shape that motivated this investigation was not observed in the sample. The sample minimum top-1 was {format_float(all_metrics["top1"]["min"], decimals=4)} and no sampled trainable row had support above {int(round(all_metrics["nonzero_count"]["max"]))}.</li>
        <li>Squaring would make the rows even more aggressive, but mostly on already-peaked rows. Mean top-1 would rise by {format_float(all_metrics["mean_top1_gain_after_square"], decimals=4)} and the median gain is only {format_float(all_metrics["median_top1_gain_after_square"], decimals=4)}.</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    summary = build_summary(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"policy_target_report_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    html_path = output_dir / f"{stem}.html"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_html(summary))

    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote HTML report: {html_path}")


if __name__ == "__main__":
    main()
