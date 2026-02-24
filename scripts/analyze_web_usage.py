#!/usr/bin/env python3
"""Simple analyzer for Snowflake web usage JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze web_usage JSONL files and print a basic usage summary."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSONL files (for example: logs/hetzner_usage/web_usage_2026_02.jsonl).",
    )
    parser.add_argument(
        "--session-gap-minutes",
        type=float,
        default=30.0,
        help="Gap threshold used to split sessions per client_id (default: 30).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top items to show in each breakdown (default: 10).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full summary as JSON.",
    )
    parser.add_argument(
        "--exclude-top-clients",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Also print extra summaries with top-N busiest client_ids excluded "
            "(for example: --exclude-top-clients 1 3)."
        ),
    )
    return parser.parse_args()


def parse_iso_ts(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return float(sorted_vals[floor])
    lower = sorted_vals[floor]
    upper = sorted_vals[ceil]
    return float(lower + (upper - lower) * (k - floor))


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    count = len(values)
    return {
        "count": count,
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / count),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def counter_to_sorted_pairs(counter: Counter[Any]) -> list[tuple[str, int]]:
    return sorted(((str(k), int(v)) for k, v in counter.items()), key=lambda x: (-x[1], x[0]))


def print_counter(title: str, counter: Counter[Any], total: int | None, top_n: int) -> None:
    print(f"\n{title}")
    for key, count in counter_to_sorted_pairs(counter)[:top_n]:
        if total:
            pct = (count / total) * 100.0
            print(f"  {key}: {count} ({pct:.1f}%)")
        else:
            print(f"  {key}: {count}")


def short_id(value: str, keep: int = 8) -> str:
    if len(value) <= keep:
        return value
    return value[:keep] + "..."


def analyze(
    files: list[Path],
    session_gap_seconds: float,
    excluded_client_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded_client_ids = excluded_client_ids or set()

    rows_total = 0
    rows_valid = 0
    rows_invalid = 0
    rows_excluded_client = 0

    first_ts: datetime | None = None
    last_ts: datetime | None = None

    event_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    status_counts: Counter[int] = Counter()
    method_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    elo_counts: Counter[int] = Counter()
    board_size_counts: Counter[int] = Counter()
    state_reason_counts: Counter[str] = Counter()
    pie_rule_counts: Counter[bool] = Counter()
    heatmap_enabled_counts: Counter[bool] = Counter()

    day_event_counts: Counter[str] = Counter()
    hour_event_counts: Counter[int] = Counter()
    day_clients: dict[str, set[str]] = defaultdict(set)
    client_event_counts: Counter[str] = Counter()
    client_timestamps: dict[str, list[datetime]] = defaultdict(list)

    duration_values: list[float] = []
    trmph_moves_values: list[float] = []
    trmph_len_values: list[float] = []
    durations_by_event: dict[str, list[float]] = defaultdict(list)

    ua_hashes: set[str] = set()
    ip_hashes: set[str] = set()
    client_ids: set[str] = set()

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows_total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    rows_invalid += 1
                    continue

                client_id = row.get("client_id")
                if isinstance(client_id, str) and client_id in excluded_client_ids:
                    rows_excluded_client += 1
                    continue

                rows_valid += 1

                ts_raw = row.get("ts")
                ts = None
                if isinstance(ts_raw, str):
                    try:
                        ts = parse_iso_ts(ts_raw)
                    except Exception:
                        ts = None

                if ts is not None:
                    if first_ts is None or ts < first_ts:
                        first_ts = ts
                    if last_ts is None or ts > last_ts:
                        last_ts = ts
                    day_key = ts.date().isoformat()
                    day_event_counts[day_key] += 1
                    hour_event_counts[ts.hour] += 1

                event = str(row.get("event", "<missing>"))
                event_counts[event] += 1

                path_value = str(row.get("path", "<missing>"))
                path_counts[path_value] += 1

                status = row.get("status")
                if isinstance(status, int):
                    status_counts[status] += 1

                method = row.get("method")
                if isinstance(method, str):
                    method_counts[method] += 1

                model_id = row.get("model_id")
                if isinstance(model_id, str):
                    model_counts[model_id] += 1

                algorithm = row.get("algorithm")
                if isinstance(algorithm, str):
                    algorithm_counts[algorithm] += 1

                elo = row.get("elo_rating")
                if isinstance(elo, int):
                    elo_counts[elo] += 1

                board_size = row.get("display_board_size")
                if isinstance(board_size, int):
                    board_size_counts[board_size] += 1

                state_reason = row.get("state_reason")
                if isinstance(state_reason, str):
                    state_reason_counts[state_reason] += 1

                if "pie_rule_enabled" in row and isinstance(row.get("pie_rule_enabled"), bool):
                    pie_rule_counts[row["pie_rule_enabled"]] += 1

                if "heatmap_enabled" in row and isinstance(row.get("heatmap_enabled"), bool):
                    heatmap_enabled_counts[row["heatmap_enabled"]] += 1

                duration_ms = row.get("duration_ms")
                if isinstance(duration_ms, (int, float)):
                    value = float(duration_ms)
                    duration_values.append(value)
                    durations_by_event[event].append(value)

                trmph_moves = row.get("trmph_moves")
                if isinstance(trmph_moves, (int, float)):
                    trmph_moves_values.append(float(trmph_moves))

                trmph_len = row.get("trmph_len")
                if isinstance(trmph_len, (int, float)):
                    trmph_len_values.append(float(trmph_len))

                if isinstance(client_id, str):
                    client_ids.add(client_id)
                    client_event_counts[client_id] += 1
                    if ts is not None:
                        client_timestamps[client_id].append(ts)
                        day_key = ts.date().isoformat()
                        day_clients[day_key].add(client_id)

                ua_hash = row.get("ua_hash")
                if isinstance(ua_hash, str):
                    ua_hashes.add(ua_hash)

                ip_hash = row.get("ip_hash")
                if isinstance(ip_hash, str):
                    ip_hashes.add(ip_hash)

    session_event_counts: list[int] = []
    session_lengths_seconds: list[float] = []
    session_count = 0

    for client_id, stamps in client_timestamps.items():
        if not stamps:
            continue
        stamps = sorted(stamps)
        session_start = stamps[0]
        prev = stamps[0]
        events_in_session = 1

        for current in stamps[1:]:
            gap = (current - prev).total_seconds()
            if gap > session_gap_seconds:
                session_count += 1
                session_event_counts.append(events_in_session)
                session_lengths_seconds.append((prev - session_start).total_seconds())
                session_start = current
                events_in_session = 1
            else:
                events_in_session += 1
            prev = current

        session_count += 1
        session_event_counts.append(events_in_session)
        session_lengths_seconds.append((prev - session_start).total_seconds())

    duration_by_event_summary = {
        event: summarize_numeric(values) for event, values in durations_by_event.items()
    }

    summary: dict[str, Any] = {
        "files": [str(path) for path in files],
        "rows_total": rows_total,
        "rows_valid": rows_valid,
        "rows_invalid_json": rows_invalid,
        "rows_excluded_by_client_filter": rows_excluded_client,
        "excluded_client_ids": sorted(excluded_client_ids),
        "time_range_utc": {
            "start": first_ts.isoformat() if first_ts else None,
            "end": last_ts.isoformat() if last_ts else None,
            "span_days": (
                (last_ts - first_ts).total_seconds() / 86400.0
                if first_ts is not None and last_ts is not None
                else None
            ),
        },
        "unique_ids": {
            "client_ids": len(client_ids),
            "ua_hashes": len(ua_hashes),
            "ip_hashes": len(ip_hashes),
        },
        "events": dict(counter_to_sorted_pairs(event_counts)),
        "paths": dict(counter_to_sorted_pairs(path_counts)),
        "status_codes": dict(counter_to_sorted_pairs(status_counts)),
        "methods": dict(counter_to_sorted_pairs(method_counts)),
        "models": dict(counter_to_sorted_pairs(model_counts)),
        "algorithms": dict(counter_to_sorted_pairs(algorithm_counts)),
        "elo_ratings": dict(counter_to_sorted_pairs(elo_counts)),
        "display_board_sizes": dict(counter_to_sorted_pairs(board_size_counts)),
        "state_reasons": dict(counter_to_sorted_pairs(state_reason_counts)),
        "pie_rule_enabled": dict(counter_to_sorted_pairs(pie_rule_counts)),
        "heatmap_enabled": dict(counter_to_sorted_pairs(heatmap_enabled_counts)),
        "timing": {
            "event_counts_by_day_utc": dict(sorted(day_event_counts.items())),
            "event_counts_by_hour_utc": dict(counter_to_sorted_pairs(hour_event_counts)),
        },
        "active_clients_by_day_utc": {
            day: len(clients) for day, clients in sorted(day_clients.items())
        },
        "duration_ms": {
            "overall": summarize_numeric(duration_values),
            "by_event": duration_by_event_summary,
        },
        "trmph_metrics": {
            "trmph_moves": summarize_numeric(trmph_moves_values),
            "trmph_len": summarize_numeric(trmph_len_values),
        },
        "sessions": {
            "session_gap_seconds": session_gap_seconds,
            "session_count_estimate": session_count,
            "events_per_session": summarize_numeric([float(v) for v in session_event_counts]),
            "session_length_seconds": summarize_numeric(session_lengths_seconds),
        },
        "top_clients_by_event_count": [
            {"client_id": client_id, "event_count": count}
            for client_id, count in client_event_counts.most_common(20)
        ],
    }

    return summary


def print_summary(summary: dict[str, Any], top_n: int) -> None:
    print("Web Usage Summary")
    print("=" * 80)
    print(f"Files analyzed: {len(summary['files'])}")
    for path in summary["files"]:
        print(f"  - {path}")

    print(
        f"\nRows: {summary['rows_valid']} valid / {summary['rows_total']} total "
        f"(invalid JSON: {summary['rows_invalid_json']})"
    )
    if summary.get("excluded_client_ids"):
        print(
            "Excluded client filter: "
            f"{len(summary['excluded_client_ids'])} client_id(s), "
            f"excluded rows={summary.get('rows_excluded_by_client_filter', 0)}"
        )

    time_range = summary["time_range_utc"]
    print(f"Time range (UTC): {time_range['start']} -> {time_range['end']}")
    print(f"Span: {time_range['span_days']:.2f} days" if time_range["span_days"] is not None else "Span: n/a")

    unique_ids = summary["unique_ids"]
    print(
        f"Unique IDs: clients={unique_ids['client_ids']}, "
        f"ua_hashes={unique_ids['ua_hashes']}, ip_hashes={unique_ids['ip_hashes']}"
    )

    events_total = sum(summary["events"].values())
    print_counter("Events", Counter(summary["events"]), events_total, top_n)
    print_counter("Paths", Counter(summary["paths"]), events_total, top_n)
    print_counter("Status Codes", Counter(summary["status_codes"]), events_total, top_n)

    print_counter("ELO Ratings", Counter(summary["elo_ratings"]), events_total, top_n)
    print_counter("Display Board Sizes", Counter(summary["display_board_sizes"]), events_total, top_n)

    if summary["models"]:
        model_total = sum(summary["models"].values())
        print_counter("Model Usage", Counter(summary["models"]), model_total, top_n)
    if summary["algorithms"]:
        algo_total = sum(summary["algorithms"].values())
        print_counter("Algorithm Usage", Counter(summary["algorithms"]), algo_total, top_n)

    if summary["state_reasons"]:
        state_total = sum(summary["state_reasons"].values())
        print_counter("State Reasons", Counter(summary["state_reasons"]), state_total, top_n)

    overall_duration = summary["duration_ms"]["overall"]
    print(
        "\nDuration (ms): "
        f"mean={overall_duration['mean']:.1f}, "
        f"p50={overall_duration['p50']:.1f}, "
        f"p95={overall_duration['p95']:.1f}, "
        f"max={overall_duration['max']:.1f}"
        if overall_duration["count"] > 0
        else "\nDuration (ms): n/a"
    )

    print("\nDuration by Event (ms)")
    for event, stats in sorted(
        summary["duration_ms"]["by_event"].items(),
        key=lambda kv: (-kv[1]["count"], kv[0]),
    )[:top_n]:
        if stats["count"] == 0:
            continue
        print(
            f"  {event}: count={stats['count']}, mean={stats['mean']:.1f}, "
            f"p50={stats['p50']:.1f}, p95={stats['p95']:.1f}, max={stats['max']:.1f}"
        )

    print("\nEvents by Day (UTC)")
    for day, count in sorted(summary["timing"]["event_counts_by_day_utc"].items()):
        clients = summary["active_clients_by_day_utc"].get(day, 0)
        print(f"  {day}: {count} events, {clients} active clients")

    print("\nSessions (estimated from client_id and time gaps)")
    sessions = summary["sessions"]
    eps = sessions["events_per_session"]
    slen = sessions["session_length_seconds"]
    if sessions["session_count_estimate"] > 0:
        print(
            f"  session_gap_seconds={sessions['session_gap_seconds']}, "
            f"sessions={sessions['session_count_estimate']}, "
            f"events/session mean={eps['mean']:.2f}, p50={eps['p50']:.1f}, p95={eps['p95']:.1f}"
        )
        print(
            f"  session length seconds mean={slen['mean']:.1f}, "
            f"p50={slen['p50']:.1f}, p95={slen['p95']:.1f}, max={slen['max']:.1f}"
        )
    else:
        print("  no sessions could be estimated")

    print("\nTop Clients by Event Count")
    for entry in summary["top_clients_by_event_count"][:top_n]:
        print(f"  {short_id(entry['client_id'])}: {entry['event_count']}")


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in args.inputs]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"Missing input file(s): {', '.join(missing)}")

    session_gap_seconds = float(args.session_gap_minutes) * 60.0
    summary = analyze(files, session_gap_seconds=session_gap_seconds)
    print_summary(summary, top_n=max(1, args.top_n))

    scenarios: dict[str, Any] = {"baseline": summary}
    requested_top_n = sorted({n for n in args.exclude_top_clients if n > 0})
    if requested_top_n:
        baseline_top_clients = [entry["client_id"] for entry in summary["top_clients_by_event_count"]]
        for n in requested_top_n:
            excluded_ids = set(baseline_top_clients[:n])
            scenario_summary = analyze(
                files,
                session_gap_seconds=session_gap_seconds,
                excluded_client_ids=excluded_ids,
            )
            scenarios[f"exclude_top_{n}"] = scenario_summary
            print("\n")
            print_summary(scenario_summary, top_n=max(1, args.top_n))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = summary if not requested_top_n else scenarios
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
