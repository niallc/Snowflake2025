#!/usr/bin/env python3
"""Adaptive MCTS parameter job manager for fixed-checkpoint A/B tournaments.

This script runs staged A/B tournaments, escalates sample size when results are
close, records per-batch and per-job summaries, and can promote branch-best
configs when evidence is strong enough.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENING_FILE = REPO_ROOT / "data/tournament_play/two_stage_tournament_20260228_212421/openings.txt"
DEFAULT_OUT_DIR = REPO_ROOT / "temp/mcts_param_investigation_20260301_adaptive"
RUN_TOURNAMENT = REPO_ROOT / "scripts/run_tournament.py"
VENV_DIR = REPO_ROOT / "hex_ai_env"
VENV_BIN = VENV_DIR / "bin"
PYTHON_BIN = VENV_BIN / "python"

STATE_FILE = "state.json"
BATCH_RESULTS_FILE = "batch_results.csv"
JOB_SUMMARY_FILE = "job_summaries.csv"
RAW_LOG_FILE = "raw.log"

WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Config:
    sims: int
    c_puct: float = 2.9
    c_scale: float = 100.0
    gumbel_power_scale: float = 75.0
    gumbel_power_rate: float = 0.42
    gumbel_power_offset: float = -15.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sims": self.sims,
            "c_puct": self.c_puct,
            "c_scale": self.c_scale,
            "gumbel_power_scale": self.gumbel_power_scale,
            "gumbel_power_rate": self.gumbel_power_rate,
            "gumbel_power_offset": self.gumbel_power_offset,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        return Config(
            sims=int(d["sims"]),
            c_puct=float(d.get("c_puct", 2.9)),
            c_scale=float(d.get("c_scale", 100.0)),
            gumbel_power_scale=float(d.get("gumbel_power_scale", 75.0)),
            gumbel_power_rate=float(d.get("gumbel_power_rate", 0.42)),
            gumbel_power_offset=float(d.get("gumbel_power_offset", -15.0)),
        )

    def with_overrides(self, overrides: Dict[str, Any]) -> "Config":
        return Config(
            sims=int(overrides.get("sims", self.sims)),
            c_puct=float(overrides.get("c_puct", self.c_puct)),
            c_scale=float(overrides.get("c_scale", self.c_scale)),
            gumbel_power_scale=float(overrides.get("gumbel_power_scale", self.gumbel_power_scale)),
            gumbel_power_rate=float(overrides.get("gumbel_power_rate", self.gumbel_power_rate)),
            gumbel_power_offset=float(overrides.get("gumbel_power_offset", self.gumbel_power_offset)),
        )

    def label(self) -> str:
        return (
            "s"
            f"{self.sims}_cp{self.c_puct:g}_cs{self.c_scale:g}"
            f"_gps{self.gumbel_power_scale:g}_gpr{self.gumbel_power_rate:g}_gpo{self.gumbel_power_offset:g}"
        )


@dataclass
class BatchResult:
    job_id: str
    batch_index: int
    seed: int
    openings: int
    challenger: Config
    incumbent: Config
    challenger_wins: int
    incumbent_wins: int
    games: int
    challenger_rate: float
    challenger_ci_low: float
    challenger_ci_high: float
    challenger_avg_time: float
    incumbent_avg_time: float
    total_avg_time: float
    output_dir: str
    csv_path: str


@dataclass
class AggregateResult:
    challenger_wins: int
    incumbent_wins: int
    games: int
    challenger_rate: float
    challenger_ci_low: float
    challenger_ci_high: float
    challenger_avg_time: float
    incumbent_avg_time: float
    total_avg_time: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wilson_interval(wins: int, total: int, z: float = WILSON_Z_95) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = wins / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    radius = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    return (max(0.0, center - radius), min(1.0, center + radius))


def decision_label(ci_low: float, ci_high: float, rate: float, games: int, max_games: int) -> str:
    if ci_low >= 0.53:
        return "challenger_clearly_better"
    if ci_high <= 0.47:
        return "challenger_clearly_worse"
    if games >= max_games:
        if ci_low > 0.5:
            return "challenger_slight_edge"
        if ci_high < 0.5:
            return "challenger_slight_deficit"
        if 0.47 <= rate <= 0.53:
            return "near_tie_no_material_effect"
        return "uncertain_at_max_games"
    return "need_more_games"


def is_terminal_decision(label: str) -> bool:
    return label != "need_more_games"


def should_promote(label: str) -> bool:
    return label == "challenger_clearly_better"


def append_csv_header_if_missing(path: Path, headers: List[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_batch_result(path: Path, result: BatchResult) -> None:
    append_csv_header_if_missing(
        path,
        [
            "timestamp_utc",
            "job_id",
            "batch_index",
            "seed",
            "openings",
            "challenger",
            "incumbent",
            "challenger_wins",
            "incumbent_wins",
            "games",
            "challenger_rate",
            "challenger_ci_low",
            "challenger_ci_high",
            "challenger_avg_time",
            "incumbent_avg_time",
            "total_avg_time",
            "output_dir",
            "csv_path",
        ],
    )
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                utc_now_iso(),
                result.job_id,
                result.batch_index,
                result.seed,
                result.openings,
                json.dumps(result.challenger.as_dict(), sort_keys=True),
                json.dumps(result.incumbent.as_dict(), sort_keys=True),
                result.challenger_wins,
                result.incumbent_wins,
                result.games,
                f"{result.challenger_rate:.6f}",
                f"{result.challenger_ci_low:.6f}",
                f"{result.challenger_ci_high:.6f}",
                f"{result.challenger_avg_time:.6f}",
                f"{result.incumbent_avg_time:.6f}",
                f"{result.total_avg_time:.6f}",
                result.output_dir,
                result.csv_path,
            ]
        )


def append_job_summary(
    path: Path,
    *,
    job_id: str,
    branch: Optional[str],
    challenger: Config,
    incumbent: Config,
    aggregate: AggregateResult,
    decision: str,
    promoted: bool,
) -> None:
    append_csv_header_if_missing(
        path,
        [
            "timestamp_utc",
            "job_id",
            "branch",
            "challenger",
            "incumbent",
            "challenger_wins",
            "incumbent_wins",
            "games",
            "challenger_rate",
            "challenger_ci_low",
            "challenger_ci_high",
            "challenger_avg_time",
            "incumbent_avg_time",
            "total_avg_time",
            "decision",
            "promoted",
        ],
    )
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                utc_now_iso(),
                job_id,
                branch or "",
                json.dumps(challenger.as_dict(), sort_keys=True),
                json.dumps(incumbent.as_dict(), sort_keys=True),
                aggregate.challenger_wins,
                aggregate.incumbent_wins,
                aggregate.games,
                f"{aggregate.challenger_rate:.6f}",
                f"{aggregate.challenger_ci_low:.6f}",
                f"{aggregate.challenger_ci_high:.6f}",
                f"{aggregate.challenger_avg_time:.6f}",
                f"{aggregate.incumbent_avg_time:.6f}",
                f"{aggregate.total_avg_time:.6f}",
                decision,
                "true" if promoted else "false",
            ]
        )


def summarize_tournament_csv(csv_path: Path) -> Dict[str, Any]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in tournament CSV: {csv_path}")

    strategy_a = rows[0]["strategy_a"]
    strategy_b = rows[0]["strategy_b"]
    wins = {strategy_a: 0, strategy_b: 0}
    strategy_time = {strategy_a: 0.0, strategy_b: 0.0}
    total_game_time = 0.0

    for row in rows:
        winner = row["winner_strategy"]
        if winner in wins:
            wins[winner] += 1
        strategy_time[row["strategy_a"]] += float(row["strategy_a_time"])
        strategy_time[row["strategy_b"]] += float(row["strategy_b_time"])
        total_game_time += float(row["total_game_time"])

    total_games = len(rows)
    wa = wins[strategy_a]
    wb = wins[strategy_b]
    ci_low, ci_high = wilson_interval(wa, total_games)
    return {
        "strategy_a_name": strategy_a,
        "strategy_b_name": strategy_b,
        "a_wins": wa,
        "b_wins": wb,
        "games": total_games,
        "a_rate": wa / total_games,
        "a_ci_low": ci_low,
        "a_ci_high": ci_high,
        "a_avg_time": strategy_time[strategy_a] / total_games,
        "b_avg_time": strategy_time[strategy_b] / total_games,
        "total_avg_time": total_game_time / total_games,
    }


def aggregate_batches(batches: List[BatchResult]) -> AggregateResult:
    if not batches:
        raise RuntimeError("Cannot aggregate empty batch list")

    total_games = sum(b.games for b in batches)
    challenger_wins = sum(b.challenger_wins for b in batches)
    incumbent_wins = sum(b.incumbent_wins for b in batches)
    challenger_rate = challenger_wins / total_games
    ci_low, ci_high = wilson_interval(challenger_wins, total_games)

    total_challenger_time = sum(b.challenger_avg_time * b.games for b in batches)
    total_incumbent_time = sum(b.incumbent_avg_time * b.games for b in batches)
    total_game_time = sum(b.total_avg_time * b.games for b in batches)

    return AggregateResult(
        challenger_wins=challenger_wins,
        incumbent_wins=incumbent_wins,
        games=total_games,
        challenger_rate=challenger_rate,
        challenger_ci_low=ci_low,
        challenger_ci_high=ci_high,
        challenger_avg_time=total_challenger_time / total_games,
        incumbent_avg_time=total_incumbent_time / total_games,
        total_avg_time=total_game_time / total_games,
    )


def find_tournament_output_dir(stdout: str) -> Path:
    matches = re.findall(r"Directory:\s*(\S+)", stdout)
    if not matches:
        raise RuntimeError("Could not parse tournament output directory from run_tournament output")
    out_path = Path(matches[-1])
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    if not out_path.is_dir():
        raise RuntimeError(f"Parsed output directory does not exist: {out_path}")
    return out_path


def run_single_batch(
    *,
    raw_log: Path,
    opening_file: Path,
    seed: int,
    openings: int,
    job_id: str,
    batch_index: int,
    challenger: Config,
    incumbent: Config,
    model_spec: str,
) -> BatchResult:
    if model_spec == "best":
        model_args = ["--models=best,best"]
    else:
        raise ValueError(f"Unsupported model_spec: {model_spec}")

    cmd = [
        str(PYTHON_BIN),
        str(RUN_TOURNAMENT),
        f"--round-robin-games={openings}",
        f"--opening-file={opening_file}",
        f"--seed={seed}",
        *model_args,
        "--strategies=mcts,mcts",
        f"--mcts-sims={challenger.sims},{incumbent.sims}",
        f"--c-puct={challenger.c_puct},{incumbent.c_puct}",
        "--enable-gumbel=true,true",
        f"--gumbel-c-scale={challenger.c_scale},{incumbent.c_scale}",
        (
            "--gumbel-candidate-power-scale="
            f"{challenger.gumbel_power_scale},{incumbent.gumbel_power_scale}"
        ),
        (
            "--gumbel-candidate-power-rate="
            f"{challenger.gumbel_power_rate},{incumbent.gumbel_power_rate}"
        ),
        (
            "--gumbel-candidate-power-offset="
            f"{challenger.gumbel_power_offset},{incumbent.gumbel_power_offset}"
        ),
        "--temperature=1.0",
        f"--run-desc=adaptive_mcts_{job_id}_b{batch_index}",
    ]

    started = utc_now_iso()
    cmd_str = " ".join(shlex.quote(p) for p in cmd)
    with raw_log.open("a") as logf:
        logf.write(f"\n[{started}] RUN job={job_id} batch={batch_index}\n")
        logf.write(f"CMD: {cmd_str}\n")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={
            **os.environ,
            "VIRTUAL_ENV": str(VENV_DIR),
            "PATH": f"{VENV_BIN}:{os.environ.get('PATH', '')}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    finished = utc_now_iso()
    with raw_log.open("a") as logf:
        logf.write(f"[{finished}] RETURN CODE {proc.returncode}\n")
        if proc.stdout:
            logf.write("STDOUT:\n")
            logf.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                logf.write("\n")
        if proc.stderr:
            logf.write("STDERR:\n")
            logf.write(proc.stderr)
            if not proc.stderr.endswith("\n"):
                logf.write("\n")

    if proc.returncode != 0:
        raise RuntimeError(
            f"run_tournament failed for job={job_id} batch={batch_index}, "
            f"seed={seed}, return_code={proc.returncode}. See {raw_log}."
        )

    output_dir = find_tournament_output_dir(proc.stdout)
    csv_files = sorted(output_dir.glob("*.csv"))
    if len(csv_files) != 1:
        raise RuntimeError(f"Expected exactly one CSV in {output_dir}, found {len(csv_files)}")
    csv_path = csv_files[0]

    summary = summarize_tournament_csv(csv_path)
    return BatchResult(
        job_id=job_id,
        batch_index=batch_index,
        seed=seed,
        openings=openings,
        challenger=challenger,
        incumbent=incumbent,
        challenger_wins=summary["a_wins"],
        incumbent_wins=summary["b_wins"],
        games=summary["games"],
        challenger_rate=summary["a_rate"],
        challenger_ci_low=summary["a_ci_low"],
        challenger_ci_high=summary["a_ci_high"],
        challenger_avg_time=summary["a_avg_time"],
        incumbent_avg_time=summary["b_avg_time"],
        total_avg_time=summary["total_avg_time"],
        output_dir=str(output_dir),
        csv_path=str(csv_path),
    )


def default_jobs() -> List[Dict[str, Any]]:
    return [
        {
            "job_id": "head_to_head_high_vs_fast",
            "job_type": "branch_vs_branch",
            "challenger_branch": "high_budget_best",
            "incumbent_branch": "fast_budget_best",
            "branch_to_update": None,
            "notes": "Quantify strength and speed tradeoff between current best high and fast configs.",
        },
        {
            "job_id": "confirm_high_vs56",
            "job_type": "branch_vs_fixed",
            "challenger_branch": "high_budget_best",
            "incumbent_fixed": Config(sims=56, c_puct=2.9, c_scale=100.0).as_dict(),
            "branch_to_update": None,
            "notes": "Re-check high-budget branch against sims56 baseline under current best settings.",
        },
        {
            "job_id": "high_gpow_scale_60",
            "job_type": "branch_variant",
            "branch": "high_budget_best",
            "overrides": {"gumbel_power_scale": 60.0},
            "branch_to_update": "high_budget_best",
            "notes": "Probe lower gumbel candidate power scale at high-budget branch.",
        },
        {
            "job_id": "high_gpow_scale_90",
            "job_type": "branch_variant",
            "branch": "high_budget_best",
            "overrides": {"gumbel_power_scale": 90.0},
            "branch_to_update": "high_budget_best",
            "notes": "Probe higher gumbel candidate power scale at high-budget branch.",
        },
        {
            "job_id": "high_gpow_rate_0p36",
            "job_type": "branch_variant",
            "branch": "high_budget_best",
            "overrides": {"gumbel_power_rate": 0.36},
            "branch_to_update": "high_budget_best",
            "notes": "Probe lower gumbel candidate power rate at high-budget branch.",
        },
        {
            "job_id": "high_gpow_rate_0p48",
            "job_type": "branch_variant",
            "branch": "high_budget_best",
            "overrides": {"gumbel_power_rate": 0.48},
            "branch_to_update": "high_budget_best",
            "notes": "Probe higher gumbel candidate power rate at high-budget branch.",
        },
        {
            "job_id": "high_gpow_offset_neg25",
            "job_type": "branch_variant",
            "branch": "high_budget_best",
            "overrides": {"gumbel_power_offset": -25.0},
            "branch_to_update": "high_budget_best",
            "notes": "Probe more negative gumbel candidate power offset at high-budget branch.",
        },
        {
            "job_id": "high_gpow_offset_neg5",
            "job_type": "branch_variant",
            "branch": "high_budget_best",
            "overrides": {"gumbel_power_offset": -5.0},
            "branch_to_update": "high_budget_best",
            "notes": "Probe less negative gumbel candidate power offset at high-budget branch.",
        },
        {
            "job_id": "fast_cscale_80",
            "job_type": "branch_variant",
            "branch": "fast_budget_best",
            "overrides": {"c_scale": 80.0},
            "branch_to_update": "fast_budget_best",
            "notes": "Check whether c_scale=80 helps on the fast-budget branch.",
        },
        {
            "job_id": "fast_gpow_scale_60",
            "job_type": "branch_variant",
            "branch": "fast_budget_best",
            "overrides": {"gumbel_power_scale": 60.0},
            "branch_to_update": "fast_budget_best",
            "notes": "Probe lower gumbel candidate power scale at fast-budget branch.",
        },
    ]


def resolve_job_configs(job: Dict[str, Any], branches: Dict[str, Dict[str, Any]]) -> Tuple[Config, Config]:
    job_type = job["job_type"]
    if job_type == "branch_vs_branch":
        challenger = Config.from_dict(branches[job["challenger_branch"]])
        incumbent = Config.from_dict(branches[job["incumbent_branch"]])
        return challenger, incumbent
    if job_type == "branch_vs_fixed":
        challenger = Config.from_dict(branches[job["challenger_branch"]])
        incumbent = Config.from_dict(job["incumbent_fixed"])
        return challenger, incumbent
    if job_type == "branch_variant":
        base = Config.from_dict(branches[job["branch"]])
        challenger = base.with_overrides(job["overrides"])
        incumbent = base
        return challenger, incumbent
    raise ValueError(f"Unsupported job_type: {job_type}")


def next_seed(state: Dict[str, Any]) -> int:
    seed = int(state["next_seed"])
    state["next_seed"] = seed + 1
    return seed


def load_or_initialize_state(state_path: Path, *, seed_start: int) -> Dict[str, Any]:
    if state_path.exists():
        loaded = json.loads(state_path.read_text())
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Invalid state file format: {state_path}")
        loaded.setdefault("jobs", [])
        loaded.setdefault("job_results", {})
        loaded.setdefault("run_history", [])
        loaded.setdefault("branches", {})
        loaded.setdefault("next_seed", seed_start)
        return loaded

    state = {
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "next_seed": seed_start,
        "branches": {
            "high_budget_best": Config(sims=80, c_puct=2.9, c_scale=80.0).as_dict(),
            "fast_budget_best": Config(sims=50, c_puct=2.9, c_scale=100.0).as_dict(),
        },
        "jobs": default_jobs(),
        "job_results": {},
        "run_history": [],
    }
    return state


def persist_state(state_path: Path, state: Dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now_iso()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def run_job(
    *,
    state: Dict[str, Any],
    job: Dict[str, Any],
    out_dir: Path,
    opening_file: Path,
    opening_schedule: List[int],
    max_games: int,
    model_spec: str,
) -> None:
    job_id = job["job_id"]
    challenger, incumbent = resolve_job_configs(job, state["branches"])
    if challenger == incumbent:
        state["job_results"][job_id] = {
            "decision": "skipped_identical_configs",
            "challenger": challenger.as_dict(),
            "incumbent": incumbent.as_dict(),
            "aggregate": None,
            "promoted": False,
            "finished_at_utc": utc_now_iso(),
        }
        return

    raw_log = out_dir / RAW_LOG_FILE
    batch_csv = out_dir / BATCH_RESULTS_FILE
    summary_csv = out_dir / JOB_SUMMARY_FILE

    completed_batch_dicts = [
        row
        for row in state["run_history"]
        if row.get("job_id") == job_id
        and json.dumps(row.get("challenger", {}), sort_keys=True) == json.dumps(challenger.as_dict(), sort_keys=True)
        and json.dumps(row.get("incumbent", {}), sort_keys=True) == json.dumps(incumbent.as_dict(), sort_keys=True)
    ]
    completed_batches: List[BatchResult] = []
    for row in completed_batch_dicts:
        completed_batches.append(
            BatchResult(
                job_id=row["job_id"],
                batch_index=int(row["batch_index"]),
                seed=int(row["seed"]),
                openings=int(row["openings"]),
                challenger=Config.from_dict(row["challenger"]),
                incumbent=Config.from_dict(row["incumbent"]),
                challenger_wins=int(row["challenger_wins"]),
                incumbent_wins=int(row["incumbent_wins"]),
                games=int(row["games"]),
                challenger_rate=float(row["challenger_rate"]),
                challenger_ci_low=float(row["challenger_ci_low"]),
                challenger_ci_high=float(row["challenger_ci_high"]),
                challenger_avg_time=float(row["challenger_avg_time"]),
                incumbent_avg_time=float(row["incumbent_avg_time"]),
                total_avg_time=float(row["total_avg_time"]),
                output_dir=row["output_dir"],
                csv_path=row["csv_path"],
            )
        )
    completed_batches.sort(key=lambda b: b.batch_index)

    batch_index = len(completed_batches)
    aggregate = aggregate_batches(completed_batches) if completed_batches else None
    decision = (
        decision_label(
            aggregate.challenger_ci_low,
            aggregate.challenger_ci_high,
            aggregate.challenger_rate,
            aggregate.games,
            max_games,
        )
        if aggregate is not None
        else "need_more_games"
    )

    while not is_terminal_decision(decision):
        if batch_index >= len(opening_schedule):
            # If schedule is exhausted but decision is still not terminal, keep reusing
            # the last stage size until max_games or terminal decision.
            openings = opening_schedule[-1]
        else:
            openings = opening_schedule[batch_index]

        seed = next_seed(state)
        batch = run_single_batch(
            raw_log=raw_log,
            opening_file=opening_file,
            seed=seed,
            openings=openings,
            job_id=job_id,
            batch_index=batch_index,
            challenger=challenger,
            incumbent=incumbent,
            model_spec=model_spec,
        )
        append_batch_result(batch_csv, batch)

        state["run_history"].append(
            {
                "timestamp_utc": utc_now_iso(),
                "job_id": batch.job_id,
                "batch_index": batch.batch_index,
                "seed": batch.seed,
                "openings": batch.openings,
                "challenger": batch.challenger.as_dict(),
                "incumbent": batch.incumbent.as_dict(),
                "challenger_wins": batch.challenger_wins,
                "incumbent_wins": batch.incumbent_wins,
                "games": batch.games,
                "challenger_rate": batch.challenger_rate,
                "challenger_ci_low": batch.challenger_ci_low,
                "challenger_ci_high": batch.challenger_ci_high,
                "challenger_avg_time": batch.challenger_avg_time,
                "incumbent_avg_time": batch.incumbent_avg_time,
                "total_avg_time": batch.total_avg_time,
                "output_dir": batch.output_dir,
                "csv_path": batch.csv_path,
            }
        )
        persist_state(out_dir / STATE_FILE, state)

        completed_batches.append(batch)
        aggregate = aggregate_batches(completed_batches)
        decision = decision_label(
            aggregate.challenger_ci_low,
            aggregate.challenger_ci_high,
            aggregate.challenger_rate,
            aggregate.games,
            max_games,
        )
        batch_index += 1

        if aggregate.games >= max_games:
            decision = decision_label(
                aggregate.challenger_ci_low,
                aggregate.challenger_ci_high,
                aggregate.challenger_rate,
                aggregate.games,
                max_games,
            )
            break

    if aggregate is None:
        raise RuntimeError(f"Job {job_id} completed without aggregate result")

    promoted = False
    branch_to_update = job.get("branch_to_update")
    if branch_to_update and should_promote(decision):
        state["branches"][branch_to_update] = challenger.as_dict()
        promoted = True

    append_job_summary(
        summary_csv,
        job_id=job_id,
        branch=branch_to_update,
        challenger=challenger,
        incumbent=incumbent,
        aggregate=aggregate,
        decision=decision,
        promoted=promoted,
    )

    state["job_results"][job_id] = {
        "decision": decision,
        "challenger": challenger.as_dict(),
        "incumbent": incumbent.as_dict(),
        "aggregate": {
            "challenger_wins": aggregate.challenger_wins,
            "incumbent_wins": aggregate.incumbent_wins,
            "games": aggregate.games,
            "challenger_rate": aggregate.challenger_rate,
            "challenger_ci_low": aggregate.challenger_ci_low,
            "challenger_ci_high": aggregate.challenger_ci_high,
            "challenger_avg_time": aggregate.challenger_avg_time,
            "incumbent_avg_time": aggregate.incumbent_avg_time,
            "total_avg_time": aggregate.total_avg_time,
        },
        "promoted": promoted,
        "finished_at_utc": utc_now_iso(),
    }
    persist_state(out_dir / STATE_FILE, state)


def parse_opening_schedule(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("opening schedule must contain at least one integer")
    schedule = []
    for p in parts:
        n = int(p)
        if n <= 0:
            raise argparse.ArgumentTypeError("opening schedule values must be > 0")
        schedule.append(n)
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run adaptive fixed-model MCTS parameter jobs with staged sample sizing. "
            "Promotes branch-best configs only when the challenger has clear evidence."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for state and logs (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--opening-file",
        type=Path,
        default=DEFAULT_OPENING_FILE,
        help=f"Fixed openings file for deterministic comparisons (default: {DEFAULT_OPENING_FILE})",
    )
    parser.add_argument(
        "--opening-schedule",
        type=parse_opening_schedule,
        default=parse_opening_schedule("20,30,50"),
        help=(
            "Comma-separated openings per stage (each opening yields 2 games). "
            "Default: 20,30,50 -> up to 200 games before repeated final stage."
        ),
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=200,
        help="Maximum games per job before forcing a decision (default: 200).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=2,
        help="Maximum pending jobs to process this invocation (default: 2).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=20260301,
        help="Starting seed for first batch if no state exists (default: 20260301).",
    )
    parser.add_argument(
        "--model-spec",
        type=str,
        default="best",
        choices=["best"],
        help='Model selection mode (default: "best" -> --models=best,best).',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_games <= 0:
        raise RuntimeError("--max-games must be > 0")
    if args.max_jobs <= 0:
        raise RuntimeError("--max-jobs must be > 0")
    if not args.opening_file.is_file():
        raise RuntimeError(f"Opening file not found: {args.opening_file}")
    if not PYTHON_BIN.is_file():
        raise RuntimeError(
            f"Python interpreter not found at {PYTHON_BIN}. "
            "Ensure hex_ai_env exists and is set up."
        )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / STATE_FILE
    state = load_or_initialize_state(state_path, seed_start=args.seed_start)
    persist_state(state_path, state)

    pending_jobs = [j for j in state.get("jobs", []) if j["job_id"] not in state.get("job_results", {})]
    if not pending_jobs:
        print("No pending jobs. All configured jobs are complete.")
        return 0

    processed = 0
    for job in pending_jobs:
        if processed >= args.max_jobs:
            break
        job_id = job["job_id"]
        print(f"[start] {job_id}: {job.get('notes', '')}")
        run_job(
            state=state,
            job=job,
            out_dir=out_dir,
            opening_file=args.opening_file,
            opening_schedule=args.opening_schedule,
            max_games=args.max_games,
            model_spec=args.model_spec,
        )
        result = state["job_results"][job_id]
        aggregate = result.get("aggregate")
        if aggregate:
            print(
                f"[done] {job_id}: {aggregate['challenger_wins']}-{aggregate['incumbent_wins']} "
                f"({aggregate['challenger_rate']:.3f}, "
                f"95% CI {aggregate['challenger_ci_low']:.3f}-{aggregate['challenger_ci_high']:.3f}), "
                f"decision={result['decision']}, promoted={result['promoted']}"
            )
        else:
            print(f"[done] {job_id}: decision={result['decision']}, promoted={result['promoted']}")
        processed += 1

    persist_state(state_path, state)
    print(f"Processed {processed} job(s). State: {state_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
