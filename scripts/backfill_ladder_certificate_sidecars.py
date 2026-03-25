"""Backfill ladder-certificate sidecars for TRMPH game files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hex_ai.utils.ladder_templates import (
    build_ladder_certificate_sidecar_records_for_trmph_file,
    ladder_certificate_sidecar_path_for_trmph,
    load_hexwiki_generated_ladder_templates,
    load_ladder_template_library,
    write_ladder_certificate_sidecar,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ladder_certificate_sidecars.log"),
            logging.StreamHandler(),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build incremental ladder-certificate sidecars for .trmph files.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory to scan recursively for .trmph files when --trmph-file is not provided.",
    )
    parser.add_argument(
        "--trmph-file",
        action="append",
        default=[],
        help="Specific .trmph file to process. May be passed multiple times.",
    )
    parser.add_argument(
        "--library-dir",
        default=None,
        help="Optional ladder-template library directory. Defaults to the generated HexWiki corpus.",
    )
    parser.add_argument(
        "--include-winnerless",
        action="store_true",
        help="Also sidecar-encode parseable TRMPH lines that do not carry a winner indicator.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing .ladder_certificates.jsonl sidecars instead of skipping them.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of files processed.",
    )
    return parser.parse_args()


def _discover_trmph_files(args: argparse.Namespace) -> list[Path]:
    if args.trmph_file:
        files = [Path(path) for path in args.trmph_file]
    else:
        files = sorted(Path(args.data_dir).glob("**/*.trmph"))
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"TRMPH file not found: {path}")
        if not path.is_file():
            raise ValueError(f"TRMPH path is not a file: {path}")
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError(f"--max-files must be positive, got {args.max_files}")
        files = files[: args.max_files]
    return files


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)
    args = _parse_args()

    trmph_files = _discover_trmph_files(args)
    if not trmph_files:
        logger.warning("No .trmph files found to process.")
        return

    if args.library_dir:
        templates = load_ladder_template_library(args.library_dir, source_set="custom")
        logger.info("Loaded %d ladder templates from %s", len(templates), args.library_dir)
    else:
        templates = load_hexwiki_generated_ladder_templates()
        logger.info("Loaded %d ladder templates from the generated HexWiki corpus", len(templates))

    files_written = 0
    files_skipped = 0
    total_games = 0
    total_positions = 0
    total_embeddings = 0
    total_elapsed_ms = 0.0

    for trmph_file in trmph_files:
        sidecar_path = ladder_certificate_sidecar_path_for_trmph(trmph_file)
        if sidecar_path.exists() and not args.overwrite:
            logger.info("Skipping existing sidecar: %s", sidecar_path)
            files_skipped += 1
            continue

        records = build_ladder_certificate_sidecar_records_for_trmph_file(
            trmph_file,
            templates,
            include_winnerless=bool(args.include_winnerless),
        )
        if not records:
            logger.info("No eligible games found in %s; no sidecar written.", trmph_file)
            files_skipped += 1
            continue

        write_ladder_certificate_sidecar(sidecar_path, records)
        files_written += 1

        file_games = len(records)
        file_positions = sum(record.position_count for record in records)
        file_embeddings = sum(
            sum(record.embeddings_considered_by_position) for record in records
        )
        file_elapsed_ms = sum(sum(record.elapsed_ms_by_position) for record in records)
        total_games += file_games
        total_positions += file_positions
        total_embeddings += file_embeddings
        total_elapsed_ms += file_elapsed_ms

        logger.info(
            "Wrote %s: games=%d positions=%d embeddings=%d matcher_ms=%.2f",
            sidecar_path,
            file_games,
            file_positions,
            file_embeddings,
            file_elapsed_ms,
        )

    logger.info("")
    logger.info("LADDER SIDECAR SUMMARY")
    logger.info("  Files written: %d", files_written)
    logger.info("  Files skipped: %d", files_skipped)
    logger.info("  Games encoded: %d", total_games)
    logger.info("  Positions encoded: %d", total_positions)
    logger.info("  Embeddings considered: %d", total_embeddings)
    logger.info("  Matcher elapsed ms: %.2f", total_elapsed_ms)


if __name__ == "__main__":
    main()
