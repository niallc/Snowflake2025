#!/usr/bin/env python3
"""
Prune old hyperparameter-tuning checkpoints with conservative safety rules.

Rules:
1. Keep all checkpoint files referenced by MODEL_GENERATIONS.
2. In each directory, keep the first and last checkpoint by training order.
3. In each directory, keep every Nth checkpoint (default: every 10th).

The script only considers model files that match:
    epoch{number}_mini{number}.pt
    epoch{number}_mini{number}.pt.gz
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


MODEL_FILENAME_RE = re.compile(r"^epoch(?P<epoch>\d+)_mini(?P<mini>\d+)\.pt(?:\.gz)?$")
MODEL_NO_EXT_RE = re.compile(r"^epoch\d+_mini\d+$")


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_epoch_mini(filename: str) -> Tuple[int, int] | None:
    match = MODEL_FILENAME_RE.match(filename)
    if not match:
        return None
    return int(match.group("epoch")), int(match.group("mini"))


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def load_model_generations(config_path: Path) -> Dict[int, dict]:
    source = config_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(config_path))

    model_generations_node = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "MODEL_GENERATIONS":
                model_generations_node = node.value
                break
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_GENERATIONS":
                    model_generations_node = node.value
                    break
            if model_generations_node is not None:
                break

    if model_generations_node is None:
        raise ValueError(f"MODEL_GENERATIONS not found in {config_path}")

    data = ast.literal_eval(model_generations_node)
    if not isinstance(data, dict):
        raise ValueError("MODEL_GENERATIONS must evaluate to a dictionary")

    return data


def build_protected_paths(
    model_generations: Dict[int, dict],
    checkpoints_root: Path,
) -> Tuple[Set[Path], List[Path], List[str]]:
    protected: Set[Path] = set()
    missing: List[str] = []
    fuzzy_matches: List[Path] = []

    for generation_data in model_generations.values():
        if not isinstance(generation_data, dict):
            continue

        dir_rel = generation_data.get("dir")
        models = generation_data.get("models")
        if not isinstance(dir_rel, str) or not isinstance(models, list):
            continue

        model_dir = (checkpoints_root / dir_rel).resolve()
        for model_name in models:
            if not isinstance(model_name, str):
                continue

            found_reference = False
            expected = (model_dir / model_name).resolve()
            if expected.is_file():
                protected.add(expected)
                found_reference = True

            # Conservative fallback for entries that omit extension (e.g., "epoch80_mini1").
            if not found_reference and MODEL_NO_EXT_RE.match(model_name) and model_dir.is_dir():
                for ext in (".pt.gz", ".pt"):
                    candidate = (model_dir / f"{model_name}{ext}").resolve()
                    if candidate.is_file():
                        protected.add(candidate)
                        fuzzy_matches.append(candidate)
                        found_reference = True

            if not found_reference:
                missing.append(f"{dir_rel}/{model_name}")

    return protected, fuzzy_matches, missing


def discover_candidate_files(target_root: Path) -> Dict[Path, List[Path]]:
    grouped: Dict[Path, List[Path]] = defaultdict(list)

    for file_path in target_root.rglob("*"):
        if not file_path.is_file():
            continue
        if parse_epoch_mini(file_path.name) is None:
            continue
        grouped[file_path.parent.resolve()].append(file_path.resolve())

    return grouped


def sort_key(path: Path) -> Tuple[int, int, str]:
    parsed = parse_epoch_mini(path.name)
    if parsed is None:
        # Should not happen because we pre-filter, but keep deterministic fallback.
        return (sys.maxsize, sys.maxsize, path.name)
    epoch, mini = parsed
    return epoch, mini, path.name


def compute_keep_and_delete(
    grouped_files: Dict[Path, List[Path]],
    protected_paths: Set[Path],
    keep_every: int,
) -> Tuple[List[Path], Set[Path]]:
    to_delete: List[Path] = []
    to_keep: Set[Path] = set()

    for directory in sorted(grouped_files):
        files = sorted(grouped_files[directory], key=sort_key)
        if not files:
            continue

        # Keep first and last checkpoint in this directory.
        to_keep.add(files[0])
        to_keep.add(files[-1])

        # Keep every Nth checkpoint by training-order index (10th, 20th, ...).
        for index, file_path in enumerate(files, start=1):
            if index % keep_every == 0:
                to_keep.add(file_path)

        # Keep anything referenced by MODEL_GENERATIONS.
        for file_path in files:
            if file_path in protected_paths:
                to_keep.add(file_path)

        for file_path in files:
            if file_path not in to_keep:
                to_delete.append(file_path)

    return to_delete, to_keep


def top_directory_counts(paths: Iterable[Path], limit: int = 10) -> List[Tuple[Path, int]]:
    counts: Dict[Path, int] = defaultdict(int)
    for path in paths:
        counts[path.parent] += 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]


def delete_files(paths: Iterable[Path], checkpoints_root: Path, target_root: Path) -> int:
    deleted = 0
    checkpoints_root = checkpoints_root.resolve()
    target_root = target_root.resolve()

    for path in paths:
        resolved = path.resolve()
        if not is_within(resolved, checkpoints_root):
            raise RuntimeError(f"Refusing to delete outside checkpoints/: {resolved}")
        if not is_within(resolved, target_root):
            raise RuntimeError(f"Refusing to delete outside target root: {resolved}")
        resolved.unlink()
        deleted += 1

    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune checkpoint files in checkpoints/hyperparameter_tuning while preserving "
            "MODEL_GENERATIONS references, first+last per directory, and every 10th checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoints-root",
        default="checkpoints",
        help="Base checkpoints directory (default: checkpoints).",
    )
    parser.add_argument(
        "--target-root",
        default="checkpoints/hyperparameter_tuning",
        help="Directory tree to prune (default: checkpoints/hyperparameter_tuning).",
    )
    parser.add_argument(
        "--model-config",
        default="hex_ai/inference/model_config.py",
        help="Path to model_config.py containing MODEL_GENERATIONS.",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=10,
        help="Keep every Nth checkpoint in each directory (default: 10).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=30,
        help="How many deletion candidates to preview (default: 30).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, script runs in dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.keep_every <= 0:
        print("Error: --keep-every must be >= 1", file=sys.stderr)
        return 2

    checkpoints_root = Path(args.checkpoints_root).resolve()
    target_root = Path(args.target_root).resolve()
    model_config = Path(args.model_config).resolve()

    if not checkpoints_root.exists():
        print(f"Error: checkpoints root does not exist: {checkpoints_root}", file=sys.stderr)
        return 2
    if not target_root.exists():
        print(f"Error: target root does not exist: {target_root}", file=sys.stderr)
        return 2
    if not model_config.exists():
        print(f"Error: model config does not exist: {model_config}", file=sys.stderr)
        return 2

    if not is_within(target_root, checkpoints_root):
        print(
            "Error: --target-root must be inside --checkpoints-root. "
            f"target={target_root}, checkpoints={checkpoints_root}",
            file=sys.stderr,
        )
        return 2

    model_generations = load_model_generations(model_config)
    protected_paths, fuzzy_matches, missing_refs = build_protected_paths(
        model_generations=model_generations,
        checkpoints_root=checkpoints_root,
    )

    grouped = discover_candidate_files(target_root)
    total_candidates = sum(len(files) for files in grouped.values())

    to_delete, to_keep = compute_keep_and_delete(
        grouped_files=grouped,
        protected_paths=protected_paths,
        keep_every=args.keep_every,
    )
    delete_bytes = sum(path.stat().st_size for path in to_delete)

    protected_in_target = {p for p in protected_paths if is_within(p, target_root)}
    print(f"checkpoints root: {checkpoints_root}")
    print(f"target root: {target_root}")
    print(f"model config: {model_config}")
    print(f"mode: {'APPLY (delete)' if args.apply else 'DRY RUN (no deletion)'}")
    print()
    print(f"directories scanned: {len(grouped)}")
    print(f"candidate checkpoint files: {total_candidates}")
    print(f"files kept by rules: {len(to_keep)}")
    print(f"files to delete: {len(to_delete)}")
    print(f"estimated space reclaimed: {format_bytes(delete_bytes)}")
    print()
    print(f"MODEL_GENERATIONS references loaded: {len(protected_paths)} existing files")
    print(f"MODEL_GENERATIONS references inside target root: {len(protected_in_target)}")
    if fuzzy_matches:
        print(f"conservative extension fallbacks matched: {len(fuzzy_matches)}")
    if missing_refs:
        print(f"MODEL_GENERATIONS references not found on disk: {len(missing_refs)}")
        for ref in missing_refs[:10]:
            print(f"  missing: {ref}")
        if len(missing_refs) > 10:
            print(f"  ... and {len(missing_refs) - 10} more")
    print()

    top_dirs = top_directory_counts(to_delete, limit=10)
    if top_dirs:
        print("top directories by deletion count:")
        for directory, count in top_dirs:
            print(f"  {count:5d}  {directory}")
        print()

    if args.preview > 0:
        print(f"deletion preview (first {min(args.preview, len(to_delete))}):")
        for path in to_delete[: args.preview]:
            print(f"  {path}")
        if len(to_delete) > args.preview:
            print(f"  ... and {len(to_delete) - args.preview} more")
        print()

    if not args.apply:
        print("Dry run complete. Re-run with --apply to delete these files.")
        return 0

    deleted = delete_files(to_delete, checkpoints_root=checkpoints_root, target_root=target_root)
    print(f"Deleted {deleted} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
