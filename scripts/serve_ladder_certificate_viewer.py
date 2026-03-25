"""Serve an interactive ladder-certificate viewer for TRMPH games."""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from hex_ai.utils.ladder_templates import (
    build_ladder_certificate_file_summary,
    build_ladder_certificate_viewer_game_payload,
    list_trmph_files_under_root,
    load_hexwiki_generated_ladder_templates,
    load_ladder_template_library,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = WORKSPACE_ROOT / "hex_ai" / "web" / "static_public"


def _resolve_workspace_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Requested path is outside workspace: {raw_path}"
        ) from exc
    return resolved


def _parse_bool_arg(name: str, *, default: bool = False) -> bool:
    raw = request.args.get(name)
    if raw is None:
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "on"}


def create_app(
    *,
    templates,
    include_winnerless_default: bool,
) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_ROOT))

    @app.route("/")
    def index():
        return send_from_directory(STATIC_ROOT, "ladder_certificate_viewer.html")

    @app.route("/api/list-trmph")
    def list_trmph():
        root = request.args.get("root", "data")
        limit = int(request.args.get("limit", "400"))
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        root_path = _resolve_workspace_path(root)
        files = list_trmph_files_under_root(root_path)
        payload = [
            {
                "path": str(path),
                "relative_path": str(path.relative_to(WORKSPACE_ROOT)),
            }
            for path in files[:limit]
        ]
        return jsonify(
            {
                "root": str(root_path),
                "workspace_root": str(WORKSPACE_ROOT),
                "total_found": len(files),
                "files": payload,
            }
        )

    @app.route("/api/file-summary")
    def file_summary():
        path = _resolve_workspace_path(request.args["path"])
        include_winnerless = _parse_bool_arg(
            "include_winnerless",
            default=include_winnerless_default,
        )
        return jsonify(
            build_ladder_certificate_file_summary(
                path,
                include_winnerless=include_winnerless,
            )
        )

    @app.route("/api/game-data")
    def game_data():
        path = _resolve_workspace_path(request.args["path"])
        game_index = int(request.args.get("game_index", "0"))
        include_winnerless = _parse_bool_arg(
            "include_winnerless",
            default=include_winnerless_default,
        )
        return jsonify(
            build_ladder_certificate_viewer_game_payload(
                path,
                game_index=game_index,
                templates=templates,
                include_winnerless=include_winnerless,
            )
        )

    @app.errorhandler(Exception)
    def handle_error(exc: Exception):
        status = 400
        if isinstance(exc, FileNotFoundError):
            status = 404
        return jsonify({"error": str(exc)}), status

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a browser viewer for ladder-certificate annotations.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument(
        "--library-dir",
        default=None,
        help="Optional ladder-template library directory. Defaults to the generated HexWiki corpus.",
    )
    parser.add_argument(
        "--include-winnerless-default",
        action="store_true",
        help="Make winnerless parseable TRMPH lines visible in the viewer by default.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.port <= 0:
        raise ValueError(f"--port must be positive, got {args.port}")

    if args.library_dir:
        templates = load_ladder_template_library(args.library_dir, source_set="custom")
    else:
        templates = load_hexwiki_generated_ladder_templates()

    app = create_app(
        templates=templates,
        include_winnerless_default=bool(args.include_winnerless_default),
    )
    app.run(host=args.host, port=args.port, debug=bool(args.debug))


if __name__ == "__main__":
    main()
