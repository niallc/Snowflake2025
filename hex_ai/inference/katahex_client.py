"""
Client utilities for running a local KataHex GTP engine.

On Hex boards, KataHex GTP coordinates use the same top-left-origin convention
as Snowflake/TRMPH, so row/column conversion is direct.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Set, Tuple

from hex_ai.config import BOARD_SIZE
from hex_ai.enums import Player


_VERTEX_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _index_to_alpha(index: int) -> str:
    """Convert a zero-based column index to a GTP-style alphabet label."""
    if index < 0:
        raise ValueError(f"Column index must be non-negative, got {index}")

    label_chars: List[str] = []
    value = index
    while value >= 0:
        label_chars.append(chr(ord("A") + (value % 26)))
        value = value // 26 - 1
    return "".join(reversed(label_chars))


def _alpha_to_index(label: str) -> int:
    """Convert a GTP-style alphabet label to a zero-based column index."""
    if not label:
        raise ValueError("Column label must be non-empty")

    value = 0
    for char in label.upper():
        if not ("A" <= char <= "Z"):
            raise ValueError(f"Invalid column label: {label!r}")
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def snowflake_rowcol_to_katahex_vertex(
    row: int,
    col: int,
    board_size: int = BOARD_SIZE,
) -> str:
    """
    Convert Snowflake coordinates to a KataHex GTP vertex.

    On Hex boards, both Snowflake/TRMPH and KataHex use ``a1``/``A1`` for the
    top-left corner, so the mapping is direct.
    """
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(
            f"Invalid Snowflake coordinates ({row}, {col}) for board size {board_size}"
        )
    return f"{_index_to_alpha(col)}{row + 1}"


def katahex_vertex_to_snowflake_rowcol(
    vertex: str,
    board_size: int = BOARD_SIZE,
) -> Tuple[int, int]:
    """Convert a KataHex GTP vertex back to Snowflake row/col coordinates."""
    text = (vertex or "").strip()
    match = _VERTEX_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Invalid KataHex vertex: {vertex!r}")

    col = _alpha_to_index(match.group(1))
    gtp_row = int(match.group(2))
    row = gtp_row - 1
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(
            f"KataHex vertex {vertex!r} is out of range for board size {board_size}"
        )
    return row, col


def player_to_katahex_color(player: Player) -> str:
    """Map Snowflake players to KataHex GTP colors."""
    if player == Player.BLUE:
        return "b"
    if player == Player.RED:
        return "w"
    raise ValueError(f"Unsupported player value: {player!r}")


def alternating_player_for_move_index(move_index: int) -> Player:
    """Return the player who makes move number ``move_index`` (0-based)."""
    return Player.BLUE if (move_index % 2) == 0 else Player.RED


def build_katahex_override_config(
    *,
    max_visits: int,
    num_search_threads: int,
    nn_cache_size_power_of_two: int,
    log_dir: str = "",
    extra_override_config: Optional[str] = None,
) -> str:
    """Build a deterministic override-config string for KataHex."""
    if max_visits <= 0:
        raise ValueError(f"max_visits must be positive, got {max_visits}")
    if num_search_threads <= 0:
        raise ValueError(
            f"num_search_threads must be positive, got {num_search_threads}"
        )
    if nn_cache_size_power_of_two < 0:
        raise ValueError(
            "nn_cache_size_power_of_two must be non-negative, "
            f"got {nn_cache_size_power_of_two}"
        )

    parts = [
        f"logDir={log_dir}",
        "logAllGTPCommunication=false",
        "logSearchInfo=false",
        "logToStderr=false",
        "ponderingEnabled=false",
        f"numSearchThreads={num_search_threads}",
        f"maxVisits={max_visits}",
        f"maxPlayouts={max_visits}",
        f"nnCacheSizePowerOfTwo={nn_cache_size_power_of_two}",
        "noResultUtilityForWhite=0.0",
    ]
    if extra_override_config:
        parts.append(extra_override_config.strip())
    return ",".join(parts)


@dataclass
class KataHexPlayer:
    """Lightweight wrapper used by tournament logging helpers."""

    name: str
    engine_path: str
    config_path: str
    model_path: str
    override_config: Optional[str] = None
    policy_only: bool = False
    board_size: int = BOARD_SIZE
    command_timeout: float = 300.0
    startup_timeout: float = 300.0

    def __post_init__(self) -> None:
        self.temperature = None
        self.strategy_type = "katahex_policy" if self.policy_only else "katahex"
        self.config = {
            "override_config": self.override_config,
            "policy_only": self.policy_only,
            "board_size": self.board_size,
        }

    def create_engine(self) -> "KataHexEngine":
        """Create a fresh KataHex GTP engine session."""
        return KataHexEngine(
            engine_path=self.engine_path,
            config_path=self.config_path,
            model_path=self.model_path,
            override_config=self.override_config,
            board_size=self.board_size,
            command_timeout=self.command_timeout,
            startup_timeout=self.startup_timeout,
        )


class KataHexEngine:
    """Persistent GTP subprocess wrapper for KataHex."""

    def __init__(
        self,
        *,
        engine_path: str,
        config_path: str,
        model_path: str,
        override_config: Optional[str] = None,
        board_size: int = BOARD_SIZE,
        command_timeout: float = 300.0,
        startup_timeout: float = 300.0,
    ) -> None:
        self.engine_path = str(Path(engine_path).expanduser())
        self.config_path = str(Path(config_path).expanduser())
        self.model_path = str(Path(model_path).expanduser())
        self.override_config = override_config
        self.board_size = int(board_size)
        self.command_timeout = float(command_timeout)
        self.startup_timeout = float(startup_timeout)

        self.process: Optional[subprocess.Popen[str]] = None
        self._stdout_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stderr_tail: Deque[str] = deque(maxlen=200)
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._command_lock = threading.Lock()
        self._occupied_points: Set[Tuple[int, int]] = set()

    def __enter__(self) -> "KataHexEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Launch the KataHex subprocess and validate the GTP channel."""
        if self.process is not None:
            return

        command = [
            self.engine_path,
            "gtp",
            "-config",
            self.config_path,
            "-model",
            self.model_path,
        ]
        if self.override_config:
            command.extend(["-override-config", self.override_config])

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._drain_stdout,
            name="katahex-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="katahex-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        # Verify the GTP channel early so missing models/configs fail fast.
        self.command("name", timeout=self.startup_timeout)
        self.new_game(board_size=self.board_size)

    def close(self) -> None:
        """Shut down the KataHex subprocess."""
        process = self.process
        if process is None:
            return

        try:
            if process.poll() is None:
                try:
                    self.command("quit", timeout=5.0)
                except Exception:
                    pass
                process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        finally:
            self.process = None

    def command(self, command: str, *, timeout: Optional[float] = None) -> str:
        """Send one GTP command and return its response payload."""
        self.start()
        timeout = self.command_timeout if timeout is None else float(timeout)

        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("KataHex stdin pipe is unavailable")

        with self._command_lock:
            if process.poll() is not None:
                raise RuntimeError(
                    "KataHex process exited before command could be sent.\n"
                    f"{self._format_stderr_tail()}"
                )

            process.stdin.write(command)
            process.stdin.write("\n")
            process.stdin.flush()

            response_lines: List[str] = []
            while True:
                try:
                    raw_line = self._stdout_queue.get(timeout=timeout)
                except queue.Empty as exc:
                    raise TimeoutError(
                        f"Timed out waiting for KataHex response to {command!r}.\n"
                        f"{self._format_stderr_tail()}"
                    ) from exc

                if raw_line is None:
                    raise RuntimeError(
                        "KataHex stdout closed unexpectedly.\n"
                        f"{self._format_stderr_tail()}"
                    )

                line = raw_line.rstrip("\r\n")
                if line == "":
                    break
                response_lines.append(line)

        if not response_lines:
            raise RuntimeError(
                f"KataHex returned an empty GTP response to {command!r}.\n"
                f"{self._format_stderr_tail()}"
            )

        status_line = response_lines[0]
        status = status_line[0]
        payload_first_line = status_line[1:].lstrip()
        payload_lines = [payload_first_line] + response_lines[1:]
        payload = "\n".join(line for line in payload_lines if line).strip()

        if status == "?":
            raise RuntimeError(
                f"KataHex GTP error for {command!r}: {payload or '<no payload>'}\n"
                f"{self._format_stderr_tail()}"
            )
        if status != "=":
            raise RuntimeError(
                f"Unexpected KataHex GTP response to {command!r}: {response_lines!r}\n"
                f"{self._format_stderr_tail()}"
            )

        return payload

    def new_game(self, *, board_size: Optional[int] = None) -> None:
        """Reset the engine to a fresh empty board."""
        actual_board_size = self.board_size if board_size is None else int(board_size)
        self.command(f"boardsize {actual_board_size}")
        self.command("clear_board")
        self._occupied_points.clear()

    def load_moves(
        self,
        moves: Iterable[Tuple[Player, int, int]],
        *,
        board_size: Optional[int] = None,
    ) -> None:
        """Reset the board and replay the provided move sequence."""
        actual_board_size = self.board_size if board_size is None else int(board_size)
        self.new_game(board_size=actual_board_size)
        for player, row, col in moves:
            self.play_move(player, row, col, board_size=actual_board_size)

    def play_move(
        self,
        player: Player,
        row: int,
        col: int,
        *,
        board_size: Optional[int] = None,
    ) -> None:
        """Tell KataHex about an externally chosen move."""
        actual_board_size = self.board_size if board_size is None else int(board_size)
        point = (row, col)
        if point in self._occupied_points:
            raise RuntimeError(f"Attempted to replay occupied move {point} into KataHex")
        color = player_to_katahex_color(player)
        vertex = snowflake_rowcol_to_katahex_vertex(row, col, board_size=actual_board_size)
        self.command(f"play {color} {vertex}")
        self._occupied_points.add(point)

    def genmove(
        self,
        player: Player,
        *,
        board_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Ask KataHex to choose and play a move."""
        actual_board_size = self.board_size if board_size is None else int(board_size)
        color = player_to_katahex_color(player)
        response = self.command(f"genmove {color}")
        lowered = response.lower()
        if lowered in {"pass", "resign", "null"}:
            raise RuntimeError(
                f"KataHex returned unsupported move {response!r} in Hex.\n"
                f"{self._format_stderr_tail()}"
            )
        point = katahex_vertex_to_snowflake_rowcol(response, board_size=actual_board_size)
        self._occupied_points.add(point)
        return point

    def raw_policy_move(
        self,
        player: Player,
        *,
        board_size: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Select and play the argmax legal raw-policy move from KataHex.

        This bypasses MCTS and uses the neural network policy head directly via
        the ``kata-raw-nn`` GTP extension.
        """
        actual_board_size = self.board_size if board_size is None else int(board_size)
        response = self.command("kata-raw-nn 0")
        policy_values = _parse_raw_policy_values(response, board_size=actual_board_size)

        best_index: Optional[int] = None
        best_value: Optional[float] = None
        for index, value in enumerate(policy_values):
            row, col = divmod(index, actual_board_size)
            if (row, col) in self._occupied_points:
                continue
            if value != value:
                continue
            if best_value is None or value > best_value:
                best_index = index
                best_value = value

        if best_index is None:
            raise RuntimeError(
                "KataHex raw policy response did not include any legal moves.\n"
                f"{self._format_stderr_tail()}"
            )

        row, col = divmod(best_index, actual_board_size)
        self.play_move(player, row, col, board_size=actual_board_size)
        return row, col

    def _require_process(self) -> subprocess.Popen[str]:
        if self.process is None:
            raise RuntimeError("KataHex process has not been started")
        return self.process

    def _drain_stdout(self) -> None:
        process = self._require_process()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                self._stdout_queue.put(line)
        finally:
            self._stdout_queue.put(None)

    def _drain_stderr(self) -> None:
        process = self._require_process()
        assert process.stderr is not None
        for line in process.stderr:
            self._stderr_tail.append(line.rstrip("\r\n"))

    def _format_stderr_tail(self) -> str:
        if not self._stderr_tail:
            return "KataHex stderr tail: <empty>"
        tail = "\n".join(self._stderr_tail)
        return f"KataHex stderr tail:\n{tail}"


def _parse_raw_policy_values(response: str, *, board_size: int) -> List[float]:
    """Extract the raw policy vector from a ``kata-raw-nn`` response."""
    tokens = response.split()
    expected_count = board_size * board_size
    try:
        policy_index = tokens.index("policy")
    except ValueError as exc:
        raise RuntimeError(f"KataHex raw policy response missing 'policy' field: {response!r}") from exc

    start = policy_index + 1
    end = start + expected_count
    if len(tokens) < end:
        raise RuntimeError(
            "KataHex raw policy response was truncated: "
            f"expected {expected_count} policy values, got {len(tokens) - start}"
        )

    try:
        return [float(token) for token in tokens[start:end]]
    except ValueError as exc:
        raise RuntimeError(
            "KataHex raw policy response contained a non-float policy value.\n"
            f"Response: {response!r}"
        ) from exc
