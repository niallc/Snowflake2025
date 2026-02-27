"""Typed summary payloads for self-play generation results."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_RED_WIN


@dataclass(frozen=True)
class SelfPlayGenerationSummary:
    """Compact, typed summary for a self-play generation run."""

    num_games: int
    red_wins: int
    blue_wins: int
    trmph_file: Optional[str] = None
    provenance_file: Optional[str] = None

    @classmethod
    def from_games(
        cls,
        games: Iterable[Dict[str, Any]],
        *,
        trmph_file: Optional[str] = None,
        provenance_file: Optional[str] = None,
    ) -> "SelfPlayGenerationSummary":
        """Build summary counts from game records."""
        total_games = 0
        red_wins = 0
        blue_wins = 0
        for game_index, game in enumerate(games):
            winner = game.get("winner")
            if winner == TRMPH_RED_WIN:
                red_wins += 1
            elif winner == TRMPH_BLUE_WIN:
                blue_wins += 1
            else:
                raise ValueError(
                    f"Unexpected winner while summarizing game {game_index}: {winner!r}"
                )
            total_games += 1
        return cls(
            num_games=total_games,
            red_wins=red_wins,
            blue_wins=blue_wins,
            trmph_file=trmph_file,
            provenance_file=provenance_file,
        )

    def with_files(
        self,
        *,
        trmph_file: Optional[str] = None,
        provenance_file: Optional[str] = None,
    ) -> "SelfPlayGenerationSummary":
        """Return a copy with updated output file paths."""
        return SelfPlayGenerationSummary(
            num_games=self.num_games,
            red_wins=self.red_wins,
            blue_wins=self.blue_wins,
            trmph_file=trmph_file if trmph_file is not None else self.trmph_file,
            provenance_file=(
                provenance_file
                if provenance_file is not None
                else self.provenance_file
            ),
        )
